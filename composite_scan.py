#!/usr/bin/env python3
"""
Composite Scanner - nightly end-of-day job.

Scores every stock in the universe 0-100 on three independent blocks
(Trend, Relative Strength, Momentum), averages them into one composite
score, then runs 16 rule-based scans. Results are written as static JSON
that composite.html reads, so the whole thing can live on GitHub Pages.

Data (all free, no API keys):
  * Universe + industries : Wikipedia index tables, iShares holdings as fallback
  * Daily bars            : Yahoo Finance (split/dividend adjusted)
  * Short interest        : Yahoo Finance (twice-monthly exchange data, cached)

Run:  python scripts/composite_scan.py [--force] [--limit N]

Every threshold lives in CFG below. Where the published rule left a term
undefined ("rising", "near support", "contracting range"...), the value in
CFG is this project's interpretation - see README-composite.md.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import math
import os
import re
import sys
import time
from typing import Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
CFG = {
    "benchmark": "IWB",            # iShares Russell 1000 ETF (adjusted = total return)
    "history_period": "2y",
    "min_bars": 260,               # need 12 months + a little for every input
    # --- score definitions ---
    "slope_lookback": 20,          # sessions used to measure 50-day slope
    "rvol_window": 50,             # average-volume window for relative volume
    "ext_base": 20,                # ATR extension is measured from this average
    "atr_len": 14,
    "rsi_len": 14,
    # --- shared scan terms ---
    "rising50_lookback": 10,
    "rising200_lookback": 20,
    "pivot_len": 55,
    "max_ext_atr": 2.75,
    "fresh_window": 10,            # no earlier pivot break in this many sessions
    "contract_recent": 10,         # "contracting": last 10 sessions ...
    "contract_base": 30,           # ... versus the 30 before them
    "contract_range_ratio": 0.85,
    "contract_vol_ratio": 0.90,
    "strong_close": 0.65,          # close in upper 35% of the day's range
    "weak_close": 0.35,
    "leader_ex6": 10.0,            # "prior 6-month leadership": +10 pts vs benchmark
    "leader_high_within": 63,      # ... and a 52-week high in the last 63 sessions
    "pullback_depth_atr": 1.5,     # pulled back at least this far off the 20-day high
    "near_ma_below_atr": 0.5,      # "near" an average: from 0.5 ATR below ...
    "near_ma_above_atr": 1.0,      # ... to 1.0 ATR above
    "controlled_vol": 1.10,        # recent volume <= 1.10x the 50-day average
    "continuation_vol": 1.25,
    "reclaim_rvol": 1.20,
    "reclaim_days_below": 5,       # of the prior 10 closes below the 50-day
    "gap_min": 0.04,
    "event_vol": 1.5,
    "min_group_size": 4,
    "group_stable_accel": -2.0,
    # --- short interest ---
    "si_max_lookups": 250,
    "si_ttl_days": 10,
    # --- safety ---
    "max_day_up": 2.0,             # +200% / -70% in one session = treat as bad data
    "max_day_down": -0.70,
    "min_coverage": 0.50,          # refuse to publish if under half the universe scored
    "chart_bars": 180,
}

SCANS = [
    # id, name, mode, stage, family
    ("el", "Established Leaders", "Flow", "State", "Leadership"),
    ("em", "Emerging Leaders", "Flow + Burst", "Watch", "Leadership"),
    ("br", "Breakout Ready", "Flow + Burst", "Setup", "Breakout"),
    ("fb", "Fresh Breakouts", "Flow + Burst", "Trigger", "Breakout"),
    ("bc", "Breakout Continuation", "Flow", "Trigger", "Continuation"),
    ("cp", "Constructive Pullbacks", "Flow", "Watch", "Pullback"),
    ("pc", "Pullback Confirmed", "Flow", "Trigger", "Pullback"),
    ("lr", "Leadership Reclaim", "Flow", "Trigger", "Reclaim"),
    ("ei", "Event Ignition", "Burst", "Trigger", "Event"),
    ("ed", "Event Drift", "Flow", "Trigger", "Event"),
    ("slg", "Stock Leads Group", "Flow + Burst", "Watch", "Leadership"),
    ("grl", "Group Rotation Leaders", "Flow + Burst", "Watch", "Rotation"),
    ("gcl", "Group-Confirmed Leaders", "Flow", "Overlay", "Group"),
    ("bdr", "Breakdown Ready", "Burst", "Setup", "Breakdown"),
    ("bb", "Bearish Burst", "Burst", "Trigger", "Breakdown"),
    ("sq", "Squeeze Radar", "Burst", "Watch", "Squeeze"),
]

UA = "Mozilla/5.0 (compatible; composite-scanner/1.0; personal EOD research)"
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------
# Indicator helpers (pure numpy)
# --------------------------------------------------------------------------
def scale(x: float, lo: float, hi: float) -> float:
    """Linear map of x from [lo, hi] onto 0-100, clamped."""
    if x is None or not math.isfinite(x):
        return 0.0
    return float(min(100.0, max(0.0, (x - lo) / (hi - lo) * 100.0)))


def rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        cs = np.cumsum(np.insert(x.astype(float), 0, 0.0))
        out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def true_range(h, l, c) -> np.ndarray:
    tr = h - l
    pc = np.roll(c, 1)
    tr[1:] = np.maximum(tr[1:], np.maximum(np.abs(h[1:] - pc[1:]), np.abs(l[1:] - pc[1:])))
    return tr


def wilder(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder smoothing (used by ATR)."""
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    v = float(np.mean(x[:n]))
    out[n - 1] = v
    for k in range(n, len(x)):
        v = (v * (n - 1) + x[k]) / n
        out[k] = v
    return out


def rsi_last(c: np.ndarray, n: int = 14) -> float:
    d = np.diff(c)
    if len(d) < n:
        return float("nan")
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    au, ad = float(up[:n].mean()), float(dn[:n].mean())
    for k in range(n, len(d)):
        au = (au * (n - 1) + up[k]) / n
        ad = (ad * (n - 1) + dn[k]) / n
    if ad == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + au / ad)


def rsi_quality(r: float) -> float:
    """100 inside the healthy 55-70 zone, fading on both sides."""
    if not math.isfinite(r):
        return 0.0
    if r < 30:
        return 0.0
    if r < 55:
        return (r - 30) / 25 * 100
    if r <= 70:
        return 100.0
    if r < 90:
        return 100 - (r - 70) / 20 * 60
    return 40.0


# --------------------------------------------------------------------------
# Per-stock analysis
# --------------------------------------------------------------------------
def analyze(o, h, l, c, v, b) -> Optional[dict]:
    """
    o/h/l/c/v : adjusted daily bars (numpy, oldest first)
    b         : benchmark adjusted close on the same dates
    Returns a dict of scores, raw inputs and scan flags, or None if the
    history is too short to score honestly.
    """
    n = len(c)
    if n < CFG["min_bars"] or not np.all(np.isfinite(c[-CFG["min_bars"]:])):
        return None
    if c[-1] <= 0 or np.any(c[-CFG["min_bars"]:] <= 0):
        return None
    i = n - 1
    o, h, l, c, v, b = (np.asarray(x, dtype=float) for x in (o, h, l, c, v, b))
    # A one-day move this large in adjusted data is almost always a bad
    # split/symbol-change adjustment rather than a real session: do not score it.
    day = c[-252:] / c[-253:-1] - 1
    if np.any(day > CFG["max_day_up"]) or np.any(day < CFG["max_day_down"]):
        return None

    s20, s50, s200 = rolling_mean(c, 20), rolling_mean(c, 50), rolling_mean(c, 200)
    tr = true_range(h.copy(), l, c)
    atr_arr = wilder(tr, CFG["atr_len"])
    atr = atr_arr[i]
    if not (math.isfinite(atr) and atr > 0 and math.isfinite(s200[i])):
        return None
    px = c[i]

    # ---------------- Trend ----------------
    t_above = (int(px > s20[i]) + int(px > s50[i]) + int(px > s200[i])) / 3 * 100
    t_align = 50.0 * (s20[i] > s50[i]) + 50.0 * (s50[i] > s200[i])
    slope50 = s50[i] / s50[i - CFG["slope_lookback"]] - 1
    hi52 = float(np.max(h[i - 251: i + 1]))
    off_high = px / hi52 - 1
    share60 = float(np.mean(c[i - 59: i + 1] > s50[i - 59: i + 1])) * 100
    T = (0.30 * t_above + 0.25 * t_align + 0.20 * scale(slope50, -0.06, 0.08)
         + 0.15 * scale(off_high, -0.30, 0.0) + 0.10 * share60)

    # ---------------- Relative strength ----------------
    def ret(k):
        return px / c[i - k] - 1

    def excess(k):
        return (ret(k) - (b[i] / b[i - k] - 1)) * 100

    ex12, ex6, ex3 = excess(252), excess(126), excess(63)
    RS = (0.45 * scale(ex12, -30, 30) + 0.35 * scale(ex6, -20, 20)
          + 0.20 * scale(ex3, -12, 12))
    rs_accel = ex3 - ex6 / 2          # 3-month excess vs the 6-month pace

    # ---------------- Momentum ----------------
    r1, r3, r6 = ret(21), ret(63), ret(126)
    accel = (r1 - r3 / 3) * 100
    rsi = rsi_last(c[-250:], CFG["rsi_len"])
    w = CFG["rvol_window"]
    avg_vol = float(np.mean(v[i - w: i]))
    rvol = v[i] / avg_vol if avg_vol > 0 else float("nan")
    M = (0.30 * scale(r1, -0.12, 0.12) + 0.25 * scale(r3, -0.20, 0.30)
         + 0.20 * scale(accel, -8, 8) + 0.15 * rsi_quality(rsi)
         + 0.10 * scale(rvol, 0.5, 2.0))

    score = int(round((T + RS + M) / 3))

    # ---------------- Shared scan inputs ----------------
    base = {20: s20, 50: s50}[CFG["ext_base"]]
    ext = (px - base[i]) / atr
    rising50 = s50[i] > s50[i - CFG["rising50_lookback"]]
    rising200 = s200[i] > s200[i - CFG["rising200_lookback"]]
    rng = h[i] - l[i]
    clv = (px - l[i]) / rng if rng > 0 else 0.5
    chg = px / c[i - 1] - 1

    P = CFG["pivot_len"]

    def pivot(d):                      # highest high of the P sessions before d
        return float(np.max(h[d - P: d]))

    def broke(d):
        return c[d] > pivot(d)

    def initial_break(d):              # a break with no break in the prior window
        if not broke(d):
            return False
        return not any(broke(k) for k in range(d - CFG["fresh_window"], d))

    piv = pivot(i)

    cr, cb = CFG["contract_recent"], CFG["contract_base"]
    rng_ratio = float(np.mean(tr[i - cr + 1: i + 1]) / np.mean(tr[i - cr - cb + 1: i - cr + 1]))
    vb = float(np.mean(v[i - cr - cb + 1: i - cr + 1]))
    vol_ratio = float(np.mean(v[i - cr + 1: i + 1]) / vb) if vb > 0 else float("nan")
    contracting = (rng_ratio <= CFG["contract_range_ratio"]
                   and vol_ratio <= CFG["contract_vol_ratio"])

    hi_pos = int(np.argmax(h[i - 251: i + 1]))          # 0..251, 251 = today
    leader6 = ex6 >= CFG["leader_ex6"] and (251 - hi_pos) <= CFG["leader_high_within"]

    hi20 = float(np.max(h[i - 20: i + 1]))
    low20 = float(np.min(l[i - 20: i]))                 # prior 20 sessions
    low50 = float(np.min(l[i - 50: i]))

    def near(level, price):
        d = (price - level) / atr
        return -CFG["near_ma_below_atr"] <= d <= CFG["near_ma_above_atr"]

    scans: List[str] = []
    note: dict = {}

    above_rising = px > s50[i] and px > s200[i] and rising50 and rising200

    # 1. Established Leaders
    established = (T >= 80 and RS >= 80 and above_rising
                   and off_high >= -0.15 and ext <= CFG["max_ext_atr"])
    if established:
        scans.append("el")

    # 2. Emerging Leaders
    if (not established and M >= 80 and RS >= 65 and rs_accel >= 4 and rising50):
        scans.append("em")

    # 3. Breakout Ready
    to_pivot = px / piv - 1
    if (T >= 60 and RS >= 65 and above_rising and -0.03 <= to_pivot <= 0 and contracting):
        scans.append("br")

    # 4. Fresh Breakouts
    if (T >= 60 and RS >= 65 and initial_break(i) and 0 < to_pivot <= 0.05
            and rvol >= 1.35 and clv >= CFG["strong_close"] and ext <= CFG["max_ext_atr"]):
        scans.append("fb")

    # 5. Breakout Continuation
    if T >= 70 and RS >= 70 and px > h[i - 1]:
        for d in range(i - 2, i - 11, -1):
            if initial_break(d):
                p0 = pivot(d)
                held = float(np.min(c[d: i + 1])) >= p0
                calm = float(np.mean(v[d + 1: i + 1])) <= CFG["continuation_vol"] * avg_vol
                paused = bool(np.any(c[d + 1: i] < c[d: i - 1]))   # at least one down close
                if held and calm and paused:
                    scans.append("bc")
                    note["bc"] = {"pivot": round(p0, 2), "ago": i - d}
                break

    # 6 / 7. Pullbacks
    recent5_vol_ok = float(np.mean(v[i - 4: i + 1])) <= CFG["controlled_vol"] * avg_vol
    depth_close = (hi20 - px) / atr
    low5 = float(np.min(l[i - 4: i + 1]))
    low3 = float(np.min(l[i - 2: i + 1]))
    depth_low = (hi20 - low5) / atr
    if T >= 70 and RS >= 70 and leader6 and rising200:
        if (depth_close >= CFG["pullback_depth_atr"] and (near(s20[i], px) or near(s50[i], px))
                and px >= s50[i] - CFG["near_ma_below_atr"] * atr and recent5_vol_ok):
            scans.append("cp")
        touched = (abs(low3 - s20[i]) <= 0.75 * atr) or (abs(low3 - s50[i]) <= 0.75 * atr)
        if (depth_low >= CFG["pullback_depth_atr"] and touched and px > h[i - 1]
                and clv >= CFG["strong_close"]):
            scans.append("pc")
            note["pc"] = {"support_low": round(low3, 2)}

    # 8. Leadership Reclaim
    below_cnt = int(np.sum(c[i - 10: i] < s50[i - 10: i]))
    if (T >= 55 and RS >= 55 and r6 > 0 and px > s200[i] and rising200
            and px > s50[i] and c[i - 1] <= s50[i - 1]
            and below_cnt >= CFG["reclaim_days_below"]
            and rvol >= CFG["reclaim_rvol"] and rs_accel > 0):
        scans.append("lr")
        note["lr"] = {"reclaim_low": round(float(l[i]), 2)}

    # 9 / 10. Events (most recent qualifying gap in the last 20 sessions)
    ev = None
    for d in range(i, i - 21, -1):
        gap = o[d] / c[d - 1] - 1
        if gap >= CFG["gap_min"]:
            av = float(np.mean(v[d - w: d]))
            if av > 0 and v[d] / av >= CFG["event_vol"]:
                kept = (px - c[d - 1]) / (o[d] - c[d - 1])
                ev = {"ago": i - d, "gap": gap, "kept": kept,
                      "level": float(o[d]), "evol": float(v[d] / av)}
                break
    if ev:
        if ev["ago"] <= 2 and ev["kept"] >= 0.70 and M >= 65 and RS >= 65:
            scans.append("ei")
        if 4 <= ev["ago"] <= 20 and ev["kept"] >= 0.80 and T >= 60 and RS >= 70:
            scans.append("ed")

    # 14. Breakdown Ready
    if (RS <= 45 and M <= 50 and px < s20[i] and low20 <= px <= low20 * 1.05
            and rs_accel < 0 and contracting):
        scans.append("bdr")

    # 15. Bearish Burst
    def fresh_low_break(win):
        if not c[i] < np.min(l[i - win: i]):
            return False
        return not any(c[k] < np.min(l[k - win: k]) for k in range(i - 5, i))

    tr_atr = tr[i] / atr_arr[i - 1] if atr_arr[i - 1] > 0 else float("nan")
    if ((fresh_low_break(20) or fresh_low_break(50)) and RS <= 55 and M <= 45
            and -0.10 <= chg <= -0.015 and 1.15 <= tr_atr <= 3.5
            and rvol >= 1.25 and clv <= CFG["weak_close"]):
        scans.append("bb")

    # Price side of Squeeze Radar (short interest is added later)
    squeeze_px = (M >= 60 and RS >= 55 and px > s20[i]
                  and px >= hi20 * 0.92 and rs_accel >= 0)

    return {
        "score": score, "T": T, "RS": RS, "M": M,
        "close": px, "chg": chg, "rs_accel": rs_accel, "rvol": rvol, "ext": ext,
        "off_high": off_high, "atr": atr, "atr_pct": atr / px,
        "dvol": float(np.mean(c[i - 19: i + 1] * v[i - 19: i + 1])),
        "s20": s20[i], "s50": s50[i], "s200": s200[i],
        "pivot": piv, "low20": low20, "low50": low50, "hi52": hi52,
        "rsi": rsi, "r1": r1, "r3": r3, "r6": r6,
        "ex12": ex12, "ex6": ex6, "ex3": ex3,
        "slope50": slope50, "share60": share60, "accel": accel,
        "above50": bool(px > s50[i]), "squeeze_px": bool(squeeze_px),
        "ev": ev, "scans": scans, "note": note,
    }


# --------------------------------------------------------------------------
# Group (industry) pass
# --------------------------------------------------------------------------
def pct_rank(values: List[float]) -> List[float]:
    """Percentile rank 0-100 of each value within the list (100 = highest)."""
    n = len(values)
    if n == 1:
        return [50.0]
    order = np.argsort(np.argsort(np.asarray(values, dtype=float), kind="stable"), kind="stable")
    return [float(r) / (n - 1) * 100 for r in order]


def group_pass(stocks: Dict[str, dict]) -> List[dict]:
    groups: Dict[str, List[str]] = {}
    for sym, m in stocks.items():
        key = m.get("group")
        if key:
            groups.setdefault(key, []).append(sym)

    rows = []
    for key, syms in groups.items():
        if len(syms) < CFG["min_group_size"]:
            continue
        rs = [stocks[s]["RS"] for s in syms]
        acc = [stocks[s]["rs_accel"] for s in syms]
        rows.append({
            "name": key, "n": len(syms),
            "rs": float(np.median(rs)), "accel": float(np.median(acc)),
            "improving": float(np.mean([a > 0 for a in acc])),
            "score": float(np.median([stocks[s]["score"] for s in syms])),
            "syms": syms,
        })
    if not rows:
        return []
    for row, p in zip(rows, pct_rank([r["rs"] for r in rows])):
        row["pct"] = p

    for g in rows:
        peers = pct_rank([stocks[s]["RS"] for s in g["syms"]])
        for s, pp in zip(g["syms"], peers):
            m = stocks[s]
            m["peer_pct"], m["ind_pct"] = pp, g["pct"]
            mid = 25 <= g["pct"] < 75
            # 11. Stock Leads Group
            if (m["T"] >= 70 and m["RS"] >= 85 and m["rs_accel"] > 0 and pp >= 85
                    and m["RS"] - g["rs"] >= 15 and mid
                    and g["accel"] >= CFG["group_stable_accel"]):
                m["scans"].append("slg")
            # 12. Group Rotation Leaders
            if (35 <= g["pct"] < 75 and g["accel"] > 0 and g["improving"] >= 0.5
                    and pp >= 70 and m["RS"] >= 65 and m["M"] >= 65):
                m["scans"].append("grl")
            # 13. Group-Confirmed Leaders
            if (m["T"] >= 65 and m["RS"] >= 75 and m["above50"]
                    and g["pct"] >= 75 and pp >= 70):
                m["scans"].append("gcl")
    rows.sort(key=lambda r: -r["pct"])
    return rows


def squeeze_pass(stocks: Dict[str, dict], si: Dict[str, dict]) -> None:
    for sym, m in stocks.items():
        rec = si.get(sym)
        if not rec:
            continue
        m["si"], m["dtc"] = rec.get("pct"), rec.get("dtc")
        if (m["squeeze_px"] and (rec.get("pct") or 0) >= 8.0 and (rec.get("dtc") or 0) >= 2.5):
            m["scans"].append("sq")


# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------
WIKI = {
    "R1000": ("https://en.wikipedia.org/wiki/Russell_1000_Index", 800),
    "SP500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", 450),
    "SP400": ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", 350),
    "SP600": ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", 500),
}
_ISH = "https://www.ishares.com/us/products/{}/1467271812596.ajax?fileType=csv&fileName={}_holdings&dataType=fund"
ISHARES = {
    "R1000": _ISH.format("239707/ishares-russell-1000-etf", "IWB"),
    "SP500": _ISH.format("239726/ishares-core-sp-500-etf", "IVV"),
    "SP400": _ISH.format("239763/ishares-core-sp-midcap-etf", "IJH"),
    "SP600": _ISH.format("239774/ishares-core-sp-smallcap-etf", "IJR"),
}
SHARE_CLASS = {"BRKB": "BRK-B", "BFB": "BF-B", "BFA": "BF-A", "LENB": "LEN-B", "HEIA": "HEI-A",
               "UHALB": "UHAL-B", "MOGA": "MOG-A", "CWENA": "CWEN-A", "GEFB": "GEF-B"}
SYM_RE = re.compile(r"^[A-Z]{1,5}(-[A-Z]{1,2})?$")


def norm_symbol(s) -> Optional[str]:
    s = str(s).strip().upper().replace(".", "-").replace("/", "-")
    return s if SYM_RE.match(s) else None


def _col(cols, *needles):
    for needle in needles:
        for c in cols:
            if needle in str(c).lower():
                return c
    return None


def parse_wiki_tables(html: str, min_rows: int) -> List[dict]:
    import pandas as pd
    for t in pd.read_html(io.StringIO(html)):
        if hasattr(t.columns, "levels"):
            t.columns = [" ".join(str(x) for x in col) for col in t.columns]
        sym_c = _col(t.columns, "symbol", "ticker")
        if sym_c is None or len(t) < min_rows:
            continue
        name_c = _col(t.columns, "security", "company", "name")
        ind_c = _col(t.columns, "sub-industry", "sub industry", "industry")
        sec_c = next((c for c in t.columns if "sector" in str(c).lower() and c != ind_c), None)
        out = []
        for _, r in t.iterrows():
            sym = norm_symbol(r[sym_c])
            if not sym:
                continue

            def val(cn):
                x = r[cn] if cn is not None else None
                return None if x is None or (isinstance(x, float) and math.isnan(x)) else str(x).strip()
            out.append({"s": sym, "n": val(name_c), "sec": val(sec_c), "ind": val(ind_c)})
        if len(out) >= min_rows:
            return out
    return []


def parse_ishares_csv(text: str) -> List[dict]:
    lines = text.splitlines()
    start = next((k for k, ln in enumerate(lines) if ln.startswith("Ticker,")), None)
    if start is None:
        return []
    out = []
    for r in csv.DictReader(lines[start:]):
        if (r.get("Asset Class") or "").strip() != "Equity":
            continue
        raw = (r.get("Ticker") or "").strip().upper()
        sym = SHARE_CLASS.get(raw) or norm_symbol(raw)
        if sym:
            out.append({"s": sym, "n": (r.get("Name") or "").strip().title() or None,
                        "sec": (r.get("Sector") or "").strip() or None, "ind": None})
    return out


def http_get(url: str, ua: str = UA, timeout: int = 30) -> str:
    import requests
    last = None
    for attempt in range(3):
        try:
            resp = requests.get(url, headers={"User-Agent": ua}, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:      # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET failed: {url} ({last})")


def build_universe() -> List[dict]:
    merged: Dict[str, dict] = {}
    for idx, (url, min_rows) in WIKI.items():
        rows: List[dict] = []
        try:
            rows = parse_wiki_tables(http_get(url), min_rows)
            log(f"  {idx}: {len(rows)} names from Wikipedia")
        except Exception as e:      # noqa: BLE001
            log(f"  {idx}: Wikipedia failed ({e})")
        if not rows:
            try:
                rows = parse_ishares_csv(http_get(ISHARES[idx], BROWSER_UA))
                log(f"  {idx}: {len(rows)} names from iShares holdings")
            except Exception as e:  # noqa: BLE001
                log(f"  {idx}: iShares failed ({e})")
        for r in rows:
            cur = merged.setdefault(r["s"], {"s": r["s"], "n": None, "sec": None, "ind": None, "idx": []})
            for k in ("n", "sec", "ind"):
                if not cur[k] and r.get(k):
                    cur[k] = r[k]
            if idx not in cur["idx"]:
                cur["idx"].append(idx)
    return sorted(merged.values(), key=lambda r: r["s"])


def load_universe(out_dir: str, max_age_days: int = 7) -> List[dict]:
    path = os.path.join(out_dir, "universe.json")
    cached = None
    if os.path.exists(path):
        with open(path) as f:
            cached = json.load(f)
        age = (dt.date.today() - dt.date.fromisoformat(cached["built"])).days
        if age < max_age_days and len(cached["stocks"]) > 500:
            log(f"Universe: {len(cached['stocks'])} names (cached {cached['built']})")
            return cached["stocks"]
    log("Universe: rebuilding index membership")
    stocks = build_universe()
    if len(stocks) >= 1000:
        with open(path, "w") as f:
            json.dump({"built": dt.date.today().isoformat(), "stocks": stocks}, f, separators=(",", ":"))
        log(f"Universe: {len(stocks)} names")
        return stocks
    if cached:
        log(f"Universe: rebuild came back short ({len(stocks)}), keeping cached list")
        return cached["stocks"]
    raise SystemExit(f"Could not build a universe (only {len(stocks)} names) and no cached copy exists.")


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------
def _frame_to_bars(df) -> Optional[dict]:
    df = df.dropna(subset=["Close"])
    if len(df) == 0:
        return None
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "o": df["Open"].to_numpy(float), "h": df["High"].to_numpy(float),
        "l": df["Low"].to_numpy(float), "c": df["Close"].to_numpy(float),
        "v": np.nan_to_num(df["Volume"].to_numpy(float)),
    }


def fetch_yf(symbols: List[str], batch: int = 100) -> Dict[str, dict]:
    import yfinance as yf
    out: Dict[str, dict] = {}
    for k in range(0, len(symbols), batch):
        chunk = symbols[k: k + batch]
        df = None
        for attempt in range(3):
            try:
                df = yf.download(chunk, period=CFG["history_period"], interval="1d",
                                 auto_adjust=True, group_by="ticker", threads=True,
                                 progress=False, timeout=30)
                if df is not None and len(df):
                    break
            except Exception as e:  # noqa: BLE001
                log(f"  batch {k // batch + 1}: {e}")
            time.sleep(5 * (attempt + 1))
        if df is None or not len(df):
            continue
        multi = hasattr(df.columns, "levels")
        for sym in chunk:
            try:
                sub = df[sym] if multi else df
                bars = _frame_to_bars(sub)
                if bars:
                    out[sym] = bars
            except Exception:       # noqa: BLE001
                pass
        log(f"  prices: {min(k + batch, len(symbols))}/{len(symbols)} requested, {len(out)} received")
        time.sleep(1.0)
    return out


def fetch_direct(sym: str) -> Optional[dict]:
    """Fallback: Yahoo chart endpoint, adjusted here by the adjclose ratio."""
    import requests
    url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?range={CFG['history_period']}&interval=1d&events=div%2Csplits")
    try:
        j = requests.get(url, headers={"User-Agent": BROWSER_UA}, timeout=20).json()
        res = j["chart"]["result"][0]
        off = res["meta"].get("gmtoffset", 0)
        q = res["indicators"]["quote"][0]
        adj = res["indicators"]["adjclose"][0]["adjclose"]
        rows = []
        for t, o, h, l, c, v, a in zip(res["timestamp"], q["open"], q["high"], q["low"],
                                       q["close"], q["volume"], adj):
            if None in (o, h, l, c, a) or c <= 0:
                continue
            f = a / c
            day = dt.datetime.fromtimestamp(t + off, dt.timezone.utc).strftime("%Y-%m-%d")
            rows.append((day, o * f, h * f, l * f, a, v or 0))
        if not rows:
            return None
        cols = list(zip(*rows))
        return {"dates": list(cols[0]), "o": np.array(cols[1], float), "h": np.array(cols[2], float),
                "l": np.array(cols[3], float), "c": np.array(cols[4], float), "v": np.array(cols[5], float)}
    except Exception:               # noqa: BLE001
        return None


def fetch_prices(symbols: List[str]) -> Dict[str, dict]:
    out = fetch_yf(symbols)
    missing = [s for s in symbols if s not in out]
    if missing and len(missing) <= 400:
        log(f"  retrying {len(missing)} missing symbols one by one")
        fails = 0
        for s in missing:
            bars = fetch_direct(s)
            if bars:
                out[s], fails = bars, 0
            else:
                fails += 1
                if fails >= 25:
                    log("  fallback endpoint not answering, stopping retries")
                    break
            time.sleep(0.25)
    return out


def align(bars: dict, bench: dict) -> Optional[dict]:
    """Keep only sessions the stock and the benchmark share, in order."""
    bmap = bench["_map"]
    keep = [k for k, d in enumerate(bars["dates"]) if d in bmap]
    if not keep:
        return None
    idx = np.array(keep)
    out = {key: bars[key][idx] for key in ("o", "h", "l", "c", "v")}
    out["dates"] = [bars["dates"][k] for k in keep]
    out["b"] = np.array([bench["c"][bmap[d]] for d in out["dates"]])
    return out


# --------------------------------------------------------------------------
# Short interest (cached; only looked up for stocks that pass the price side)
# --------------------------------------------------------------------------
def fetch_short_interest(sym: str) -> Optional[dict]:
    import yfinance as yf
    info = yf.Ticker(sym).info or {}
    pct, dtc = info.get("shortPercentOfFloat"), info.get("shortRatio")
    if pct is None and dtc is None:
        return None
    return {"pct": round(pct * 100, 2) if pct is not None else None,
            "dtc": round(float(dtc), 2) if dtc is not None else None}


def update_short_interest(out_dir: str, candidates: List[str], today: str) -> Dict[str, dict]:
    path = os.path.join(out_dir, "short_interest.json")
    cache: Dict[str, dict] = {}
    if os.path.exists(path):
        with open(path) as f:
            cache = json.load(f)
    t0 = dt.date.fromisoformat(today)

    def stale(s):
        rec = cache.get(s)
        return not rec or (t0 - dt.date.fromisoformat(rec["on"])).days >= CFG["si_ttl_days"]

    todo = [s for s in candidates if stale(s)][: CFG["si_max_lookups"]]
    log(f"Short interest: {len(candidates)} candidates, {len(todo)} lookups")
    fails = 0
    for s in todo:
        try:
            rec = fetch_short_interest(s)
            cache[s] = dict(rec or {"pct": None, "dtc": None}, on=today)
            fails = 0
        except Exception:           # noqa: BLE001
            fails += 1
            if fails >= 10:
                log("  short-interest lookups failing, stopping for today")
                break
        time.sleep(0.35)
    with open(path, "w") as f:
        json.dump(cache, f, separators=(",", ":"), sort_keys=True)
    return cache


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def rnd(x, nd=1):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return round(x, nd)


COLS = ["sym", "name", "sec", "ind", "close", "chg", "score", "T", "RS", "M", "dScore",
        "rsAccel", "rvol", "ext", "offHigh", "atrPct", "dvolM", "s20", "s50", "s200",
        "pivot", "low20", "hi52", "atr", "rsi", "r1", "r3", "r6", "ex12", "ex6", "ex3",
        "slope50", "share60", "accel", "peerPct", "indPct", "si", "dtc", "ev",
        "scans", "newScans", "note"]


def stock_row(sym: str, m: dict, prev: dict) -> list:
    p = prev.get(sym) or {}
    ev = m.get("ev")
    return [
        sym, m.get("name"), m.get("sec"), m.get("ind"),
        rnd(m["close"], 2), rnd(m["chg"] * 100, 2), m["score"],
        rnd(m["T"]), rnd(m["RS"]), rnd(m["M"]),
        (m["score"] - p["score"]) if "score" in p else None,
        rnd(m["rs_accel"]), rnd(m["rvol"], 2), rnd(m["ext"], 2), rnd(m["off_high"] * 100),
        rnd(m["atr_pct"] * 100, 2), rnd(m["dvol"] / 1e6, 1),
        rnd(m["s20"], 2), rnd(m["s50"], 2), rnd(m["s200"], 2),
        rnd(m["pivot"], 2), rnd(m["low20"], 2), rnd(m["hi52"], 2), rnd(m["atr"], 2),
        rnd(m["rsi"]), rnd(m["r1"] * 100), rnd(m["r3"] * 100), rnd(m["r6"] * 100),
        rnd(m["ex12"]), rnd(m["ex6"]), rnd(m["ex3"]),
        rnd(m["slope50"] * 100, 2), rnd(m["share60"], 0), rnd(m["accel"]),
        rnd(m.get("peer_pct"), 0), rnd(m.get("ind_pct"), 0),
        m.get("si"), m.get("dtc"),
        ([ev["ago"], rnd(ev["gap"] * 100), rnd(ev["kept"] * 100, 0), rnd(ev["level"], 2),
          rnd(ev["evol"], 2)] if ev else None),
        m["scans"],
        [s for s in m["scans"] if "scans" in p and s not in p["scans"]],
        m.get("note") or None,
    ]


def load_previous(out_dir: str) -> tuple:
    path = os.path.join(out_dir, "latest.json")
    if not os.path.exists(path):
        return None, {}
    try:
        with open(path) as f:
            j = json.load(f)
        ci = {c: k for k, c in enumerate(j["cols"])}
        prev = {r[ci["sym"]]: {"score": r[ci["score"]], "scans": r[ci["scans"]]} for r in j["rows"]}
        return j.get("asOf"), prev
    except Exception:               # noqa: BLE001
        return None, {}


def write_history(out_dir: str, asof: str, stocks: Dict[str, dict]) -> None:
    hdir = os.path.join(out_dir, "history")
    os.makedirs(hdir, exist_ok=True)
    with open(os.path.join(hdir, f"{asof}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sym", "score", "T", "RS", "M", "close", "scans"])
        for sym in sorted(stocks):
            m = stocks[sym]
            w.writerow([sym, m["score"], rnd(m["T"]), rnd(m["RS"]), rnd(m["M"]),
                        rnd(m["close"], 4), "|".join(m["scans"])])


BUCKETS = [("80-100", 80, 100), ("60-79", 60, 79), ("40-59", 40, 59), ("0-39", 0, 39)]
HORIZONS = [21, 63, 126]


def build_validation(out_dir: str, series: Dict[str, dict]) -> dict:
    """
    Forward validation from stored snapshots only - never a backtest.
    For each stored day, measure what each stock did versus the benchmark
    over the following 21 / 63 / 126 sessions, once those sessions exist.
    """
    hdir = os.path.join(out_dir, "history")
    files = sorted(f for f in os.listdir(hdir) if f.endswith(".csv")) if os.path.isdir(hdir) else []
    acc = {hz: {"b": {b[0]: [] for b in BUCKETS}, "s": {s[0]: [] for s in SCANS}, "days": set()}
           for hz in HORIZONS}
    pos_cache: Dict[str, dict] = {}
    for fn in files:
        day = fn[:-4]
        with open(os.path.join(hdir, fn)) as f:
            for r in csv.DictReader(f):
                ser = series.get(r["sym"])
                if not ser:
                    continue
                pos = pos_cache.get(r["sym"])
                if pos is None:
                    pos = pos_cache[r["sym"]] = {d: k for k, d in enumerate(ser["dates"])}
                k = pos.get(day)
                if k is None:
                    continue
                sc = int(r["score"])
                for hz in HORIZONS:
                    if k + hz >= len(ser["c"]):
                        continue
                    ex = (ser["c"][k + hz] / ser["c"][k] - ser["b"][k + hz] / ser["b"][k]) * 100
                    acc[hz]["days"].add(day)
                    for label, lo, hi in BUCKETS:
                        if lo <= sc <= hi:
                            acc[hz]["b"][label].append(ex)
                    for sid in filter(None, r["scans"].split("|")):
                        if sid in acc[hz]["s"]:
                            acc[hz]["s"][sid].append(ex)

    def stats(xs):
        if not xs:
            return None
        a = np.array(xs)
        return {"n": int(len(a)), "mean": rnd(a.mean(), 2), "median": rnd(np.median(a), 2),
                "win": rnd(float(np.mean(a > 0)) * 100, 0)}

    return {
        "snapshots": len(files),
        "first": files[0][:-4] if files else None,
        "horizons": {str(hz): {"days": len(acc[hz]["days"]),
                               "buckets": {k: stats(v) for k, v in acc[hz]["b"].items()},
                               "scans": {k: stats(v) for k, v in acc[hz]["s"].items()}}
                     for hz in HORIZONS},
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run(out_dir: str, force: bool = False, limit: Optional[int] = None) -> int:
    os.makedirs(out_dir, exist_ok=True)
    universe = load_universe(out_dir)
    if limit:
        universe = universe[:limit]
    meta = {u["s"]: u for u in universe}
    bench_sym = CFG["benchmark"]

    # Benchmark first: it tells us the latest completed session, so a repeat
    # run on the same day (or a holiday) can stop before fetching 1,900 symbols.
    bench = fetch_prices([bench_sym]).get(bench_sym)
    if not bench or len(bench["c"]) < CFG["min_bars"]:
        raise SystemExit("Benchmark history unavailable - not publishing.")
    bench["_map"] = {d: k for k, d in enumerate(bench["dates"])}
    asof = bench["dates"][-1]

    prev_asof, prev = load_previous(out_dir)
    if prev_asof == asof and not force:
        log(f"Already published for {asof}; nothing to do.")
        return 0

    log(f"Session {asof}: fetching daily bars for {len(meta)} stocks")
    prices = fetch_prices([s for s in meta if s != bench_sym])

    stocks: Dict[str, dict] = {}
    series: Dict[str, dict] = {}
    skipped = {"no_data": 0, "stale": 0, "unscorable": 0}
    for sym, u in meta.items():
        bars = prices.get(sym)
        if not bars:
            skipped["no_data"] += 1
            continue
        if bars["dates"][-1] != asof:
            skipped["stale"] += 1
            continue
        a = align(bars, bench)
        m = analyze(a["o"], a["h"], a["l"], a["c"], a["v"], a["b"]) if a else None
        if m is None:
            skipped["unscorable"] += 1
            continue
        m.update(name=u.get("n"), sec=u.get("sec"), ind=u.get("ind"),
                 group=u.get("ind") or u.get("sec"))
        stocks[sym] = m
        series[sym] = {"dates": a["dates"], "c": a["c"], "b": a["b"]}

    coverage = len(stocks) / max(1, len(meta))
    log(f"Scored {len(stocks)}/{len(meta)} ({coverage:.0%}); skipped {skipped}")
    if coverage < CFG["min_coverage"]:
        raise SystemExit("Coverage too low - data source problem. Not publishing.")

    groups = group_pass(stocks)

    si: Dict[str, dict] = {}
    try:
        cands = sorted((s for s, m in stocks.items() if m["squeeze_px"]),
                       key=lambda s: -stocks[s]["M"])
        si = update_short_interest(out_dir, cands, asof)
    except Exception as e:          # noqa: BLE001
        log(f"Short interest unavailable today ({e})")
    squeeze_pass(stocks, si)

    order = {s[0]: k for k, s in enumerate(SCANS)}
    for m in stocks.values():
        m["scans"] = sorted(set(m["scans"]), key=order.get)

    rows = [stock_row(s, stocks[s], prev) for s in sorted(stocks, key=lambda s: -stocks[s]["score"])]
    counts = {sid: sum(1 for m in stocks.values() if sid in m["scans"]) for sid, *_ in SCANS}

    write_history(out_dir, asof, stocks)
    validation = build_validation(out_dir, series)

    latest = {
        "asOf": asof,
        "generatedAt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark": bench_sym,
        "universe": len(meta), "scored": len(stocks), "skipped": skipped,
        "shortInterestCovered": sum(1 for m in stocks.values() if m.get("si") is not None),
        "scans": [{"id": s[0], "name": s[1], "mode": s[2], "stage": s[3], "family": s[4],
                   "count": counts[s[0]]} for s in SCANS],
        "groups": [{"name": g["name"], "n": g["n"], "rs": rnd(g["rs"]), "accel": rnd(g["accel"]),
                    "improving": rnd(g["improving"] * 100, 0), "pct": rnd(g["pct"], 0),
                    "score": rnd(g["score"], 0)} for g in groups],
        "validation": validation,
        "cols": COLS, "rows": rows,
    }
    with open(os.path.join(out_dir, "latest.json"), "w") as f:
        json.dump(latest, f, separators=(",", ":"), allow_nan=False)

    nb = CFG["chart_bars"]
    charts = {"asOf": asof, "bars": nb,
              "c": {s: [rnd(x, 2) for x in series[s]["c"][-nb:]]
                    for s, m in stocks.items() if m["scans"] or m["score"] >= 70}}
    with open(os.path.join(out_dir, "charts.json"), "w") as f:
        json.dump(charts, f, separators=(",", ":"), allow_nan=False)

    log(f"Published {asof}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/composite")
    ap.add_argument("--force", action="store_true", help="re-run even if this session is already published")
    ap.add_argument("--limit", type=int, default=None, help="only the first N symbols (testing)")
    args = ap.parse_args()
    sys.exit(run(args.out, force=args.force, limit=args.limit))
