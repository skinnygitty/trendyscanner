#!/usr/bin/env python3
"""
Composite Scanner - nightly end-of-day job.

Scores every stock in the universe 0-100 on three independent blocks
(Trend, Relative Strength, Momentum), averages them into one composite
score, then runs 16 rule-based scans. Results are written as static JSON
that composite.html reads, so the whole thing can live on GitHub Pages.

Data (all free, no API keys):
  * Universe  : iShares Russell 1000 ETF holdings; Wikipedia S&P 500/400/600 tables
  * Daily bars: Yahoo Finance (split-adjusted; dividends NOT added back)
  * Profiles  : Yahoo Finance industry, market cap, short interest (cached)

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
    "benchmark": "IWB",            # iShares Russell 1000 ETF
    "history_period": "2y",
    "min_bars": 260,               # need 12 months + a little for every input
    # ---- Scores. Fitted to the original's full export for the 2026-10-06 close:
    # Trend 96% identical (rest are corporate-action data differences), RS 97%,
    # Momentum 99.9% when fed the original's own inputs.
    "integer_scores": True,        # round Trend / RS / Momentum first, then average
    "dividend_adjusted": False,    # returns exclude dividends for stocks and benchmark alike
    "slope_lookback": 20,          # sessions used to measure 50-day slope
    "align_partial": 60,           # alignment: 100 if 20>50>200, 60 if only one pair holds, else 0
    "rvol_window": 50,             # relative volume: today / average of the last 50 sessions (today included)
    "rsi_len": 14,
    "rsi_method": "sma",           # simple-average RSI over the last 14 changes ("wilder" = classic)
    "rsi_zero_high": 80,           # RSI quality: 100 from 55 to 70, fading to 0 at 80 ...
    "rsi_zero_low": 30,            # ... and to 0 at 30
    "atr_len": 14,
    "atr_method": "sma",           # plain 14-session average of true range ("wilder" = smoothed)
    # ---- Scan terms. "fitted" = reproduces the original's list on that date;
    # "guess" = no evidence yet either way.
    "ext_ema": 21,                 # fitted: extension = (close - 21-day EMA) / ATR
    "max_ext_atr": 2.75,
    "rising50_lookback": 20,       # fitted
    "rising200_lookback": 20,
    "em_rs_lookback": 63,          # Emerging Leaders: the panel says "3-month RS acceleration"
    "rs_change_lookback": 21,      # fitted: "RS acceleration" / "falling RS" = RS today - RS 21 sessions ago
    "pivot_len": 55,
    "pivot_min_age": 5,            # guess: the pivot high must be at least 5 sessions old
    "fresh_window": 10,            # guess: no earlier pivot break in this many sessions
    "tight_range_ratio": 0.85,     # Breakout Ready: 10-day range <= 85% of 50-day ...
    "tight_vol_ratio": 0.90,       # ... and 10-day volume <= 90% of 50-day (fits 1 of 1; thin evidence)
    "strong_close": 0.65,          # close in upper 35% of the day's range
    "weak_close": 0.35,
    "near_20_pct": 2.0,            # fitted: "near the 20-day" = within 2% of it
    "near_50_pct": 2.5,            # fitted: "near the 50-day" = within 2.5% of it
    "controlled_vol": 1.15,        # fitted: 10-day average volume <= 1.15x the 50-day
    "continuation_vol": 1.25,      # guess
    "reclaim_rvol": 1.20,          # guess
    "reclaim_days_below": 5,       # guess: of the prior 10 closes below the 50-day
    "gap_min": 0.04,
    "event_vol": 1.5,
    "bdr_close_max": 0.60,         # partly fitted: Breakdown Ready closes are never in the top 40% of the range
    "min_group_size": 5,           # fitted
    "group_stable_change": -2.0,   # guess: "stable" industry = mean RS change no worse than -2
    # ---- Universe / profiles
    "core_index": "R1000",         # index whose members define industry groups and peer ranks
    "profile_max_lookups": 1200,   # Yahoo profile lookups per night (industry, market cap, short interest)
    "profile_ttl_days": 45,
    "si_ttl_days": 10,             # short interest refresh for Squeeze Radar candidates
    "yahoo_industry_min_cover": 0.90,
    # ---- Safety
    "max_day_up": 2.0,             # +200% / -70% in one session = treat as bad data
    "max_day_down": -0.70,
    "min_coverage": 0.50,          # never publish if under half the universe scored
    "stale_retry_max": 900,        # symbols without today's bar are re-fetched one by one
    "max_missing_core": 0.03,      # do not publish while more than 3% of core symbols lack today's bar
    "chart_bars": 180,
    "dump_bars": 330,
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


def ema_last(x: np.ndarray, n: int) -> float:
    """Last value of an n-period exponential average, seeded far enough back to converge."""
    x = x[-max(8 * n, 120):]
    a, e = 2.0 / (n + 1), float(x[0])
    for val in x[1:]:
        e = a * val + (1 - a) * e
    return e


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


def rsi_sma(c: np.ndarray, n: int = 14) -> float:
    """RSI from plain sums of the last n gains and losses (no Wilder smoothing)."""
    d = np.diff(c[-(n + 1):])
    if len(d) < n:
        return float("nan")
    up, dn = float(d[d > 0].sum()), float(-d[d < 0].sum())
    return 50.0 if up + dn == 0 else 100.0 * up / (up + dn)


def rsi_quality(r: float) -> float:
    """100 inside the healthy 55-70 zone, fading linearly to 0 on both sides."""
    if not math.isfinite(r):
        return 0.0
    lo, hi = CFG["rsi_zero_low"], CFG["rsi_zero_high"]
    if r < 55:
        return max(0.0, (r - lo) / (55 - lo) * 100)
    if r <= 70:
        return 100.0
    return max(0.0, (hi - r) / (hi - 70) * 100)


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
    atr_arr = rolling_mean(tr, CFG["atr_len"]) if CFG["atr_method"] == "sma" else wilder(tr, CFG["atr_len"])
    atr = atr_arr[i]
    if not (math.isfinite(atr) and atr > 0 and math.isfinite(s200[i])):
        return None
    px = c[i]
    hl = h - l

    def mr(a, k, end=i):               # mean of the k values ending at `end`
        return float(np.mean(a[end - k + 1: end + 1]))

    def div(a, b_):
        return a / b_ if b_ and math.isfinite(b_) and b_ != 0 else float("nan")

    def whole(x):
        return float(math.floor(x + 0.5)) if CFG["integer_scores"] and math.isfinite(x) else x

    # ---------------- Trend ----------------
    t_above = (int(px > s20[i]) + int(px > s50[i]) + int(px > s200[i])) / 3 * 100
    a1, a2 = s20[i] > s50[i], s50[i] > s200[i]
    t_align = 100.0 if (a1 and a2) else float(CFG["align_partial"]) if (a1 or a2) else 0.0
    slope50 = s50[i] / s50[i - CFG["slope_lookback"]] - 1
    hi52 = float(np.max(h[i - 251: i + 1]))
    off_high = px / hi52 - 1
    share60 = float(np.mean(c[i - 59: i + 1] > s50[i - 59: i + 1])) * 100
    T = (0.30 * t_above + 0.25 * t_align + 0.20 * scale(slope50, -0.06, 0.08)
         + 0.15 * scale(off_high, -0.30, 0.0) + 0.10 * share60)

    # ---------------- Relative strength ----------------
    def ret(k):
        return px / c[i - k] - 1

    def rs_at(j):
        """(score, 12m, 6m, 3m excess) as of bar j; NaNs when there is not enough history."""
        if j - 252 < 0:
            return float("nan"), float("nan"), float("nan"), float("nan")

        def e(k):
            return ((c[j] / c[j - k] - 1) - (b[j] / b[j - k] - 1)) * 100
        e12, e6, e3 = e(252), e(126), e(63)
        return (0.45 * scale(e12, -30, 30) + 0.35 * scale(e6, -20, 20) + 0.20 * scale(e3, -12, 12),
                e12, e6, e3)

    RS, ex12, ex6, ex3 = rs_at(i)

    # ---------------- Momentum ----------------
    r1, r3, r6 = ret(21), ret(63), ret(126)
    accel = (r1 - r3 / 3) * 100
    rsi_w = rsi_last(c[-250:], CFG["rsi_len"])
    rsi_s = rsi_sma(c, CFG["rsi_len"])
    rsi = rsi_s if CFG["rsi_method"] == "sma" else rsi_w
    w = CFG["rvol_window"]
    avg_vol = mr(v, w)
    rvol = div(v[i], avg_vol)
    M = (0.30 * scale(r1, -0.12, 0.12) + 0.25 * scale(r3, -0.20, 0.30)
         + 0.20 * scale(accel, -8, 8) + 0.15 * rsi_quality(rsi)
         + 0.10 * scale(rvol, 0.5, 2.0))

    # The three blocks are whole numbers and the composite averages those
    # whole numbers (98, 100, 85 -> 94), which is what the original shows.
    T, RS, M = whole(T), whole(RS), whole(M)
    score = int(math.floor((T + RS + M) / 3 + 0.5))

    # ---------------- Shared scan inputs ----------------
    L = CFG["rs_change_lookback"]
    rs_chg = RS - whole(rs_at(i - L)[0])            # NaN when history is short: such scans simply don't fire
    rs_chg3m = RS - whole(rs_at(i - CFG["em_rs_lookback"])[0])
    ext = (px - ema_last(c, CFG["ext_ema"])) / atr
    rising50 = s50[i] > s50[i - CFG["rising50_lookback"]]
    rising200 = s200[i] > s200[i - CFG["rising200_lookback"]]
    rng = h[i] - l[i]
    clv = (px - l[i]) / rng if rng > 0 else 0.5
    chg = px / c[i - 1] - 1
    p20, p50 = (px / s20[i] - 1) * 100, (px / s50[i] - 1) * 100
    hl10_50 = div(mr(hl, 10), mr(hl, 50))           # 10-day average range vs 50-day
    v10_50 = div(mr(v, 10), mr(v, 50))              # 10-day average volume vs 50-day

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
    pivot_age = P - int(np.argmax(h[i - P: i]))         # sessions since the pivot high (1 = yesterday)
    to_pivot = px / piv - 1
    hi_pos = int(np.argmax(h[i - 251: i + 1]))          # 0..251, 251 = today
    hi20 = float(np.max(h[i - 20: i + 1]))
    low20 = float(np.min(l[i - 20: i]))                 # prior 20 sessions
    low50 = float(np.min(l[i - 50: i]))

    scans: List[str] = []
    note: dict = {}

    above_rising = px > s50[i] and px > s200[i] and rising50 and rising200

    # 1. Established Leaders  (fitted: 34 of 35, no false matches)
    established = (T >= 80 and RS >= 80 and above_rising
                   and off_high >= -0.15 and ext <= CFG["max_ext_atr"])
    if established:
        scans.append("el")

    # 2. Emerging Leaders  (thin evidence)
    if (not established and M >= 80 and RS >= 65 and rs_chg3m >= 4 and rising50):
        scans.append("em")

    # 3. Breakout Ready  (thin evidence)
    tight = hl10_50 <= CFG["tight_range_ratio"] and v10_50 <= CFG["tight_vol_ratio"]
    if (T >= 60 and RS >= 65 and above_rising and -0.03 <= to_pivot <= 0 and tight
            and pivot_age >= CFG["pivot_min_age"]):
        scans.append("br")

    # 4. Fresh Breakouts  (no evidence yet)
    if (T >= 60 and RS >= 65 and initial_break(i) and 0 < to_pivot <= 0.05
            and rvol >= 1.35 and clv >= CFG["strong_close"] and ext <= CFG["max_ext_atr"]):
        scans.append("fb")

    # 5. Breakout Continuation  (no evidence yet)
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

    # 6 / 7. Pullbacks  (fitted: 16 of 16 and 1 of 1, one false match)
    # "Prior 6-month leadership" adds nothing beyond Trend and RS >= 70 on the
    # evidence so far; a positive 6-month return is kept as the floor.
    near_support = abs(p20) <= CFG["near_20_pct"] or abs(p50) <= CFG["near_50_pct"]
    if T >= 70 and RS >= 70 and rising200 and r6 > 0 and near_support:
        if px > h[i - 1] and clv >= CFG["strong_close"]:
            scans.append("pc")
            note["pc"] = {"support_low": round(float(np.min(l[i - 2: i + 1])), 2)}
        elif v10_50 <= CFG["controlled_vol"]:
            scans.append("cp")

    # 8. Leadership Reclaim  (no evidence yet)
    below_cnt = int(np.sum(c[i - 10: i] < s50[i - 10: i]))
    if (T >= 55 and RS >= 55 and r6 > 0 and px > s200[i] and rising200
            and px > s50[i] and c[i - 1] <= s50[i - 1]
            and below_cnt >= CFG["reclaim_days_below"]
            and rvol >= CFG["reclaim_rvol"] and rs_chg > 0):
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

    # 14. Breakdown Ready  (partly fitted: finds 40 of 41 but also ~27 the original leaves out)
    if (RS <= 45 and M <= 50 and px < s20[i]
            and low20 <= px <= low20 * 1.05 and l[i] >= low20      # near support, not yet undercut
            and rs_chg <= 0
            and hl10_50 < 1.0 and v10_50 <= 1.0
            and clv <= CFG["bdr_close_max"]):
        scans.append("bdr")

    # 15. Bearish Burst  (the original's reading of "fresh break" is still unknown)
    def fresh_low_break(win):
        if not c[i] < np.min(l[i - win: i]):
            return False
        return not any(c[k] < np.min(l[k - win: k]) for k in range(i - 5, i))

    tr_atr = div(tr[i], atr_arr[i - 1])
    if ((fresh_low_break(20) or fresh_low_break(50)) and RS <= 55 and M <= 45
            and -0.10 <= chg <= -0.015 and 1.15 <= tr_atr <= 3.5
            and rvol >= 1.25 and clv <= CFG["weak_close"]):
        scans.append("bb")

    # Price side of Squeeze Radar (short interest is added later)
    squeeze_px = (M >= 60 and RS >= 55 and px > s20[i]
                  and px >= hi20 * 0.92 and rs_chg >= 0)

    # ---- Diagnostics: other plausible readings of the undefined terms.
    # Not used by any scan; written to diag.json so thresholds can be
    # calibrated against a reference list without re-fetching prices.
    brk = [k for k in range(0, 16) if broke(i - k)]
    wild = wilder(tr, CFG["atr_len"])[i]
    diag = {
        "tr5_20": div(mr(tr, 5), mr(tr, 20)), "tr5_20p": div(mr(tr, 5), mr(tr, 20, i - 5)),
        "tr10_30p": div(mr(tr, 10), mr(tr, 30, i - 10)), "tr10_50": div(mr(tr, 10), mr(tr, 50)),
        "hl5_20": div(mr(hl, 5), mr(hl, 20)), "hl10_50": hl10_50, "hl10_20": div(mr(hl, 10), mr(hl, 20)),
        "v5_20": div(mr(v, 5), mr(v, 20)), "v5_20p": div(mr(v, 5), mr(v, 20, i - 5)),
        "v10_30p": div(mr(v, 10), mr(v, 30, i - 10)), "v10_50": v10_50, "v5_50": div(mr(v, 5), mr(v, 50)),
        "rvol20": div(v[i], mr(v, 20, i - 1)), "rvol50": div(v[i], mr(v, 50, i - 1)),
        "rvol20i": div(v[i], mr(v, 20)), "rvol50i": rvol,
        "rsi_sma": rsi_s, "rsi_wilder": rsi_w,
        "to_pivot": to_pivot * 100, "pivot_age": pivot_age,
        "bo_ago": brk[0] if brk else None,
        "low20_dist": (px / low20 - 1) * 100, "low_today_vs20": (l[i] / low20 - 1) * 100,
        "low50_dist": (px / low50 - 1) * 100,
        "rsacc_6": ex3 - ex6 / 2, "rsacc_12": ex3 - ex12 / 4,
        "rs_d5": RS - rs_at(i - 5)[0], "rs_d10": RS - rs_at(i - 10)[0],
        "rs_d21": RS - rs_at(i - 21)[0], "rs_d63": RS - rs_at(i - 63)[0],
        "s50_d5": (s50[i] / s50[i - 5] - 1) * 100, "s50_d10": (s50[i] / s50[i - 10] - 1) * 100,
        "s50_d20": slope50 * 100, "s200_d20": (s200[i] / s200[i - 20] - 1) * 100,
        "ext_ema21": ext, "ext_sma20": (px - s20[i]) / atr, "ext_sma50": (px - s50[i]) / atr,
        "atr_sma": atr_arr[i] if CFG["atr_method"] == "sma" else mr(tr, CFG["atr_len"]), "atr_wilder": wild,
        "p20": p20, "p50": p50, "depth_atr": (hi20 - px) / atr, "clv": clv, "tr_atr": tr_atr,
        "chg": chg * 100, "hi_ago": 251 - hi_pos, "below50_10": below_cnt,
        "gt_prev_high": int(px > h[i - 1]), "rising200": int(rising200), "rising50": int(rising50),
        "open_gap": (o[i] / c[i - 1] - 1) * 100, "close_vs_open": (px / o[i] - 1) * 100,
    }

    return {
        "diag": diag,
        "score": score, "T": T, "RS": RS, "M": M,
        "close": px, "chg": chg, "rs_chg": rs_chg, "rvol": rvol, "ext": ext,
        "adr": float(np.mean(h[i - 19: i + 1] / l[i - 19: i + 1] - 1)) * 100,
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
def avg_rank(values: List[float]) -> List[float]:
    """1-based ranks, lowest value = 1, ties share the average rank."""
    order = sorted(range(len(values)), key=lambda k: values[k])
    out = [0.0] * len(values)
    k = 0
    while k < len(order):
        j = k
        while j + 1 < len(order) and values[order[j + 1]] == values[order[k]]:
            j += 1
        for t in range(k, j + 1):
            out[order[t]] = (k + j) / 2 + 1
        k = j + 1
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def _group_rules(m: dict, g: dict, pp: float) -> None:
    """Apply the three industry scans to one stock, given its group row and peer percentile."""
    m["peer_pct"], m["ind_pct"] = pp, g["pct"]
    # 11. Stock Leads Group  (thin evidence: 1 match in the reference list)
    if (m["T"] >= 70 and m["RS"] >= 85 and m["rs_chg"] > 0 and pp >= 85
            and m["RS"] - g["rs"] >= 15 and 25 <= g["pct"] < 75
            and g["accel"] >= CFG["group_stable_change"]):
        m["scans"].append("slg")
    # 12. Group Rotation Leaders  (no matches in the reference list)
    if (35 <= g["pct"] < 75 and g["accel"] > 0 and g["improving"] >= 0.5
            and pp >= 70 and m["RS"] >= 65 and m["M"] >= 65):
        m["scans"].append("grl")
    # 13. Group-Confirmed Leaders  (fitted: 53 of 53, no false matches)
    if (m["T"] >= 65 and m["RS"] >= 75 and m["above50"]
            and g["pct"] >= 75 and pp >= 70):
        m["scans"].append("gcl")


def group_pass(stocks: Dict[str, dict], core: Optional[set] = None) -> List[dict]:
    """
    Industry strength = MEAN relative-strength score of the industry's core
    members (the Russell 1000), for industries with at least 5 of them.
      industry percentile = (rank - 1) / (industries - 1)
      peer percentile     = rank / members   (ties share the average rank)
    Stocks outside the core are ranked as if added to their industry's core peers.
    """
    if not core:
        core = set(stocks)
    groups: Dict[str, List[str]] = {}
    for sym, m in stocks.items():
        if m.get("group") and sym in core:
            groups.setdefault(m["group"], []).append(sym)

    rows = []
    for key, syms in groups.items():
        if len(syms) < CFG["min_group_size"]:
            continue
        chg = [stocks[s]["rs_chg"] for s in syms]
        ok = [x for x in chg if math.isfinite(x)]
        rows.append({
            "name": key, "n": len(syms),
            "rs": float(np.mean([stocks[s]["RS"] for s in syms])),
            "accel": _mean(chg),
            "improving": float(np.mean([x > 0 for x in ok])) if ok else 0.0,
            "score": float(np.mean([stocks[s]["score"] for s in syms])),
            "syms": syms,
        })
    if not rows:
        return []
    ranks = avg_rank([r["rs"] for r in rows])
    for row, rk in zip(rows, ranks):
        row["pct"] = (rk - 1) / (len(rows) - 1) * 100 if len(rows) > 1 else 50.0
        if not math.isfinite(row["accel"]):
            row["accel"] = 0.0

    by_name = {g["name"]: g for g in rows}
    for g in rows:
        vals = [stocks[s]["RS"] for s in g["syms"]]
        for s, rk in zip(g["syms"], avg_rank(vals)):
            _group_rules(stocks[s], g, rk / len(vals) * 100)
    for sym, m in stocks.items():
        g = by_name.get(m.get("group"))
        if g is None or sym in core:
            continue
        vals = [stocks[s]["RS"] for s in g["syms"]] + [m["RS"]]
        _group_rules(m, g, avg_rank(vals)[-1] / len(vals) * 100)
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
_WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_{}_companies"
_ISH = "https://www.ishares.com/us/products/{}/latest-holdings.csv"
# index, Wikipedia table (None when Wikipedia has no member list), minimum rows, iShares fund
SOURCES = [
    ("SP500", _WIKI.format("500"), 450, "239726/ishares-core-sp-500-etf"),
    ("SP400", _WIKI.format("400"), 350, "239763/ishares-core-sp-midcap-etf"),
    ("SP600", _WIKI.format("600"), 500, "239774/ishares-core-sp-smallcap-etf"),
    ("R1000", None, 800, "239707/ishares-russell-1000-etf"),
]
SHARE_CLASS = {"BRKB": "BRK-B", "BFB": "BF-B", "BFA": "BF-A", "LENB": "LEN-B", "HEIA": "HEI-A",
               "UHALB": "UHAL-B", "MOGA": "MOG-A", "CWENA": "CWEN-A", "GEFB": "GEF-B"}
SYM_RE = re.compile(r"^[A-Z]{1,5}(-[A-Z]{1,2})?$")


def norm_symbol(s) -> Optional[str]:
    s = re.sub(r"[.\/ _*]+", "-", str(s).strip().upper())
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
    lines = text.lstrip("\ufeff").splitlines()
    start = next((k for k, ln in enumerate(lines) if ln.startswith("Ticker,")), None)
    if start is None:
        return []
    out = []
    for r in csv.DictReader(lines[start:]):
        if (r.get("Asset Class") or "").strip() != "Equity":
            continue
        if "NO MARKET" in (r.get("Exchange") or "").upper():      # delisted remnants
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
    for idx, wiki_url, min_rows, fund in SOURCES:
        rows: List[dict] = []
        if wiki_url:
            try:
                rows = parse_wiki_tables(http_get(wiki_url), min_rows)
                log(f"  {idx}: {len(rows)} names from Wikipedia")
            except Exception as e:      # noqa: BLE001
                log(f"  {idx}: Wikipedia failed ({e})")
        if len(rows) < min_rows:
            try:
                rows = parse_ishares_csv(http_get(_ISH.format(fund), BROWSER_UA))
                log(f"  {idx}: {len(rows)} names from iShares holdings")
            except Exception as e:      # noqa: BLE001
                log(f"  {idx}: iShares failed ({e})")
            if len(rows) < min_rows:
                log(f"  {idx}: WARNING - member list unavailable ({len(rows)} rows)")
                rows = []
        for r in rows:
            cur = merged.setdefault(r["s"], {"s": r["s"], "n": None, "sec": None, "ind": None, "idx": []})
            for k in ("n", "sec", "ind"):
                if not cur[k] and r.get(k):
                    cur[k] = r[k]
            if idx not in cur["idx"]:
                cur["idx"].append(idx)
    return sorted(merged.values(), key=lambda r: r["s"])


UNIVERSE_VERSION = 2          # 2: share-class tickers such as BRK.B / BF.A are kept


def _has(stocks: List[dict], idx: str) -> int:
    return sum(1 for s in stocks if idx in s.get("idx", []))


def load_universe(out_dir: str, max_age_days: int = 7) -> List[dict]:
    path = os.path.join(out_dir, "universe.json")
    core = CFG["core_index"]
    cached = None
    if os.path.exists(path):
        with open(path) as f:
            cached = json.load(f)
        age = (dt.date.today() - dt.date.fromisoformat(cached["built"])).days
        # "v" changes whenever the way symbols are read changes, forcing one rebuild
        if age < max_age_days and cached.get("v") == UNIVERSE_VERSION and _has(cached["stocks"], core) >= 800:
            log(f"Universe: {len(cached['stocks'])} names (cached {cached['built']})")
            return cached["stocks"]
    log("Universe: rebuilding index membership")
    stocks = build_universe()
    if cached:
        # Any index whose list could not be fetched today keeps its last known
        # membership, so one failing source never shrinks the universe.
        by = {s["s"]: s for s in stocks}
        for idx, *_ in SOURCES:
            if _has(stocks, idx) or not _has(cached["stocks"], idx):
                continue
            log(f"Universe: {idx} list unavailable today, carrying over the cached membership")
            for old in cached["stocks"]:
                if idx in old.get("idx", []):
                    cur = by.setdefault(old["s"], dict(old, idx=[]))
                    if idx not in cur["idx"]:
                        cur["idx"].append(idx)
        stocks = sorted(by.values(), key=lambda r: r["s"])
    if len(stocks) >= 900:
        with open(path, "w") as f:
            json.dump({"built": dt.date.today().isoformat(), "v": UNIVERSE_VERSION, "stocks": stocks}, f, separators=(",", ":"))
        log(f"Universe: {len(stocks)} names ({core}: {_has(stocks, core)})")
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
                                 auto_adjust=CFG["dividend_adjusted"], group_by="ticker", threads=True,
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
    """Fallback: Yahoo chart endpoint (already split-adjusted; dividends optional)."""
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
            f = a / c if CFG["dividend_adjusted"] else 1.0
            day = dt.datetime.fromtimestamp(t + off, dt.timezone.utc).strftime("%Y-%m-%d")
            rows.append((day, o * f, h * f, l * f, c * f, v or 0))
        if not rows:
            return None
        cols = list(zip(*rows))
        return {"dates": list(cols[0]), "o": np.array(cols[1], float), "h": np.array(cols[2], float),
                "l": np.array(cols[3], float), "c": np.array(cols[4], float), "v": np.array(cols[5], float)}
    except Exception:               # noqa: BLE001
        return None


def fetch_one(sym: str) -> Optional[dict]:
    """A single symbol on its own request: yfinance first, the plain chart endpoint second."""
    try:
        import yfinance as yf
        df = yf.Ticker(sym).history(period=CFG["history_period"], interval="1d",
                                    auto_adjust=CFG["dividend_adjusted"])
        bars = _frame_to_bars(df) if df is not None and len(df) else None
        if bars:
            return bars
    except Exception:               # noqa: BLE001
        pass
    return fetch_direct(sym)


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
# Yahoo profiles: industry, market cap, short interest (one cached lookup per stock)
# --------------------------------------------------------------------------
def fetch_profile(sym: str) -> Optional[dict]:
    import yfinance as yf
    info = yf.Ticker(sym).info or {}
    ind, mcap = info.get("industry"), info.get("marketCap")
    pct, dtc = info.get("shortPercentOfFloat"), info.get("shortRatio")
    if not ind and mcap is None and pct is None and dtc is None:
        return None
    return {"ind": ind or None, "mcap": float(mcap) if mcap else None,
            "pct": round(pct * 100, 2) if pct is not None else None,
            "dtc": round(float(dtc), 2) if dtc is not None else None}


def update_profiles(out_dir: str, urgent: List[str], core: List[str], others: List[str],
                    today: str) -> Dict[str, dict]:
    """
    Lookup order: squeeze candidates whose short interest is stale, then core
    (Russell 1000) names never looked up, then the rest, then the oldest
    records. Capped per night; whatever is left is picked up on later nights.
    """
    path = os.path.join(out_dir, "profiles.json")
    cache: Dict[str, dict] = {}
    if os.path.exists(path):
        with open(path) as f:
            cache = json.load(f)
    t0 = dt.date.fromisoformat(today)

    def age(s):
        rec = cache.get(s)
        return 10 ** 6 if not rec else (t0 - dt.date.fromisoformat(rec["on"])).days

    todo, seen = [], set()

    def add(seq):
        for s in seq:
            if s not in seen:
                seen.add(s)
                todo.append(s)
    add(s for s in urgent if age(s) >= CFG["si_ttl_days"])
    add(s for s in core if s not in cache)
    add(s for s in others if s not in cache)
    add(sorted((s for s in list(core) + list(others) if age(s) >= CFG["profile_ttl_days"]), key=lambda s: -age(s)))
    todo = todo[: CFG["profile_max_lookups"]]
    log(f"Profiles: {len(cache)} cached, {len(todo)} lookups tonight")
    fails = done = 0
    for s in todo:
        try:
            rec = fetch_profile(s)
            old = cache.get(s) or {}
            new = dict(rec or {"ind": None, "mcap": None, "pct": None, "dtc": None}, on=today)
            if not new.get("ind") and old.get("ind"):        # never lose a known industry
                new["ind"] = old["ind"]
            cache[s] = new
            fails, done = 0, done + 1
        except Exception:           # noqa: BLE001
            fails += 1
            if fails >= 10:
                log("  profile lookups failing, stopping for today")
                break
        time.sleep(0.35)
    log(f"Profiles: {done} updated")
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
        "scans", "newScans", "note", "r1k", "mcapB", "adr"]


def stock_row(sym: str, m: dict, prev: dict) -> list:
    p = prev.get(sym) or {}
    ev = m.get("ev")
    return [
        sym, m.get("name"), m.get("sec"), m.get("ind"),
        rnd(m["close"], 2), rnd(m["chg"] * 100, 2), m["score"],
        rnd(m["T"]), rnd(m["RS"]), rnd(m["M"]),
        (m["score"] - p["score"]) if "score" in p else None,
        rnd(m["rs_chg"], 0), rnd(m["rvol"], 2), rnd(m["ext"], 2), rnd(m["off_high"] * 100),
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
        1 if m.get("core") else 0,
        rnd(m["mcap"] / 1e9, 2) if m.get("mcap") else None,
        rnd(m["adr"], 2),
    ]


def load_previous(out_dir: str, asof: Optional[str] = None) -> tuple:
    """
    Returns (asOf of the published file, whether that file was complete,
    {symbol: score/scans} for the last stored session BEFORE `asof`).
    Day-over-day changes always compare with the previous session, so a
    re-run of the same day does not wipe them out.
    """
    pub_asof, complete = None, True
    path = os.path.join(out_dir, "latest.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                j = json.load(f)
            pub_asof = j.get("asOf")
            cu = j.get("coreUniverse") or 0
            complete = not j.get("partial") and (not cu or j.get("coreScored", 0) >= 0.9 * cu)
        except Exception:           # noqa: BLE001
            pass
    prev: Dict[str, dict] = {}
    hdir = os.path.join(out_dir, "history")
    if asof and os.path.isdir(hdir):
        days = sorted(f[:-4] for f in os.listdir(hdir) if f.endswith(".csv") and f[:-4] < asof)
        if days:
            with open(os.path.join(hdir, days[-1] + ".csv")) as f:
                for r in csv.DictReader(f):
                    prev[r["sym"]] = {"score": int(r["score"]), "scans": [x for x in r["scans"].split("|") if x]}
    return pub_asof, complete, prev


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
# NYSE full-day closures (weekends are handled separately)
MARKET_HOLIDAYS = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
    "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}


def expected_session(now: Optional[dt.datetime] = None) -> str:
    """The latest trading day whose close should be in by now (New York time, close + 2h)."""
    from zoneinfo import ZoneInfo
    ny = (now or dt.datetime.now(dt.timezone.utc)).astimezone(ZoneInfo("America/New_York"))
    d = ny.date() if ny.hour >= 18 else ny.date() - dt.timedelta(days=1)
    while d.weekday() >= 5 or d.isoformat() in MARKET_HOLIDAYS:
        d -= dt.timedelta(days=1)
    return d.isoformat()


def write_status(out_dir: str, **kw) -> None:
    """Small file the page reads to explain a night when nothing new was published."""
    kw["attempt"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(os.path.join(out_dir, "status.json"), "w") as f:
        json.dump(kw, f, separators=(",", ":"))


def refetch_stale(prices: Dict[str, dict], symbols: List[str], asof: str) -> int:
    """
    Yahoo sometimes answers a bulk request with yesterday's last bar for part
    of the list. Ask again, one symbol at a time, for everything that lacks
    the latest session. Returns how many were repaired.
    """
    todo = [s for s in symbols if s not in prices or prices[s]["dates"][-1] != asof]
    todo = todo[: CFG["stale_retry_max"]]
    if not todo:
        return 0
    log(f"  {len(todo)} symbols lack the {asof} bar - re-fetching them individually")
    fixed = fails = 0
    for s in todo:
        bars = fetch_one(s)
        if bars and bars["dates"][-1] == asof:
            prices[s] = bars
            fixed += 1
            fails = 0
        else:
            fails += 1
            if fails >= 60:             # the source simply does not have today's bars yet
                log("  re-fetch is not returning the latest session, stopping")
                break
        time.sleep(0.15)
    log(f"  repaired {fixed} of {len(todo)}")
    return fixed


def run(out_dir: str, force: bool = False, limit: Optional[int] = None,
        allow_partial: bool = False, dump_bars: bool = False) -> int:
    os.makedirs(out_dir, exist_ok=True)
    universe = load_universe(out_dir)
    if limit:
        universe = universe[:limit]
    meta = {u["s"]: u for u in universe}
    bench_sym = CFG["benchmark"]
    core_idx = CFG["core_index"]
    core_all = {s for s, u in meta.items() if core_idx in u.get("idx", [])}
    if not core_all:
        log(f"WARNING: no {core_idx} members in the universe - group ranks will use every stock")

    # Benchmark first: it tells us the latest completed session, so a repeat
    # run on the same day (or a holiday) can stop before fetching 1,900 symbols.
    bench = fetch_prices([bench_sym]).get(bench_sym)
    if not bench or len(bench["c"]) < CFG["min_bars"]:
        raise SystemExit("Benchmark history unavailable - not publishing.")
    bench["_map"] = {d: k for k, d in enumerate(bench["dates"])}
    asof = bench["dates"][-1]
    expected = expected_session()
    behind = asof < expected
    if behind:
        # Yahoo sometimes releases a day's closing prices hours late. Work with
        # what exists (the previous session may still be unpublished) and tell
        # the page that the newer session is still pending; a later run retries.
        log(f"Latest bar available is {asof}; {expected} has not been released yet")

    def pending_note():
        if behind:
            write_status(out_dir, asOf=expected, published=False, missing=None, of=None,
                         message=f"closing prices for {expected} not released yet")

    prev_asof, prev_complete, prev = load_previous(out_dir, asof)
    if prev_asof == asof and not force:
        if prev_complete:
            log(f"Already published for {asof}; nothing to do.")
            pending_note()
            return 0
        log(f"{asof} was published incomplete earlier - trying again")

    log(f"Session {asof}: fetching daily bars for {len(meta)} stocks")
    symbols = [s for s in meta if s != bench_sym]
    prices = fetch_prices(symbols)
    refetch_stale(prices, symbols, asof)

    # Completeness gate. Publishing with a third of the list missing (as happened
    # for the 2026-10-06 close) silently drops real matches, so wait instead:
    # a later scheduled run tries again, and the page keeps the previous session.
    gate = sorted(core_all) if core_all else symbols
    lacking = [s for s in gate if s not in prices or prices[s]["dates"][-1] != asof]
    frac = len(lacking) / max(1, len(gate))
    partial = frac > CFG["max_missing_core"]
    if partial:
        msg = (f"{len(lacking)} of {len(gate)} {core_idx if core_all else 'universe'} symbols "
               f"have no bar for {asof} yet")
        if not allow_partial:
            log(f"NOT PUBLISHING: {msg}. A later run will retry.")
            write_status(out_dir, asOf=asof, published=False, missing=len(lacking), of=len(gate), message=msg)
            return 0
        log(f"WARNING: {msg} - publishing anyway (last attempt for this session)")

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
        m.update(name=u.get("n"), sec=u.get("sec"), gics=u.get("ind"), core=sym in core_all)
        stocks[sym] = m
        series[sym] = {"dates": a["dates"], "c": a["c"], "b": a["b"]}
        if dump_bars and sym in core_all:
            series[sym]["ohlcv"] = a

    coverage = len(stocks) / max(1, len(meta))
    log(f"Scored {len(stocks)}/{len(meta)} ({coverage:.0%}); skipped {skipped}")
    if coverage < CFG["min_coverage"]:
        raise SystemExit("Coverage too low - data source problem. Not publishing.")

    # Profiles (industry, market cap, short interest)
    profiles: Dict[str, dict] = {}
    try:
        cands = sorted((s for s, m in stocks.items() if m["squeeze_px"]), key=lambda s: -stocks[s]["M"])
        by_score = sorted(stocks, key=lambda s: -stocks[s]["score"])
        profiles = update_profiles(out_dir, cands,
                                   [s for s in by_score if stocks[s]["core"]],
                                   [s for s in by_score if not stocks[s]["core"]], asof)
    except Exception as e:          # noqa: BLE001
        log(f"Profiles unavailable today ({e})")

    # One industry taxonomy at a time: Yahoo's once it covers the core, GICS until then.
    core = {s for s in stocks if stocks[s]["core"]}
    base = core or set(stocks)
    cover = sum(1 for s in base if (profiles.get(s) or {}).get("ind")) / max(1, len(base))
    ind_source = "yahoo" if cover >= CFG["yahoo_industry_min_cover"] else "gics"
    log(f"Industries: Yahoo covers {cover:.0%} of {'the ' + core_idx if core else 'the universe'} -> using {ind_source}")
    for sym, m in stocks.items():
        rec = profiles.get(sym) or {}
        m["ind"] = rec.get("ind") if ind_source == "yahoo" else m.get("gics")
        m["group"] = m["ind"]
        m["mcap"] = rec.get("mcap")

    groups = group_pass(stocks, core)
    squeeze_pass(stocks, profiles)

    order = {s[0]: k for k, s in enumerate(SCANS)}
    for m in stocks.values():
        m["scans"] = sorted(set(m["scans"]), key=order.get)

    ranked = sorted(stocks, key=lambda s: -stocks[s]["score"])
    rows = [stock_row(s, stocks[s], prev) for s in ranked]
    counts = {sid: sum(1 for m in stocks.values() if sid in m["scans"]) for sid, *_ in SCANS}

    write_history(out_dir, asof, stocks)
    validation = build_validation(out_dir, series)

    latest = {
        "asOf": asof,
        "generatedAt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark": bench_sym,
        "universe": len(meta), "scored": len(stocks), "skipped": skipped,
        "coreIndex": core_idx, "coreUniverse": len(core_all), "coreScored": len(core),
        "coreMissing": len(lacking), "partial": bool(partial),
        "industrySource": ind_source,
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

    dcols = list(next(iter(stocks.values()))["diag"].keys())
    diag = {"asOf": asof, "note": "calibration inputs only - not read by the page",
            "cols": ["sym"] + dcols,
            "rows": [[s] + [rnd(stocks[s]["diag"][k], 3) for k in dcols] for s in ranked]}
    with open(os.path.join(out_dir, "diag.json"), "w") as f:
        json.dump(diag, f, separators=(",", ":"), allow_nan=False)

    if dump_bars:
        # One-off export of recent daily bars for the core list, so scan
        # definitions can be fitted offline against a reference list.
        import gzip
        nb = CFG["dump_bars"]
        out = {"asOf": asof, "dates": bench["dates"][-nb:], "bench": [rnd(x, 3) for x in bench["c"][-nb:]], "bars": {}}
        for s, ser in series.items():
            a = ser.get("ohlcv")
            if a is None:
                continue
            out["bars"][s] = {"d": a["dates"][-nb:][0],
                              **{k: [rnd(x, 3) for x in a[k][-nb:]] for k in ("o", "h", "l", "c")},
                              "v": [int(x) for x in a["v"][-nb:]]}
        with gzip.open(os.path.join(out_dir, "bars.json.gz"), "wt") as f:
            json.dump(out, f, separators=(",", ":"), allow_nan=False)
        log(f"Wrote bars.json.gz for {len(out['bars'])} symbols")

    write_status(out_dir, asOf=asof, published=True, missing=len(lacking), of=len(gate),
                 message="partial data" if partial else "ok")
    pending_note()
    log(f"Published {asof}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/composite")
    ap.add_argument("--force", action="store_true", help="re-run even if this session is already published")
    ap.add_argument("--limit", type=int, default=None, help="only the first N symbols (testing)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="publish even if many symbols lack the latest bar (last retry of the night)")
    ap.add_argument("--dump-bars", action="store_true", help="also write bars.json.gz (calibration export)")
    args = ap.parse_args()
    sys.exit(run(args.out, force=args.force, limit=args.limit,
                 allow_partial=args.allow_partial, dump_bars=args.dump_bars))
