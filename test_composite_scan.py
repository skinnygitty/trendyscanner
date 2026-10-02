#!/usr/bin/env python3
"""
Offline tests for composite_scan.py (no network).

  python scripts/test_composite_scan.py            # run checks
  python scripts/test_composite_scan.py --demo DIR # also write a synthetic latest.json to DIR

1. Scores are recomputed with an independent pandas implementation and compared.
2. A synthetic universe is scanned; every scan must fire at least once, and each
   hit is re-checked against the literal rule using independently computed inputs.
3. The full pipeline runs end to end with the network layer stubbed out.
"""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composite_scan as cs  # noqa: E402

N_BARS = 502


def make_bench(rng):
    r = rng.normal(0.0005, 0.008, N_BARS)
    return 100 * np.exp(np.cumsum(r))


def make_stock(rng, kind):
    """Random walk with a regime that makes particular setups likely."""
    n = N_BARS
    vol = rng.uniform(0.010, 0.030)
    drift = np.full(n, rng.normal(0.0002, 0.0009))
    volu = rng.lognormal(13.5, 0.25, n)
    gaps = np.zeros(n)
    if kind == "leader":
        drift[:] = rng.uniform(0.0015, 0.0030)
    elif kind == "emerging":
        drift[:-70] = rng.uniform(-0.0005, 0.0005); drift[-70:] = rng.uniform(0.004, 0.007)
    elif kind == "base":            # strong run then a tight, quiet base
        drift[:-25] = rng.uniform(0.0015, 0.0030); drift[-25:] = 0.0
    elif kind == "pullback":
        drift[:-8] = rng.uniform(0.0020, 0.0030); drift[-8:] = rng.uniform(-0.008, -0.003)
    elif kind == "reclaim":
        drift[:-45] = rng.uniform(0.0015, 0.0025); drift[-45:-6] = rng.uniform(-0.004, -0.002)
        drift[-6:] = rng.uniform(0.006, 0.012)
    elif kind == "event":
        drift[:] = rng.uniform(0.0008, 0.0020)
        d = n - 1 - rng.integers(0, 18)
        gaps[d] = rng.uniform(0.05, 0.15); volu[d] *= rng.uniform(2, 5)
    elif kind == "weak":
        drift[:] = rng.uniform(-0.0025, -0.0008)
    elif kind == "squeeze":
        drift[:-40] = rng.uniform(-0.001, 0.0005); drift[-40:] = rng.uniform(0.003, 0.006)
    noise = rng.normal(0, vol, n)
    if kind == "base":
        noise[-25:] *= 0.3; volu[-25:] *= 0.55
    if kind == "weak" and rng.random() < 0.5:
        noise[-12:] *= 0.35; volu[-12:] *= 0.6
    close = 50 * np.exp(np.cumsum(drift + noise + gaps))
    prev = np.roll(close, 1); prev[0] = close[0]
    open_ = prev * (1 + gaps + rng.normal(0, vol * 0.3, n))
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol * 0.4, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol * 0.4, n)))
    # nudge the last bar so triggers are reachable in a random sample
    t = rng.random()
    if t < 0.25:                    # strong up day on volume, closing near the high
        close[-1] = max(close[-1], hi[-2] * (1 + rng.uniform(0.002, 0.03)))
        hi[-1] = close[-1] * 1.002; lo[-1] = min(lo[-1], open_[-1]); volu[-1] *= rng.uniform(1.3, 2.5)
        lo[-1] = min(lo[-1], close[-1] * 0.97)
    elif t < 0.40 and kind == "weak":   # weak down day on volume, closing near the low
        close[-1] = close[-2] * (1 - rng.uniform(0.02, 0.07))
        open_[-1] = close[-2]; hi[-1] = close[-2] * 1.003; lo[-1] = close[-1] * 0.997
        volu[-1] *= rng.uniform(1.4, 2.5)
    return open_, hi, lo, close, volu


KINDS = ["leader", "emerging", "base", "pullback", "reclaim", "event", "weak", "squeeze", "plain"]
SECTORS = {"Technology": ["Semiconductors", "Application Software", "Systems Software", "IT Hardware"],
           "Health Care": ["Biotechnology", "Medical Devices", "Managed Care"],
           "Industrials": ["Aerospace & Defense", "Machinery", "Building Products"],
           "Financials": ["Regional Banks", "Insurance", "Asset Managers"],
           "Energy": ["Oil & Gas E&P", "Oil Services"],
           "Consumer": ["Restaurants", "Specialty Retail", "Homebuilding"]}


def synthetic_universe(n_stocks=1900, seed=7):
    rng = np.random.default_rng(seed)
    bench = make_bench(rng)
    inds = [(s, i) for s, lst in SECTORS.items() for i in lst]
    tilt = {i: rng.normal(0, 0.0008) for _, i in inds}      # industries trend together
    out = {}
    for k in range(n_stocks):
        sec, ind = inds[k % len(inds)]
        kind = KINDS[rng.integers(0, len(KINDS))] if rng.random() < 0.45 else "plain"
        o, h, l, c, v = make_stock(rng, kind)
        g = np.exp(np.cumsum(np.full(N_BARS, tilt[ind])))
        o, h, l, c = o * g, h * g, l * g, c * g
        sym = "".join(chr(65 + (k // 26 ** j) % 26) for j in (2, 1, 0)) + "X"
        out[sym] = dict(o=o, h=h, l=l, c=c, v=v, kind=kind, sec=sec, ind=ind)
    return bench, out


# ---------------------------------------------------------------- independent
def reference_scores(o, h, l, c, v, b):
    """Scores written the long way with pandas, straight from the published text."""
    c_, b_, h_, v_ = (pd.Series(x) for x in (c, b, h, v))
    s20, s50, s200 = (c_.rolling(k).mean() for k in (20, 50, 200))
    last = c_.iloc[-1]
    def lin(x, lo, hi): return float(np.clip((x - lo) / (hi - lo), 0, 1) * 100)
    above = sum([last > s20.iloc[-1], last > s50.iloc[-1], last > s200.iloc[-1]]) / 3 * 100
    align = 50 * (s20.iloc[-1] > s50.iloc[-1]) + 50 * (s50.iloc[-1] > s200.iloc[-1])
    slope = s50.iloc[-1] / s50.iloc[-21] - 1
    off = last / h_.iloc[-252:].max() - 1
    share = (c_.iloc[-60:] > s50.iloc[-60:]).mean() * 100
    T = .30 * above + .25 * align + .20 * lin(slope, -.06, .08) + .15 * lin(off, -.30, 0) + .10 * share
    sr, br = c_.pct_change, b_.pct_change
    ex = {k: (sr(k).iloc[-1] - br(k).iloc[-1]) * 100 for k in (252, 126, 63)}
    RS = .45 * lin(ex[252], -30, 30) + .35 * lin(ex[126], -20, 20) + .20 * lin(ex[63], -12, 12)
    r1, r3 = sr(21).iloc[-1], sr(63).iloc[-1]
    d = c_.iloc[-250:].diff().dropna()
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    au, ad = up.iloc[:14].mean(), dn.iloc[:14].mean()
    for k in range(14, len(d)):
        au = (au * 13 + up.iloc[k]) / 14; ad = (ad * 13 + dn.iloc[k]) / 14
    rsi = 100 - 100 / (1 + au / ad)
    q = 0 if rsi < 30 else (rsi - 30) * 4 if rsi < 55 else 100 if rsi <= 70 else 100 - (rsi - 70) * 3 if rsi < 90 else 40
    rvol = v_.iloc[-1] / v_.iloc[-51:-1].mean()
    M = (.30 * lin(r1, -.12, .12) + .25 * lin(r3, -.20, .30) + .20 * lin((r1 - r3 / 3) * 100, -8, 8)
         + .15 * q + .10 * lin(rvol, .5, 2))
    return T, RS, M


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run_tests(demo_dir=None):
    # ---- scale / boundary behaviour
    check(cs.scale(0, -30, 30) == 50, "zero excess return must map to 50")
    check(cs.scale(-99, -30, 30) == 0 and cs.scale(99, -30, 30) == 100, "scale clamps")
    check(cs.rsi_quality(60) == 100 and cs.rsi_quality(55) == 100 and cs.rsi_quality(70) == 100, "RSI 55-70 = 100")
    check(cs.rsi_quality(20) == 0 and cs.rsi_quality(42.5) == 50 and cs.rsi_quality(95) == 40, "RSI tails")

    bench, uni = synthetic_universe()
    stocks, n_checked, worst = {}, 0, 0.0
    for sym, d in uni.items():
        m = cs.analyze(d["o"], d["h"], d["l"], d["c"], d["v"], bench)
        if m is None:
            continue
        m.update(name=sym + " Corp", sec=d["sec"], ind=d["ind"], group=d["ind"])
        stocks[sym] = m
        if n_checked < 300:
            T, RS, M = reference_scores(d["o"], d["h"], d["l"], d["c"], d["v"], bench)
            err = max(abs(T - m["T"]), abs(RS - m["RS"]), abs(M - m["M"]))
            worst = max(worst, err); n_checked += 1
            check(err < 1e-6, f"{sym}: score mismatch vs reference ({err})")
            check(m["score"] == int(round((T + RS + M) / 3)), f"{sym}: composite is not the simple average")
            check(0 <= m["T"] <= 100 and 0 <= m["RS"] <= 100 and 0 <= m["M"] <= 100, "scores in 0-100")
    print(f"scores: {n_checked} stocks match the independent reference (max diff {worst:.2e})")

    # benchmark against itself must sit at RS = 50
    mb = cs.analyze(bench, bench * 1.005, bench * 0.995, bench, np.full(N_BARS, 1e6), bench)
    check(abs(mb["RS"] - 50) < 1e-9, "benchmark vs itself should be RS 50")

    groups = cs.group_pass(stocks)
    rng = np.random.default_rng(3)
    si = {s: {"pct": float(rng.uniform(0, 25)), "dtc": float(rng.uniform(0.5, 8))} for s in stocks}
    cs.squeeze_pass(stocks, si)

    counts = {sid: sum(sid in m["scans"] for m in stocks.values()) for sid, *_ in cs.SCANS}
    print("scan hits:", counts)
    for sid, n in counts.items():
        check(n > 0, f"scan {sid} never fired on the synthetic universe")
        check(n < len(stocks) * 0.5, f"scan {sid} fires on over half the universe - rule too loose")

    # ---- re-check each hit against the literal rule with independent inputs
    gmap = {s: g for g in groups for s in g["syms"]}
    for sym, m in stocks.items():
        d = uni[sym]; c, h, l, o, v = (pd.Series(d[k]) for k in "chlov")
        s20, s50, s200 = (c.rolling(k).mean() for k in (20, 50, 200))
        px, sc = c.iloc[-1], m["scans"]
        pivot = h.iloc[-56:-1].max()
        rvol = v.iloc[-1] / v.iloc[-51:-1].mean()
        clv = (px - l.iloc[-1]) / (h.iloc[-1] - l.iloc[-1])
        if "el" in sc:
            check(m["T"] >= 80 and m["RS"] >= 80, f"{sym} el scores")
            check(px > s50.iloc[-1] > s50.iloc[-11] and px > s200.iloc[-1] > s200.iloc[-21], f"{sym} el averages")
            check(px >= 0.85 * h.iloc[-252:].max() and m["ext"] <= 2.75, f"{sym} el high/extension")
            check("em" not in sc, f"{sym} cannot be both established and emerging")
        if "em" in sc:
            check(m["M"] >= 80 and m["RS"] >= 65 and m["rs_accel"] >= 4, f"{sym} em")
        if "br" in sc:
            check(m["T"] >= 60 and m["RS"] >= 65 and 0.97 * pivot <= px <= pivot, f"{sym} br pivot distance")
        if "fb" in sc:
            check(pivot < px <= 1.05 * pivot and rvol >= 1.35 and clv >= 0.65, f"{sym} fb trigger")
            for k in range(1, 11):      # no earlier break in the window
                check(c.iloc[-1 - k] <= h.iloc[-56 - k:-1 - k].max(), f"{sym} fb not fresh")
        if "bc" in sc:
            check(m["T"] >= 70 and m["RS"] >= 70 and px > h.iloc[-2], f"{sym} bc")
            ago, p0 = m["note"]["bc"]["ago"], m["note"]["bc"]["pivot"]
            check(2 <= ago <= 10 and c.iloc[-1 - ago:].min() >= p0 - 0.01, f"{sym} bc pivot held")
        if "pc" in sc:
            check(px > h.iloc[-2] and clv >= 0.65 and m["T"] >= 70 and m["RS"] >= 70, f"{sym} pc trigger")
            check(s200.iloc[-1] > s200.iloc[-21], f"{sym} pc rising 200")
        if "cp" in sc:
            check(m["T"] >= 70 and m["RS"] >= 70 and m["ex6"] >= 10, f"{sym} cp")
        if "lr" in sc:
            check(px > s50.iloc[-1] and c.iloc[-2] <= s50.iloc[-2] and px > s200.iloc[-1], f"{sym} lr cross")
            check(c.iloc[-127] < px and rvol >= 1.2 and m["rs_accel"] > 0, f"{sym} lr")
        if "ei" in sc or "ed" in sc:
            ev = m["ev"]; k = len(c) - 1 - ev["ago"]
            gap = o.iloc[k] / c.iloc[k - 1] - 1
            check(gap >= 0.04 and v.iloc[k] / v.iloc[k - 50:k].mean() >= 1.5, f"{sym} event gap/volume")
            kept = (px - c.iloc[k - 1]) / (o.iloc[k] - c.iloc[k - 1])
            if "ei" in sc:
                check(ev["ago"] <= 2 and kept >= 0.70 and m["M"] >= 65 and m["RS"] >= 65, f"{sym} ei")
            if "ed" in sc:
                check(4 <= ev["ago"] <= 20 and kept >= 0.80 and m["T"] >= 60 and m["RS"] >= 70, f"{sym} ed")
        if "bdr" in sc:
            lo20 = l.iloc[-21:-1].min()
            check(m["RS"] <= 45 and m["M"] <= 50 and px < s20.iloc[-1] and lo20 <= px <= 1.05 * lo20, f"{sym} bdr")
        if "bb" in sc:
            chg = px / c.iloc[-2] - 1
            check(-0.10 <= chg <= -0.015 and rvol >= 1.25 and clv <= 0.35, f"{sym} bb day")
            check(px < l.iloc[-21:-1].min() or px < l.iloc[-51:-1].min(), f"{sym} bb break")
            check(m["RS"] <= 55 and m["M"] <= 45, f"{sym} bb scores")
        if "sq" in sc:
            check(si[sym]["pct"] >= 8 and si[sym]["dtc"] >= 2.5 and m["M"] >= 60 and m["RS"] >= 55, f"{sym} sq")
            check(px > s20.iloc[-1] and px >= 0.92 * h.iloc[-21:].max() and m["rs_accel"] >= 0, f"{sym} sq price")
        g = gmap.get(sym)
        if "slg" in sc:
            check(m["RS"] >= 85 and m["T"] >= 70 and 25 <= g["pct"] < 75 and m["RS"] - g["rs"] >= 15
                  and m["peer_pct"] >= 85, f"{sym} slg")
        if "grl" in sc:
            check(35 <= g["pct"] < 75 and g["accel"] > 0 and g["improving"] >= 0.5 and m["peer_pct"] >= 70, f"{sym} grl")
        if "gcl" in sc:
            check(g["pct"] >= 75 and m["peer_pct"] >= 70 and m["T"] >= 65 and m["RS"] >= 75 and px > s50.iloc[-1], f"{sym} gcl")
    print("scan rules: every hit re-verified against the literal rule")

    # ---- guards
    short = {k: uni[next(iter(uni))][k][:200] for k in "ohlcv"}
    check(cs.analyze(short["o"], short["h"], short["l"], short["c"], short["v"], bench[:200]) is None,
          "short history must not be scored")
    d = dict(uni[next(iter(uni))]); bad = d["c"].copy(); bad[-40:] *= 0.2
    check(cs.analyze(d["o"], d["h"], d["l"], bad, d["v"], bench) is None, "discontinuity must not be scored")

    # ---- parsers
    wiki = "<table><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th></tr>" + \
           "".join(f"<tr><td>T{chr(65+k%26)}{chr(65+k//26)}</td><td>Co {k}</td><td>Tech</td><td>Software</td></tr>"
                   for k in range(30)) + "<tr><td>BRK.B</td><td>Berkshire</td><td>Fin</td><td>Multi</td></tr></table>"
    rows = cs.parse_wiki_tables(wiki, 10)
    check(len(rows) == 31 and rows[-1]["s"] == "BRK-B" and rows[0]["ind"] == "Software" and rows[0]["sec"] == "Tech",
          "wikipedia table parser")
    ish = 'iShares Fund\nFund Holdings as of,"Sep 30, 2026"\n\xa0\nTicker,Name,Sector,Asset Class,Market Value\n' \
          '"AAPL","APPLE INC","Information Technology","Equity","1"\n"BRKB","BERKSHIRE","Financials","Equity","1"\n' \
          '"USD","USD CASH","Cash and/or Derivatives","Cash","1"\n'
    rows = cs.parse_ishares_csv(ish)
    check([r["s"] for r in rows] == ["AAPL", "BRK-B"], "iShares holdings parser")
    print("parsers and guards: ok")

    # ---- full pipeline with the network stubbed out
    out = demo_dir or tempfile.mkdtemp()
    os.makedirs(out, exist_ok=True)
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(end="2026-09-30", periods=N_BARS)]

    def bars(d, cut=None):
        sl = slice(None, cut)
        return {"dates": days[sl], **{k: d[k][sl] for k in "ohlcv"}}

    bench_bars = {"o": bench, "h": bench * 1.004, "l": bench * 0.996, "c": bench, "v": np.full(N_BARS, 3e6)}
    universe = [{"s": s, "n": s.title() + " Holdings", "sec": d["sec"], "ind": d["ind"], "idx": ["R1000"]}
                for s, d in uni.items()]
    cs.load_universe = lambda *_a, **_k: universe
    cs.fetch_short_interest = lambda s: {"pct": round(si[s]["pct"], 2), "dtc": round(si[s]["dtc"], 2)} if s in si else None
    orig_sleep, cs.time.sleep = cs.time.sleep, lambda *_: None

    # day 1: as of 30 sessions ago; day 2: latest. Then validation must see a matured 21-session window.
    for cut in (N_BARS - 30, None):
        cs.fetch_prices = lambda syms, cut=cut: {"IWB": bars(bench_bars, cut), **{s: bars(uni[s], cut) for s in syms if s in uni}}
        check(cs.run(out) == 0, "pipeline run failed")
    check(cs.run(out) == 0, "idempotent re-run failed")           # same day again -> no-op
    cs.time.sleep = orig_sleep

    with open(os.path.join(out, "latest.json")) as f:
        j = json.load(f)
    ci = {c: k for k, c in enumerate(j["cols"])}
    check(j["asOf"] == days[-1] and j["scored"] > 1500, "latest.json header")
    check(all(len(r) == len(j["cols"]) for r in j["rows"]), "row width")
    check(j["rows"][0][ci["score"]] >= j["rows"][-1][ci["score"]], "rows sorted by score")
    check(any(r[ci["dScore"]] is not None for r in j["rows"]), "score change vs previous run")
    check(j["validation"]["snapshots"] == 2, "two snapshots stored")
    v21 = j["validation"]["horizons"]["21"]
    check(v21["days"] == 1 and v21["buckets"]["80-100"]["n"] > 0, "21-session outcomes matured")
    check(j["validation"]["horizons"]["63"]["days"] == 0, "63-session outcomes must not be reported early")
    check(os.path.exists(os.path.join(out, "charts.json")), "charts.json written")
    print(f"pipeline: ok ({j['scored']} scored, latest.json {os.path.getsize(os.path.join(out, 'latest.json')) // 1024} KB)")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    demo = sys.argv[sys.argv.index("--demo") + 1] if "--demo" in sys.argv else None
    run_tests(demo)
