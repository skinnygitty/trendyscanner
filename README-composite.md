# Composite Scanner

A nightly end-of-day scanner that runs for free on GitHub: a scheduled GitHub
Actions job scores the universe after the close, commits the results as JSON,
and `composite.html` (served by GitHub Pages) displays them.

## Files

| Path | What it is |
|---|---|
| `composite.html` | The page. Reads `data/composite/latest.json`. |
| `scripts/composite_scan.py` | The nightly job: universe, prices, scores, 16 scans. |
| `scripts/requirements.txt` | Python packages the job installs. |
| `scripts/test_composite_scan.py` | Offline tests (no network needed). |
| `.github/workflows/composite-scan.yml` | The schedule. |
| `data/composite/` | Created by the job: `latest.json`, `charts.json`, `universe.json`, `profiles.json`, `diag.json`, `history/YYYY-MM-DD.csv`. |

## Setup (GitHub website, about 5 minutes)

1. Open the repo → **Add file → Upload files**. Drag in `composite.html`,
   `README-composite.md` and the whole `scripts` folder. Commit.
2. **Add file → Create new file**. In the name box type exactly
   `.github/workflows/composite-scan.yml` (the slashes create the folders).
   Paste the contents of `paste-this-workflow/composite-scan.yml`. Commit.
3. Open the **Actions** tab (enable workflows if asked) → **Composite scan
   (nightly)** → **Run workflow**. The first run takes roughly 10 minutes.
4. Open `https://<user>.github.io/<repo>/composite.html`.

After that it runs by itself every weekday just after 6 pm New York time.
If a run fails GitHub emails you; the previous day's data stays on the page.

## Schedule

GitHub's scheduler is UTC-only, so the job is queued at 22:07 and 23:07 UTC
and a first step only lets it proceed once it is 6 pm or later in New York.
That keeps it at 6 pm ET in both summer and winter. Extra runs exit in a few
seconds when the session is already published. Holidays are handled the same
way (no new session, nothing to do). GitHub can start scheduled jobs 5-20
minutes late when it is busy.

## Data sources (all free, no keys)

* **Universe**: the Russell 1000, taken from the iShares Russell 1000 ETF
  daily holdings file, plus S&P 500 / 400 / 600 members from Wikipedia.
  Refreshed weekly, cached in `universe.json`. The page shows the Russell 1000
  by default; the Universe selector adds the S&P 400 and 600 names.
* **Daily bars**: Yahoo Finance through the `yfinance` library, adjusted for
  splits but not dividends. Unofficial and rate-limited; the job retries and
  refuses to publish if under half the universe comes back.
* **Profiles**: Yahoo Finance industry, market cap and short interest, one
  lookup per stock, cached in `profiles.json` (up to 1,200 lookups a night, so
  the first run is slower and the wider universe fills in over two nights).
  Yahoo industries are used once they cover 90% of the Russell 1000; until
  then the Wikipedia GICS sub-industries are used.
* **Benchmark**: IWB (iShares Russell 1000 ETF), price only.

## Calibration

`data/composite/diag.json` is rewritten every night with alternative readings
of the terms the published rules leave undefined (range and volume contraction
over several windows, relative volume on 20 and 50 days, RS change measures,
pivot age and so on). The page does not read it. It exists so thresholds in
`CFG` can be fitted against a reference list without re-fetching prices.

Settings already fitted against published scores (2026-10-02 close): RSI is
the simple-average form with quality fading to zero at 80; returns exclude
dividends; the composite averages the three whole-number block scores.

## Changing the rules

Every threshold that is an interpretation rather than a published number is
in the `CFG` block at the top of `scripts/composite_scan.py`, and each scan is
a short labelled block inside `analyze()`. The "Scoring model" tab on the page
lists the interpretations in plain language; update that text if you change them.

Run `python scripts/test_composite_scan.py` after any change. It checks the
scores against an independent implementation and re-verifies every scan hit.

## Known limits

* End-of-day only. No intraday data.
* No point-in-time security master. Ticker changes and delistings are handled
  only by leaving out symbols with missing, stale or discontinuous history.
* S&P membership comes from Wikipedia, which can lag an index change by days.
* Forward validation starts from the first night this runs. Nothing is back-filled.
* The repo grows by roughly 0.5 MB per trading day of committed data.

For personal research and education only. Not investment advice.
