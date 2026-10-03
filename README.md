# Pairs Trading: Walk-Forward Research and Sensitivity Analysis

A Python research project that screens equity pairs for cointegration, generates spread signals, and accounts for both trading legs with transaction costs. Selection uses only each fold's preceding formation period. The project reports unsuccessful selections and cash periods as well as completed trades.

**Finding:** the default walk-forward strategy returned approximately **−0.44%** over 403 evaluation sessions, with two completed trades. Results changed substantially with the lookback and trading-block length. This sample does not establish a robust profitable strategy.

## Run the project

Use Python 3.10 or later. Keep `main2.py` and `robustness.py` together. `main2.py` is the current strategy filename; the sensitivity runner prefers it over an adjacent `main.py` and checks compatibility before importing it.

For a new environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

For an initial download, baseline evaluation, CSV exports and plots:

```bash
python3 main2.py
```

The plot windows may need to be closed for the script to finish. A fresh download can change historical adjusted prices. Archive the resulting `pairs_downloaded_prices.csv` before downloading again.

For the 15 preset sensitivity cases using the existing snapshot:

```bash
python3 robustness.py
```

The sensitivity runner does not download prices. To replay a specific archived sample and keep its outputs separate:

```bash
python3 robustness.py --prices pairs_downloaded_prices.csv --output-dir robustness_replay
```

## Default design

| Setting | Value |
| --- | --- |
| Universe | XOM, CVX, KO, PEP, JPM, BAC, GS, MS, GOOG, MSFT, AAPL, META, WMT, TGT, MCD, YUM |
| Download interval | 2020-01-01 through 2024-01-01, exclusive end |
| Price policy | Yahoo Finance via yfinance; `auto_adjust=True` |
| Initial equity | $10,000 |
| Formation window | 603 sessions |
| Trading block | Next 126 sessions; final block may be shorter |
| Rolling z-score | 60 observations; sample standard deviation |
| Entry / exit thresholds | Enter beyond ±2.0; exit when absolute z-score is below 0.5 |
| Transaction cost | 10 basis points of each leg's traded notional at entry and exit |
| Entry gross exposure | 100% of equity before entry costs |
| Screening / Holm significance | 5% |
| Minimum formation observations | 252 per stock |

The reported sample contains 1,006 sessions from 2020-01-02 to 2023-12-29. Evaluation starts on 2022-05-24. The four default trading blocks contain 126, 126, 126 and 25 sessions. Each block begins flat, closes any position at its scheduled final close, and passes its ending equity into the next block. No eligible pair means cash for that block, earning zero interest.

### Formation-only selection

1. Diagnose whether each stock is consistent with I(1). Levels use ADF and KPSS with a constant and trend; first differences use a constant. ADF uses AIC lag selection and KPSS uses automatic bandwidth selection. A passing stock must have level ADF p > 0.05, level KPSS p < 0.05, difference ADF p < 0.05, and difference KPSS p > 0.05. Conflicting or unresolved results are excluded. This screen does not prove the integration order; KPSS table bounds are recorded explicitly.
2. Test eligible stock pairs using Engle–Granger cointegration. Preserve the full original family of 120 unordered pairs for Holm adjustment. Screened-out or failed tests contribute p = 1, rather than shrinking the correction family.
3. Require a Holm-adjusted p-value ≤ 0.05, valid diagnostics, residual ADF p < 0.05, and an estimated half-life strictly between 1 and 60 sessions. The ordinary residual ADF p-value is a diagnostic filter, not independent confirmation of cointegration.
4. Rank surviving pairs by adjusted cointegration p-value, raw p-value, residual ADF p-value and half-life. Select one pair, or hold cash if none survives. Pair orientation follows the downloaded column order; both orientations are not separately searched.

OLS estimates the formation relationship `Y = alpha + beta * X + residual`. Alpha and beta remain fixed throughout that trading block. The spread is `Y − alpha − beta * X`.

Formation-half diagnostics report each half's fitted beta, their difference divided by the absolute full-period beta, the second/first variance ratio of the spread constructed with the full-period parameters, and unadjusted subperiod cointegration p-values. They do not change eligibility or ranking. `diagnostic_ok` means these calculations completed; it does not certify a stable relationship.

### Signals and accounting

Rolling statistics include the current close and up to 59 preceding formation observations at the start of a default block. Formation history warms statistics only; it does not create initial holdings. A signal calculated at close t executes at close t+1. Scheduled block-end liquidation overrides pending entries.

A long spread buys Y and shorts beta units of X per unit of Y when beta is positive; a short spread reverses the signs. Fractional quantities are scaled so entry gross notional equals the configured exposure target times current equity. Quantities stay fixed until exit, while marked exposure can drift. Regression hedging does not guarantee dollar neutrality.

Daily gross P&L is `qY * change(Y) + qX * change(X)`. Costs apply to actual absolute traded notional on both legs. Net P&L updates equity, and daily returns use that day's starting equity. The completed-trade ledger reconciles with the equity change; legacy mode also includes any open trade's marked contribution.

Annualized return uses 252 sessions per year. Sharpe uses mean daily net return divided by its sample standard deviation, multiplied by the square root of 252, with a zero risk-free rate. Drawdown includes initial capital as a starting high-water mark. Holding periods in the ledger are calendar days. Zero-volatility cash cases have an undefined Sharpe, reported as NaN.

## Reported results

These figures are transcribed from the local run reported on 2026-10-03. The table preserves the runner's three-decimal display precision; its original CSV files contain the underlying precision. Actual market results were run locally by the project author. Development checks used controlled data and are separate from this market evaluation.

Baseline: ending equity approximately **$9,955.96**, annualized return **−0.28%**, maximum drawdown **−2.92%**, and transaction costs **$39.84**. MS/YUM qualified in the first fold; the other three folds held cash.

| Scenario | Return (%) | Completed trades | Selected folds | Sharpe | Max drawdown (%) | Costs ($) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline walk-forward | −0.440 | 2 | 1/4 | −0.056 | −2.923 | 39.842 |
| Entry 1.5 | −0.579 | 3 | 1/4 | −0.060 | −4.602 | 60.408 |
| Entry 2.5 | −0.369 | 1 | 1/4 | −0.088 | −2.912 | 20.613 |
| Exit 0.25 | 0.279 | 2 | 1/4 | 0.064 | −2.923 | 39.566 |
| Exit 1.0 | −0.107 | 2 | 1/4 | −0.007 | −2.923 | 40.535 |
| Lookback 30 | 0.140 | 4 | 1/4 | 0.042 | −3.538 | 81.255 |
| Lookback 90 | 3.431 | 2 | 1/4 | 0.579 | −2.239 | 39.595 |
| Costs 0 bp | −0.043 | 2 | 1/4 | 0.011 | −2.920 | 0.000 |
| Costs 5 bp | −0.242 | 2 | 1/4 | −0.023 | −2.922 | 19.931 |
| Costs 25 bp | −1.035 | 2 | 1/4 | −0.156 | −2.928 | 99.456 |
| Formation 252 | 0.000 | 0 | 0/4 | NaN | 0.000 | 0.000 |
| Formation 504 | 0.000 | 0 | 0/4 | NaN | 0.000 | 0.000 |
| Trading block 63 | −0.618 | 1 | 1/7 | −0.475 | −1.139 | 19.680 |
| Trading block 252 | −2.743 | 4 | 1/2 | −0.219 | −7.760 | 78.701 |
| Legacy fixed split | 3.823 | 7 | 1/1 | 0.345 | −7.760 | 136.793 |

All cases completed. The 14 walk-forward cases use the same 403 evaluation dates, including the shorter formation-window cases. Each variant changes one configuration value from the baseline. Identical formation selections are cached with defensive copies; changed formation samples or screening policies require new selection.

The baseline remains slightly negative even with zero transaction costs. Changing the lookback from 60 to 90 changes the sign and magnitude of returns, but the positive 90-day result consists of just two trades in one selected fold. The shorter formation windows find no eligible pairs; their flat returns represent no activity. The first selected pair's half-period cointegration p-values, approximately 0.892 and 0.082, do not reject no cointegration at 5%; shorter samples also have less testing power. These observations limit confidence in a persistent edge.

The legacy fixed split uses one 60/40 split, ADF-only integration screening, trading-only rolling warmup, and no forced final liquidation. It retains the same Holm selection gate. Its 3.823% return is a comparison with a different evaluation design, not the current default performance or a one-factor causal experiment.

## Files and reproducibility

| Output | Contents |
| --- | --- |
| `pairs_downloaded_prices.csv` | Price snapshot; `date` index and stock columns |
| `pairs_daily_results.csv` | Daily positions, quantities, exposures, P&L, costs, returns and equity |
| `pairs_trade_ledger.csv` | Completed trades, both legs, costs, calendar holding days and exit reasons |
| `pairs_walk_forward_folds.csv` | Formation/trading dates, selection, diagnostics and block results |
| `pairs_integration_screen.csv` | Per-fold ADF/KPSS outcomes and KPSS bounds |
| `pairs_formation_scan.csv` | Full pair families, failures, raw/adjusted p-values and diagnostics |
| `improved_pairs_equity_curve.png` | Normalized equity curve |
| `improved_pairs_portfolio_value.png` | Portfolio value |
| `improved_pairs_zscore_signals.png` | Z-scores and executed entry markers |
| `robustness_results/pairs_robustness_summary.csv` | All 15 scenarios, settings, metrics, status and error columns |
| `robustness_results/pairs_robustness_folds.csv` | Fold results by scenario |
| `robustness_results/pairs_robustness_log.txt` | Evaluation logs by scenario |
| `robustness_results/pairs_run_metadata.json` | Snapshot/code SHA-256 hashes, package versions, scenarios and timestamp |

Keep the exact snapshot, matching source files, baseline exports, sensitivity outputs and metadata together. Running `main2.py` again downloads prices and overwrites baseline outputs. Running `robustness.py` with its default output directory overwrites previous sensitivity outputs; use `--output-dir` for a separate replay.

`requirements.txt` lists direct dependencies for setup and intentionally does not claim to be a tested version lock. Capture the environment that produced the reported results:

```bash
python3 --version > python-version.txt
python3 -m pip freeze > requirements-lock.txt
```

For a later replay, use the recorded Python version and install the saved lock file in a new environment. Verify the snapshot/code hashes against the archived metadata. Matching code and prices reduce variation, but platform and numerical-library differences can still affect borderline statistical decisions. The supplied source ZIP does not contain the author's price snapshot, original result CSVs, environment lock or plots; those remain in the local project folder.

## Research limits

The handpicked current 16-stock universe is not a point-in-time historical universe. Historical adjusted closes are synthetic research prices and do not implement a broker cash, dividend or corporate-action ledger. Borrow availability, borrow fees, financing, bid–ask spreads, slippage, market impact, taxes and margin constraints are not modeled. The recorded turnover cost is therefore not a complete implementation-cost estimate.

The data and settings were inspected and revised during development. These are retrospective chronological validation and sensitivity results, not a pristine untouched holdout. Holm correction applies within each formation family; it does not adjust across all folds, settings, universe choices or development attempts. The sensitivity runner reports every preset and does not select a winning configuration. Neither a positive variant nor an all-cash result establishes a trading edge.

The completed project demonstrates statistical screening, chronological evaluation, explicit two-leg accounting, audit exports and honest reporting of parameter sensitivity. Its main result is the limited evidence for this specification in this sample.
