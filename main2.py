"""Pairs trading research: formation-only selection and two-leg accounting.

Default: rolling formation windows followed by disjoint trading blocks.
Set backtest_mode = "fixed_split" for the legacy ADF-only Step 5 baseline.

Signals use close t and execute at close t+1. Fractional synthetic shares
are sized using adjusted closes. Costs apply to both legs' traded notional.
Borrow fees, financing, bid-ask spread and slippage are not modeled. Adjusted
prices encode corporate-action adjustments, not a broker cash/dividend ledger.
The 16-stock universe is handpicked and is not a point-in-time universe.
The out-of-sample data have already been inspected during project development;
these are retrospective validation results, not a pristine untouched holdout.
Holm correction is within each formation fold, not pooled across all folds.
Cash blocks earn zero interest in this simplified model.
"""
from __future__ import annotations

import itertools
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
import statsmodels.api as sm

# Configuration: keep these fixed before inspecting walk-forward performance.
tickers = [
    "XOM", "CVX", "KO", "PEP", "JPM", "BAC", "GS", "MS",
    "GOOG", "MSFT", "AAPL", "META", "WMT", "TGT", "MCD", "YUM",
]
start_date = "2020-01-01"
end_date = "2024-01-01"  # Exclusive download boundary.
initial_capital = 10_000
entry_z = 2.0
exit_z = 0.5
rolling_window = 60
transaction_cost_rate = 0.001
familywise_alpha = 0.05
i1_screen_alpha = 0.05
min_training_points = 252
gross_exposure_target = 1.0  # Entry gross / equity before costs; can drift.
backtest_mode = "walk_forward"  # Or "fixed_split" for the Step 5 baseline.
formation_window = 603  # Retains the original first formation sample.
trading_window = 126  # Roughly six trading months; last block may be shorter.

TRADE_COLUMNS = [
    "fold_id", "stock_1", "stock_2", "entry_date", "exit_date", "direction",
    "entry_y_price", "entry_x_price", "exit_y_price", "exit_x_price",
    "y_quantity", "x_quantity", "gross_pnl", "transaction_cost", "net_pnl",
    "holding_days", "exit_reason",
]

def kpss_diagnostic(series: pd.Series, regression: str):
    """Return KPSS results and label clipped p-values as bounds, not exact values."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", InterpolationWarning)
        result = kpss(series, regression=regression, nlags="auto")
    bounded = False
    for warning in captured:
        if issubclass(warning.category, InterpolationWarning):
            bounded = True
        else:
            warnings.warn(warning.message, warning.category, stacklevel=2)
    pvalue = float(result[1])
    if not np.isfinite([result[0], pvalue]).all() or not 0 <= pvalue <= 1:
        raise ValueError("KPSS returned an invalid statistic or p-value.")
    bound = "interpolated"
    if bounded:
        bound = "at_most" if pvalue <= 0.01 else "at_least"
    return result, bound


def screen_integration_order(formation_prices: pd.DataFrame, use_kpss: bool = True) -> pd.DataFrame:
    """Formation-only ADF/KPSS screen for a pattern consistent with I(1).

    Level test: constant + linear trend; difference test: constant only.
    ADF uses AIC lags; KPSS uses automatic bandwidth selection.
    Both tests must support the screen when use_kpss=True.
    Failure to reject a null does not prove the integration order.
    These unadjusted diagnostic p-values are not a separate family-level
    significance claim. Borderline or failed screens are excluded.
    """
    results = []
    for ticker in formation_prices.columns:
        prices_series = formation_prices[ticker].dropna()
        row = {
            "ticker": ticker,
            "formation_observations": len(prices_series),
            "level_adf_regression": "ct",
            "difference_adf_regression": "c",
            "level_adf_pvalue": np.nan,
            "difference_adf_pvalue": np.nan,
            "level_adf_lags": np.nan,
            "difference_adf_lags": np.nan,
            "level_kpss_pvalue": np.nan,
            "difference_kpss_pvalue": np.nan,
            "level_kpss_pvalue_bound": "not_run",
            "difference_kpss_pvalue_bound": "not_run",
            "level_kpss_lags": np.nan,
            "difference_kpss_lags": np.nan,
            "screen_method": "ADF+KPSS" if use_kpss else "ADF-only legacy",
            "i1_screen_pass": False,
            "screen_status": "test_failed",
        }
        try:
            if formation_prices[ticker].isna().any():
                raise ValueError("Formation prices have gaps; no time-compressing fill is used.")
            if len(prices_series) < min_training_points:
                raise ValueError("Too few formation observations for I(1) screening.")
            if not np.isfinite(prices_series.to_numpy(dtype=float)).all():
                raise ValueError("Formation prices contain nonfinite values.")
            level_result = adfuller(prices_series, regression="ct", autolag="AIC")
            difference_result = adfuller(
                prices_series.diff().dropna(), regression="c", autolag="AIC"
            )
            level_p = float(level_result[1])
            difference_p = float(difference_result[1])
            if not np.isfinite([level_p, difference_p]).all() or not (
                0 <= level_p <= 1 and 0 <= difference_p <= 1
            ):
                raise ValueError("ADF returned an invalid p-value.")
            row.update({
                "level_adf_pvalue": level_p,
                "difference_adf_pvalue": difference_p,
                "level_adf_lags": int(level_result[2]),
                "difference_adf_lags": int(difference_result[2]),
            })
            level_kpss_p = difference_kpss_p = np.nan
            if use_kpss:
                level_kpss, level_bound = kpss_diagnostic(prices_series, "ct")
                difference_kpss, difference_bound = kpss_diagnostic(
                    prices_series.diff().dropna(), "c"
                )
                level_kpss_p = float(level_kpss[1])
                difference_kpss_p = float(difference_kpss[1])
                row.update({
                    "level_kpss_pvalue": level_kpss_p,
                    "difference_kpss_pvalue": difference_kpss_p,
                    "level_kpss_pvalue_bound": level_bound,
                    "difference_kpss_pvalue_bound": difference_bound,
                    "level_kpss_lags": int(level_kpss[2]),
                    "difference_kpss_lags": int(difference_kpss[2]),
                })
            if level_p <= i1_screen_alpha:
                row["screen_status"] = "level_unit_root_rejected"
            elif difference_p >= i1_screen_alpha:
                row["screen_status"] = "difference_unit_root_not_rejected"
            elif use_kpss and level_kpss_p >= i1_screen_alpha:
                row["screen_status"] = "inconclusive_level_stationarity"
            elif use_kpss and difference_kpss_p <= i1_screen_alpha:
                row["screen_status"] = "difference_stationarity_rejected"
            else:
                row["screen_status"] = "consistent_with_i1"
                row["i1_screen_pass"] = True
        except Exception as exc:
            print(f"I(1) screen failed for {ticker}: {exc}")
        results.append(row)
    return pd.DataFrame(results).set_index("ticker")

def holm_adjust_pvalues(pvalues) -> np.ndarray:
    """Holm step-down p-value adjustment, returned in original pair order.

    Apply to the full attempted family, before any diagnostic filtering.
    Failed cointegration tests must be supplied conservatively as p=1.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.ndim != 1 or not np.isfinite(pvalues).all():
        raise ValueError("P-values must be a finite one-dimensional array.")
    if ((pvalues < 0) | (pvalues > 1)).any():
        raise ValueError("P-values must be between zero and one.")
    n_tests = len(pvalues)
    if n_tests == 0:
        return pvalues.copy()
    order = np.argsort(pvalues)
    # Sorted p_(i) is multiplied by m-i+1; cumulative max preserves
    # Holm's step-down constraint, then values are capped at 1.
    adjusted_sorted = np.minimum(
        1.0,
        np.maximum.accumulate(pvalues[order] * np.arange(n_tests, 0, -1)),
    )
    adjusted = np.empty(n_tests, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted

def estimate_spread_parameters(
    y: pd.Series,
    x: pd.Series
) -> tuple[float, float]:
    """
    Estimate intercept alpha and hedge ratio beta from formation OLS:

        y = alpha + beta * x + epsilon
    """

    x_const = sm.add_constant(x)

    model = sm.OLS(
        y,
        x_const
    ).fit()

    return float(model.params["const"]), float(model.params[x.name])

def compute_spread(
    y: pd.Series,
    x: pd.Series,
    alpha: float,
    beta: float
) -> pd.Series:
    """
    Construct the residual spread from the fitted formation regression.
    """

    return y - alpha - beta * x

def compute_rolling_zscore(
    spread: pd.Series,
    window: int
) -> pd.Series:
    """
    Rolling z-score.
    """

    rolling_mean = spread.rolling(
        window
    ).mean()

    rolling_std = spread.rolling(
        window
    ).std()

    z = (
        spread - rolling_mean
    ) / rolling_std

    return z

def calculate_half_life(
    spread: pd.Series
) -> float:
    """
    Estimate half-life of mean reversion
    using an AR(1)-style regression.
    """

    spread_lag = (
        spread
        .shift(1)
        .dropna()
    )

    spread_ret = (
        spread
        .diff()
        .dropna()
    )

    aligned = pd.concat(
        [spread_lag, spread_ret],
        axis=1
    ).dropna()

    if aligned.empty:
        return np.nan

    x = sm.add_constant(
        aligned.iloc[:, 0]
    )

    y = aligned.iloc[:, 1]

    model = sm.OLS(
        y,
        x
    ).fit()

    beta = model.params.iloc[1]

    if beta >= 0:
        return np.nan

    half_life = (
        -np.log(2)
        / beta
    )

    return float(half_life)

STABILITY_COLUMNS = [
    "beta_first_half", "beta_second_half", "relative_beta_change",
    "spread_variance_ratio", "coint_pvalue_first_half", "coint_pvalue_second_half",
]


def pair_stability_diagnostics(formation: pd.DataFrame, pair: pd.Series) -> dict:
    """Formation-half diagnostics; descriptive, not an additional selection gate.

    Cointegration p-values here are unadjusted and depend on half-sample
    integration assumptions. They are not independent confirmation or
    family-level significance claims. No thresholds are fitted to returns.
    Spread variance uses the full-formation frozen alpha/beta in both halves.
    """
    output = {name: np.nan for name in STABILITY_COLUMNS}
    output["stability_status"] = "diagnostic_failed"
    try:
        midpoint = len(formation) // 2
        halves = [formation.iloc[:midpoint], formation.iloc[midpoint:]]
        if min(map(len, halves)) < 60:
            raise ValueError("Too few observations for formation-half diagnostics.")
        y_name, x_name = pair["stock_1"], pair["stock_2"]
        alpha, beta = float(pair["intercept"]), float(pair["hedge_ratio"])
        half_betas, variances, half_pvalues = [], [], []
        for half in halves:
            y, x = half[y_name], half[x_name]
            _, half_beta = estimate_spread_parameters(y, x)
            _, pvalue, _ = coint(y, x)
            variance = float(compute_spread(y, x, alpha, beta).var())
            if not np.isfinite([half_beta, pvalue, variance]).all() or not 0 <= pvalue <= 1:
                raise ValueError("Invalid formation-half diagnostic result.")
            half_betas.append(float(half_beta))
            variances.append(variance)
            half_pvalues.append(float(pvalue))
        output.update({
            "beta_first_half": half_betas[0], "beta_second_half": half_betas[1],
            "relative_beta_change": abs(half_betas[1] - half_betas[0]) / abs(beta)
                if abs(beta) > 1e-12 else np.nan,
            "spread_variance_ratio": max(variances) / min(variances)
                if min(variances) > 0 else np.nan,
            "coint_pvalue_first_half": half_pvalues[0],
            "coint_pvalue_second_half": half_pvalues[1],
            "stability_status": "diagnostic_ok",
        })
        if not np.isfinite([output["relative_beta_change"], output["spread_variance_ratio"]]).all():
            output["stability_status"] = "diagnostic_inconclusive"
    except Exception as exc:
        print(f"Formation-half diagnostics failed for {pair['stock_1']}/{pair['stock_2']}: {exc}")
    return output


def select_pair(formation: pd.DataFrame, use_kpss: bool = True):
    """Select using only this formation sample; return None if no pair qualifies."""
    i1_df = screen_integration_order(formation, use_kpss=use_kpss)
    pair_results = []

    for s1, s2 in itertools.combinations(formation.columns, 2):
        y_train = formation[s1]
        x_train = formation[s2]
        pair_i1_pass = bool(i1_df.loc[s1, "i1_screen_pass"]
                            and i1_df.loc[s2, "i1_screen_pass"])
        result = {
            "stock_1": s1,
            "stock_2": s2,
            "coint_pvalue": 1.0,
            "adf_pvalue": np.nan,
            "intercept": np.nan,
            "hedge_ratio": np.nan,
            "half_life": np.nan,
            "scan_status": "cointegration_failed",
            "i1_screen_pass": pair_i1_pass,
        }

        if not pair_i1_pass:
            # Keep excluded hypotheses in the original family conservatively
            # at p=1; do not apply an I(1) cointegration test to unsupported inputs.
            result["scan_status"] = "i1_screen_failed"
            pair_results.append(result)
            continue

        try:
            coint_stat, pvalue, _ = coint(y_train, x_train)
            if not np.isfinite(pvalue) or not 0 <= pvalue <= 1:
                raise ValueError("Cointegration returned an invalid p-value.")
            result["coint_pvalue"] = float(pvalue)
        except Exception as exc:
            # Keep the attempted hypothesis in the correction family at p=1.
            print(f"Cointegration failed for {s1}/{s2}: {exc}")
        else:
            try:
                alpha, beta = estimate_spread_parameters(y_train, x_train)
                if not np.isfinite([alpha, beta]).all():
                    raise ValueError("Formation OLS returned nonfinite parameters.")
                spread_train = compute_spread(y_train, x_train, alpha, beta)
                adf_pvalue = adfuller(spread_train.dropna())[1]
                hl = calculate_half_life(spread_train.dropna())
                result.update({
                    "adf_pvalue": adf_pvalue,
                    "intercept": alpha,
                    "hedge_ratio": beta,
                    "half_life": hl,
                    "scan_status": "ok",
                })
            except Exception as exc:
                # Retain the valid cointegration p-value even if diagnostics fail.
                result["scan_status"] = "diagnostics_failed"
                print(f"Diagnostics failed for {s1}/{s2}: {exc}")

        pair_results.append(result)

    if not pair_results:
        raise ValueError("At least two stocks are required for pair selection.")

    pairs_df = pd.DataFrame(pair_results)
    pairs_df["coint_pvalue_holm"] = holm_adjust_pvalues(pairs_df["coint_pvalue"])
    pairs_df["coint_reject_holm"] = pairs_df["coint_pvalue_holm"] <= familywise_alpha

    # The residual ADF screen is retained from the previous version as a
    # diagnostic filter. Its ordinary p-value is not independent confirmation
    # of cointegration; the adjusted Engle-Granger test supplies that gate.
    filtered_pairs = pairs_df[
        pairs_df["coint_reject_holm"]
        & pairs_df["i1_screen_pass"]
        & (pairs_df["scan_status"] == "ok")
        & (pairs_df["adf_pvalue"] < 0.05)
        & (pairs_df["half_life"].notna())
        & (pairs_df["half_life"] > 1)
        & (pairs_df["half_life"] < 60)
    ].copy()


    candidates = filtered_pairs.sort_values(
        ["coint_pvalue_holm", "coint_pvalue", "adf_pvalue", "half_life"]
    )
    # Diagnose every eligible candidate before reporting. Keep diagnostics
    # separate from the pre-existing ranking and trade eligibility rules.
    for column in STABILITY_COLUMNS:
        pairs_df[column] = np.nan
    pairs_df["stability_status"] = "not_assessed"
    for index, candidate in candidates.iterrows():
        diagnostics = pair_stability_diagnostics(formation, candidate)
        for column, value in diagnostics.items():
            pairs_df.loc[index, column] = value
    candidates = pairs_df.loc[candidates.index]
    best = None if candidates.empty else candidates.iloc[0].copy()
    return best, i1_df, pairs_df, len(candidates)

def generate_positions(zscores: pd.Series) -> pd.Series:
    position = 0
    positions = []
    for z in zscores:
        if not np.isfinite(z):
            positions.append(position)
            continue
        if position == 0:
            if z < -entry_z:
                position = 1
            elif z > entry_z:
                position = -1
        elif abs(z) < exit_z:
            position = 0
        positions.append(position)
    return pd.Series(positions, index=zscores.index, dtype=int)


def build_signals(formation: pd.DataFrame, trading: pd.DataFrame,
                  pair: pd.Series, warmup: bool) -> pd.DataFrame:
    y_name, x_name = pair["stock_1"], pair["stock_2"]
    alpha, beta = float(pair["intercept"]), float(pair["hedge_ratio"])
    # Recompute history using this fold's frozen parameters. History warms
    # rolling statistics only: holdings begin flat at each block's start.
    history = formation.tail(rolling_window - 1) if warmup else formation.iloc[:0]
    combined = pd.concat([history[[y_name, x_name]], trading[[y_name, x_name]]])
    spread = compute_spread(combined[y_name], combined[x_name], alpha, beta)
    zscores = compute_rolling_zscore(spread, rolling_window).reindex(trading.index)
    zscores = zscores.replace([np.inf, -np.inf], np.nan)
    signals = pd.DataFrame({
        "y_price": trading[y_name], "x_price": trading[x_name],
        "spread": spread.reindex(trading.index), "zscore": zscores,
    }, index=trading.index)
    signals["position"] = generate_positions(zscores)
    signals["execution_position"] = signals["position"].shift(1).fillna(0).astype(int)
    return signals


def backtest_pair(signals: pd.DataFrame, beta: float, starting_equity: float,
                  *, force_close: bool, fold_id: int = 1,
                  stock_1: str = "Y", stock_2: str = "X"):
    """Fixed entry shares, next-close execution, mark-to-market before trading."""
    if signals.empty or starting_equity <= 0:
        raise ValueError("Backtest requires nonempty signals and positive capital.")
    if not np.isfinite(signals[["y_price", "x_price"]].to_numpy()).all():
        raise ValueError("Trading prices must be finite.")
    if (signals[["y_price", "x_price"]] <= 0).any().any():
        raise ValueError("Trading prices must be positive.")
    if not np.isfinite(beta):
        raise ValueError("Hedge ratio must be finite.")
    signals = signals.copy()
    equity = float(starting_equity)
    current_position = 0
    y_qty = x_qty = 0.0
    previous_y = previous_x = None
    active_trade = None
    trades, rows = [], []

    for day_number, (date, row) in enumerate(signals.iterrows()):
        y_price, x_price = float(row.y_price), float(row.x_price)
        start_equity = equity
        gross_pnl = 0.0 if previous_y is None else (
            y_qty * (y_price - previous_y) + x_qty * (x_price - previous_x)
        )
        equity += gross_pnl
        target = int(row.execution_position)
        if target not in (-1, 0, 1):
            raise ValueError("Execution position must be -1, 0, or 1.")
        scheduled_exit = force_close and day_number == len(signals) - 1
        if scheduled_exit:
            # This boundary is scheduled in advance. Liquidate at its close,
            # ignore any pending entry, and include both legs' exit costs.
            target = 0
        traded_notional = 0.0

        if target != current_position:
            if current_position != 0:
                if active_trade is None:
                    raise RuntimeError("Open position is missing trade details.")
                closing_notional = abs(y_qty * y_price) + abs(x_qty * x_price)
                traded_notional += closing_notional
                gross_trade = (
                    y_qty * (y_price - active_trade["entry_y_price"])
                    + x_qty * (x_price - active_trade["entry_x_price"])
                )
                trade_cost = active_trade["entry_cost"] + (
                    closing_notional * transaction_cost_rate
                )
                trades.append({
                    "fold_id": fold_id, "stock_1": stock_1, "stock_2": stock_2,
                    "entry_date": active_trade["entry_date"], "exit_date": date,
                    "direction": active_trade["direction"],
                    "entry_y_price": active_trade["entry_y_price"],
                    "entry_x_price": active_trade["entry_x_price"],
                    "exit_y_price": y_price, "exit_x_price": x_price,
                    "y_quantity": y_qty, "x_quantity": x_qty,
                    "gross_pnl": gross_trade, "transaction_cost": trade_cost,
                    "net_pnl": gross_trade - trade_cost,
                    "holding_days": (date - active_trade["entry_date"]).days,
                    "exit_reason": "scheduled_block_end" if scheduled_exit else (
                        "signal_exit" if target == 0 else "signal_reversal"
                    ),
                })
                y_qty = x_qty = 0.0
                active_trade = None

            if target != 0:
                if equity <= 0:
                    raise ValueError("Portfolio equity has been depleted.")
                scale = equity * gross_exposure_target / (y_price + abs(beta * x_price))
                y_qty, x_qty = target * scale, -target * beta * scale
                opening_notional = abs(y_qty * y_price) + abs(x_qty * x_price)
                traded_notional += opening_notional
                active_trade = {
                    "entry_date": date,
                    "direction": "Long Spread" if target == 1 else "Short Spread",
                    "entry_y_price": y_price, "entry_x_price": x_price,
                    "entry_cost": opening_notional * transaction_cost_rate,
                }
            current_position = target

        cost = traded_notional * transaction_cost_rate
        equity -= cost
        if equity <= 0:
            raise ValueError("Portfolio equity has been depleted; no margin model is implemented.")
        net_pnl = equity - start_equity
        rows.append({
            "execution_position": current_position, "gross_pnl": gross_pnl,
            "transaction_cost": cost, "net_pnl": net_pnl,
            "net_strategy_return": net_pnl / start_equity,
            "portfolio_value": equity,
            "gross_exposure": abs(y_qty * y_price) + abs(x_qty * x_price),
            "net_exposure": y_qty * y_price + x_qty * x_price,
            "y_quantity": y_qty, "x_quantity": x_qty,
            "traded_notional": traded_notional,
        })
        previous_y, previous_x = y_price, x_price

    accounting = pd.DataFrame(rows, index=signals.index)
    for column in accounting:
        signals[column] = accounting[column]
    signals["position_change"] = signals.execution_position.diff().abs().fillna(0)
    signals["fold_id"] = fold_id
    signals["stock_1"], signals["stock_2"] = stock_1, stock_2
    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    open_net_pnl = open_entry_cost = 0.0
    if active_trade is not None:
        open_entry_cost = active_trade["entry_cost"]
        open_net_pnl = (
            y_qty * (previous_y - active_trade["entry_y_price"])
            + x_qty * (previous_x - active_trade["entry_x_price"])
            - open_entry_cost
        )
    if not np.isclose(trades_df.net_pnl.sum() + open_net_pnl,
                      equity - starting_equity, rtol=1e-9, atol=1e-6):
        raise RuntimeError("Trade P&L does not reconcile with equity.")
    if not np.isclose(trades_df.transaction_cost.sum() + open_entry_cost,
                      signals.transaction_cost.sum(), rtol=1e-9, atol=1e-6):
        raise RuntimeError("Trade costs do not reconcile with daily costs.")
    if force_close and (active_trade is not None or y_qty != 0 or x_qty != 0):
        raise RuntimeError("Scheduled liquidation left an open position.")
    return signals, trades_df, open_net_pnl


def cash_block(index: pd.Index, equity: float, fold_id: int) -> pd.DataFrame:
    """No qualifying pair: keep all trading dates and carry capital unchanged."""
    block = pd.DataFrame(index=index)
    for column in ["y_price", "x_price", "spread", "zscore"]:
        block[column] = np.nan
    for column in ["position", "execution_position", "gross_pnl", "transaction_cost",
                   "net_pnl", "net_strategy_return", "gross_exposure", "net_exposure",
                   "y_quantity", "x_quantity", "traded_notional", "position_change"]:
        block[column] = 0
    block["portfolio_value"], block["fold_id"] = equity, fold_id
    block["stock_1"], block["stock_2"] = "", ""
    return block


def run_evaluation(prices: pd.DataFrame, mode: str = "walk_forward",
                   *, first_test_index: int | None = None):
    """Rolling windows, chronological selection, and a continuous capital path."""
    if mode not in ("walk_forward", "fixed_split"):
        raise ValueError("Mode must be walk_forward or fixed_split.")
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError("Price dates must be unique and chronological.")
    if mode == "walk_forward":
        first_test = formation_window if first_test_index is None else int(first_test_index)
        train_size, block_size = formation_window, trading_window
    else:
        if first_test_index is not None:
            raise ValueError("An explicit trading start is supported only for walk-forward mode.")
        first_test = int(len(prices) * 0.6)
        train_size, block_size = first_test, len(prices) - first_test
    if first_test < train_size:
        raise ValueError("Trading start leaves too little history for the formation window.")
    if train_size < max(min_training_points, rolling_window) or first_test >= len(prices):
        raise ValueError("Not enough prices for the configured formation and trading windows.")
    if block_size < 1:
        raise ValueError("Trading window must be positive.")

    equity = float(initial_capital)
    daily_blocks, all_trades, fold_rows, screens, scans = [], [], [], [], []
    open_net_pnl = 0.0
    for fold_id, start in enumerate(range(first_test, len(prices), block_size), start=1):
        stop = min(start + block_size, len(prices))
        formation = prices.iloc[start - train_size:start].copy()
        trading = prices.iloc[start:stop].copy()
        if formation.index[-1] >= trading.index[0]:
            raise RuntimeError("Formation and trading samples overlap.")
        print(f"\nFold {fold_id}: formation {formation.index[0].date()} to "
              f"{formation.index[-1].date()}; trading {trading.index[0].date()} "
              f"to {trading.index[-1].date()} ({len(trading)} sessions)", flush=True)
        pair, i1_df, pairs_df, eligible_count = select_pair(formation, use_kpss=mode == "walk_forward")
        screen = i1_df.reset_index()
        for frame in [screen, pairs_df]:
            frame["fold_id"] = fold_id
            frame["formation_start"] = formation.index[0]
            frame["formation_end"] = formation.index[-1]
        screens.append(screen)
        scans.append(pairs_df)
        method = "ADF+KPSS" if mode == "walk_forward" else "ADF-only legacy"
        print(f"I(1) screen ({method}): {int(i1_df.i1_screen_pass.sum())}/{len(i1_df)} stocks; "
              f"Holm family: {len(pairs_df)}; eligible pairs: {eligible_count}")
        print("Screen outcomes:", i1_df.screen_status.value_counts().to_dict())
        start_equity = equity
        row = {
            "fold_id": fold_id, "formation_start": formation.index[0],
            "formation_end": formation.index[-1], "trading_start": trading.index[0],
            "trading_end": trading.index[-1], "trading_days": len(trading),
            "eligible_pairs": eligible_count, "holm_family_size": len(pairs_df),
            "stock_1": "", "stock_2": "", "intercept": np.nan,
            "hedge_ratio": np.nan, "coint_pvalue_holm": np.nan,
            "starting_equity": start_equity,
        }
        if pair is None:
            block = cash_block(trading.index, equity, fold_id)
            trades = pd.DataFrame(columns=TRADE_COLUMNS)
            row["status"] = "cash_no_eligible_pair"
            print("No qualifying pair: holding cash for this block.")
        else:
            y_name, x_name = pair.stock_1, pair.stock_2
            alpha, beta = float(pair.intercept), float(pair.hedge_ratio)
            print(f"Selected {y_name}/{x_name}: Holm p={pair.coint_pvalue_holm:.6f}; "
                  f"alpha={alpha:.4f}; beta={beta:.4f}")
            diagnostics = {name: pair.get(name, np.nan) for name in STABILITY_COLUMNS}
            row.update(diagnostics)
            row["stability_status"] = pair.get("stability_status", "not_assessed")
            print("Formation-half diagnostics (subperiod p-values are unadjusted):")
            print(f"  beta: {diagnostics['beta_first_half']:.4f} -> "
                  f"{diagnostics['beta_second_half']:.4f}; relative change: "
                  f"{diagnostics['relative_beta_change']:.2%}; variance ratio: "
                  f"{diagnostics['spread_variance_ratio']:.2f}")
            print(f"  cointegration p-values: {diagnostics['coint_pvalue_first_half']:.4f}, "
                  f"{diagnostics['coint_pvalue_second_half']:.4f}")
            signals = build_signals(formation, trading, pair, warmup=mode == "walk_forward")
            block, trades, open_net_pnl = backtest_pair(
                signals, beta, equity, force_close=mode == "walk_forward",
                fold_id=fold_id, stock_1=y_name, stock_2=x_name,
            )
            row.update({"status": "selected", "stock_1": y_name, "stock_2": x_name,
                        "intercept": alpha, "hedge_ratio": beta,
                        "coint_pvalue_holm": float(pair.coint_pvalue_holm)})
        equity = float(block.portfolio_value.iloc[-1])
        row.update({"ending_equity": equity, "net_pnl": equity - start_equity,
                    "return_pct": 100 * (equity / start_equity - 1),
                    "transaction_cost": block.transaction_cost.sum(),
                    "completed_trades": len(trades)})
        print(f"Block return: {row['return_pct']:.2f}%; ending equity: ${equity:,.2f}; "
              f"completed trades: {len(trades)}")
        daily_blocks.append(block)
        if not trades.empty:
            all_trades.append(trades)
        fold_rows.append(row)

    daily = pd.concat(daily_blocks)
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(columns=TRADE_COLUMNS)
    daily["equity_curve"] = daily.portfolio_value / initial_capital
    if not daily.index.equals(prices.index[first_test:]):
        raise RuntimeError("Trading dates were duplicated or dropped.")
    if not np.isclose(trades.net_pnl.sum() + open_net_pnl,
                      equity - initial_capital, rtol=1e-9, atol=1e-6):
        raise RuntimeError("Combined ledger does not reconcile with portfolio equity.")
    return daily, trades, pd.DataFrame(fold_rows), pd.concat(screens, ignore_index=True), pd.concat(scans, ignore_index=True)


def performance_summary(daily: pd.DataFrame, trades: pd.DataFrame) -> dict:
    returns = daily.net_strategy_return
    equity_curve = daily.portfolio_value / initial_capital
    # Include initial capital as the starting high-water mark.
    peak = equity_curve.cummax().clip(lower=1.0)
    mdd = float((equity_curve / peak - 1).min())
    annual_return = float(equity_curve.iloc[-1] ** (252 / len(daily)) - 1)
    sharpe = np.nan if len(returns) < 2 or returns.std() == 0 else (
        np.sqrt(252) * returns.mean() / returns.std()
    )
    return {
        "Trading days": len(daily), "Completed trades": len(trades),
        "Total return (%)": 100 * (equity_curve.iloc[-1] - 1),
        "Annualized return (%)": 100 * annual_return, "Sharpe ratio": sharpe,
        "Max drawdown (%)": 100 * mdd,
        "Calmar ratio": annual_return / abs(mdd) if mdd != 0 else np.nan,
        "Starting value ($)": initial_capital, "Ending value ($)": daily.portfolio_value.iloc[-1],
        "Total transaction costs ($)": daily.transaction_cost.sum(),
        "Win rate (%)": 100 * (trades.net_pnl > 0).mean() if len(trades) else np.nan,
        "Average trade P&L ($)": trades.net_pnl.mean() if len(trades) else np.nan,
        "Median holding period (calendar days)": trades.holding_days.median() if len(trades) else np.nan,
        "Open trades at end": int(daily.execution_position.iloc[-1] != 0),
    }


def save_plots(daily: pd.DataFrame, folds: pd.DataFrame, mode: str) -> None:
    for column, ylabel, filename in [
        ("equity_curve", "Equity / initial capital", "improved_pairs_equity_curve.png"),
        ("portfolio_value", "Portfolio value ($)", "improved_pairs_portfolio_value.png"),
    ]:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(daily.index, daily[column], linewidth=1.5)
        if column == "equity_curve":
            ax.axhline(1, linestyle="--", linewidth=1)
        for start in folds.trading_start.iloc[1:]:
            ax.axvline(start, color="gray", linestyle=":", alpha=0.5)
        ax.set(title=f"Pairs trading: {mode.replace('_', ' ')}", ylabel=ylabel)
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(filename, dpi=150)
        plt.show()

    fig, ax = plt.subplots(figsize=(14, 5))
    for fold in folds.itertuples(index=False):
        block = daily[daily.fold_id == fold.fold_id]
        label = f"Fold {fold.fold_id}: {fold.stock_1}/{fold.stock_2}" if fold.status == "selected" else f"Fold {fold.fold_id}: cash"
        ax.plot(block.index, block.zscore, linewidth=1.2, label=label)
        prior = block.execution_position.shift(1).fillna(0)
        for side, marker in [(1, "^"), (-1, "v")]:
            entries = (block.execution_position == side) & (prior != side)
            ax.scatter(block.index[entries], block.loc[entries, "zscore"], marker=marker, s=45)
    for value in [entry_z, -entry_z]:
        ax.axhline(value, linestyle="--", linewidth=1, color="gray")
    for value in [exit_z, -exit_z]:
        ax.axhline(value, linestyle=":", linewidth=1, color="gray")
    ax.set(title="Rolling z-scores by trading block (markers: executed entries)", ylabel="Z-score")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("improved_pairs_zscore_signals.png", dpi=150)
    plt.show()


def main() -> None:
    print("Downloading price data...")
    # Explicit adjustment policy avoids depending on yfinance version defaults.
    prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]
    # Do not choose a historical universe by completeness of future prices.
    # Formation gaps are excluded within that fold; missing held-leg trading
    # prices raise an error rather than trigger hindsight pair substitution.
    prices = prices.sort_index().dropna(how="all")
    if prices.empty or prices.shape[1] < 2:
        raise ValueError("Download did not provide at least two complete price series.")
    print(f"Downloaded {prices.shape[1]} stocks, {len(prices)} trading days.")
    if backtest_mode == "walk_forward":
        print(f"Design: trailing {formation_window} formation sessions; "
              f"next {trading_window} trading sessions; scheduled liquidation at block end.")
        print("Rolling statistics use formation history; holdings begin flat each block.")
        print("I(1) screening uses ADF+KPSS; unresolved screens are excluded.")
        print("KPSS p-value bounds are recorded in pairs_integration_screen.csv.")
    else:
        print("Design: legacy ADF-only 60/40 baseline; test-only warmup; no forced final exit.")
    daily, trades, folds, screens, scans = run_evaluation(prices, backtest_mode)
    # Cache exactly the downloaded sample to support reproducibility checks.
    prices.to_csv("pairs_downloaded_prices.csv", index_label="date")
    daily.to_csv("pairs_daily_results.csv", index_label="date")
    trades.to_csv("pairs_trade_ledger.csv", index=False)
    folds.to_csv("pairs_walk_forward_folds.csv", index=False)
    screens.to_csv("pairs_integration_screen.csv", index=False)
    scans.to_csv("pairs_formation_scan.csv", index=False)
    print("\nWalk-forward Fold Summary" if backtest_mode == "walk_forward" else "\nFixed-split Summary")
    print(folds.to_string(index=False))
    print("\nStrategy Performance Summary\n" + "-" * 50)
    print(f"Evaluation mode: {backtest_mode}")
    for label, value in performance_summary(daily, trades).items():
        print(f"{label + ':':40} {value:,.2f}")
    print("\nCompleted Trade Ledger\n" + "-" * 100)
    print(trades.to_string(index=False) if not trades.empty else "No completed trades.")
    print("\nSaved daily results, ledger, fold summary, formation scans and downloaded prices.")
    print("Retrospective validation: settings were revised after inspecting this historical sample.")
    save_plots(daily, folds, backtest_mode)


if __name__ == "__main__":
    main()
