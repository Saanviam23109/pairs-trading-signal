"""Preset sensitivity analysis using the exact cached price sample.

Put beside main2.py (or main.py), then run: python3 robustness.py
This never downloads prices, selects a winning configuration, or edits strategy files.
The walk-forward variants use the same evaluation dates. Pair selection is
memoized only for identical formation data and screening policy, with deep
copies returned so fold reporting cannot mutate the cache.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

CONFIG_NAMES = (
    "entry_z", "exit_z", "rolling_window", "transaction_cost_rate",
    "formation_window", "trading_window",
)


def load_strategy():
    """Load the adjacent strategy file, preferring the user's main2.py name."""
    folder = Path(__file__).resolve().parent
    path = folder / "main2.py"
    if not path.is_file():
        path = folder / "main.py"
    if not path.is_file():
        raise SystemExit(f"Put the updated main2.py or main.py beside {Path(__file__).name}.")
    # Check compatibility without executing a legacy script's download/backtest.
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assigned = {target.id for node in tree.body if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name)}
    required = {"run_evaluation", "performance_summary", "select_pair"}
    evaluation = functions.get("run_evaluation")
    supports_start = evaluation is not None and any(
        arg.arg == "first_test_index" for arg in evaluation.args.kwonlyargs
    )
    if not required.issubset(functions) or not set(CONFIG_NAMES).issubset(assigned) or not supports_start:
        raise SystemExit(
            f"{path} is an older strategy version. Replace that file with the matching "
            "Step 8 strategy file, then run python3 robustness.py again."
        )
    spec = importlib.util.spec_from_file_location("pairs_strategy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


strategy = load_strategy()


def build_scenarios() -> list[dict]:
    base = {name: getattr(strategy, name) for name in CONFIG_NAMES}
    cases = [{"scenario": "baseline_walk_forward", "factor": "baseline",
              "mode": "walk_forward", "config": base.copy()}]
    variations = [
        ("entry_1p5", "entry threshold", "entry_z", 1.5),
        ("entry_2p5", "entry threshold", "entry_z", 2.5),
        ("exit_0p25", "exit threshold", "exit_z", 0.25),
        ("exit_1p0", "exit threshold", "exit_z", 1.0),
        ("lookback_30", "z-score lookback", "rolling_window", 30),
        ("lookback_90", "z-score lookback", "rolling_window", 90),
        ("cost_0bp", "transaction cost", "transaction_cost_rate", 0.0),
        ("cost_5bp", "transaction cost", "transaction_cost_rate", 0.0005),
        ("cost_25bp", "transaction cost", "transaction_cost_rate", 0.0025),
        ("formation_252", "formation window", "formation_window", 252),
        ("formation_504", "formation window", "formation_window", 504),
        ("trading_63", "trading block", "trading_window", 63),
        ("trading_252", "trading block", "trading_window", 252),
    ]
    for scenario, factor, name, value in variations:
        config = base.copy()
        config[name] = value
        cases.append({"scenario": scenario, "factor": factor,
                      "mode": "walk_forward", "config": config})
    cases.append({"scenario": "legacy_fixed_split", "factor": "legacy comparison",
                  "mode": "fixed_split", "config": base.copy()})
    return cases


def memoized_selector(original_selector):
    cache = {}
    counters = {"hits": 0, "misses": 0}

    def select(formation: pd.DataFrame, use_kpss: bool = True):
        # Hash formation values and index, never the later trading block.
        data_hash = hashlib.sha256(
            pd.util.hash_pandas_object(formation, index=True).to_numpy().tobytes()
        ).hexdigest()
        key = (data_hash, tuple(formation.columns), use_kpss,
               strategy.familywise_alpha, strategy.i1_screen_alpha,
               strategy.min_training_points)
        if key not in cache:
            counters["misses"] += 1
            pair, screen, scan, count = original_selector(formation, use_kpss=use_kpss)
            cache[key] = (None if pair is None else pair.copy(deep=True),
                          screen.copy(deep=True), scan.copy(deep=True), count)
        else:
            counters["hits"] += 1
        pair, screen, scan, count = cache[key]
        return (None if pair is None else pair.copy(deep=True),
                screen.copy(deep=True), scan.copy(deep=True), count)

    return select, counters


def run_suite(prices: pd.DataFrame, scenarios: list[dict], first_test_index: int):
    original_config = {name: getattr(strategy, name) for name in CONFIG_NAMES}
    original_selector = strategy.select_pair
    cached_selector, counters = memoized_selector(original_selector)
    rows, fold_results, logs = [], [], []
    try:
        strategy.select_pair = cached_selector
        for number, case in enumerate(scenarios, start=1):
            print(f"[{number}/{len(scenarios)}] {case['scenario']}", flush=True)
            for name in CONFIG_NAMES:
                setattr(strategy, name, case["config"][name])
            output = io.StringIO()
            row = {"scenario": case["scenario"], "factor": case["factor"],
                   "mode": case["mode"], **case["config"],
                   "cost_bps": case["config"]["transaction_cost_rate"] * 10_000}
            try:
                with contextlib.redirect_stdout(output):
                    daily, trades, folds, _, _ = strategy.run_evaluation(
                        prices, case["mode"],
                        first_test_index=first_test_index if case["mode"] == "walk_forward" else None,
                    )
                metrics = strategy.performance_summary(daily, trades)
                row.update({
                    "status": "completed", "error": "",
                    "evaluation_start": daily.index[0], "evaluation_end": daily.index[-1],
                    "trading_days": len(daily), "folds": len(folds),
                    "selected_folds": int((folds.status == "selected").sum()),
                    "completed_trades": len(trades),
                    "sessions_holding_position_pct": 100 * (daily.execution_position != 0).mean(),
                    "total_return_pct": metrics["Total return (%)"],
                    "annualized_return_pct": metrics["Annualized return (%)"],
                    "sharpe": metrics["Sharpe ratio"],
                    "max_drawdown_pct": metrics["Max drawdown (%)"],
                    "transaction_cost": daily.transaction_cost.sum(),
                    "net_pnl": daily.portfolio_value.iloc[-1] - strategy.initial_capital,
                    "ending_equity": daily.portfolio_value.iloc[-1],
                })
                folds = folds.copy()
                folds.insert(0, "scenario", case["scenario"])
                fold_results.append(folds)
                print(f"  return {row['total_return_pct']:.2f}%; "
                      f"trades {len(trades)}; qualifying folds {row['selected_folds']}/{len(folds)}")
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                row.update({"status": "failed", "error": str(exc)})
                print(f"  failed: {exc}")
            logs.append(f"\nSCENARIO: {case['scenario']}\n{output.getvalue()}")
            rows.append(row)
    finally:
        strategy.select_pair = original_selector
        for name, value in original_config.items():
            setattr(strategy, name, value)

    summary = pd.DataFrame(rows)
    baseline = summary[(summary.scenario == "baseline_walk_forward") & (summary.status == "completed")]
    summary["change_vs_baseline_pp"] = np.nan
    if not baseline.empty:
        completed = summary.status == "completed"
        summary.loc[completed, "change_vs_baseline_pp"] = (
            summary.loc[completed, "total_return_pct"] - float(baseline.iloc[0].total_return_pct)
        )
    folds = pd.concat(fold_results, ignore_index=True) if fold_results else pd.DataFrame()
    return summary, folds, "\n".join(logs), counters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=Path("pairs_downloaded_prices.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("robustness_results"))
    args = parser.parse_args()
    if not args.prices.is_file():
        parser.error(f"Price snapshot not found. Run python3 {Path(strategy.__file__).name} once in this folder first.")
    prices = pd.read_csv(args.prices, index_col="date", parse_dates=["date"])
    prices = prices.apply(pd.to_numeric, errors="raise")
    if prices.empty or len(prices.columns) < 2:
        parser.error("Price snapshot must contain at least two stock columns.")
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans:
        parser.error("Price snapshot must contain valid dates.")
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        parser.error("Price snapshot must have unique chronological dates.")

    # Keep the same evaluation start for all rolling-window comparisons.
    # For the current 1006-session sample this is index 603 / 2022-05-24.
    first_test_index = strategy.formation_window
    if first_test_index >= len(prices):
        parser.error("Price snapshot is too short for the configured evaluation start.")
    scenarios = build_scenarios()
    print(f"Using cached prices: {len(prices)} sessions, {len(prices.columns)} stocks.")
    print(f"Strategy file: {Path(strategy.__file__).resolve()}")
    print(f"{len(scenarios)} preset scenarios; no downloads or automatic parameter selection.")
    print(f"Walk-forward evaluation start: {prices.index[first_test_index].date()}")
    summary, folds, logs, counters = run_suite(prices, scenarios, first_test_index)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "pairs_robustness_summary.csv", index=False)
    folds.to_csv(args.output_dir / "pairs_robustness_folds.csv", index=False)
    (args.output_dir / "pairs_robustness_log.txt").write_text(logs, encoding="utf-8")
    versions = {}
    for package in ["numpy", "pandas", "statsmodels", "yfinance", "matplotlib"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "price_snapshot": args.prices.name,
        "price_snapshot_sha256": hashlib.sha256(args.prices.read_bytes()).hexdigest(),
        "strategy_file": Path(strategy.__file__).name,
        "strategy_sha256": hashlib.sha256(Path(strategy.__file__).read_bytes()).hexdigest(),
        "robustness_py_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "package_versions": versions, "selection_cache": counters,
        "walk_forward_first_test_date": prices.index[first_test_index].isoformat(),
        "scenarios": scenarios,
        "interpretation": "Retrospective sensitivity analysis. All presets are reported; "
                          "no winning configuration is selected. The legacy baseline differs "
                          "in screening, execution boundaries and signal warmup. "
                          "An all-cash result is not evidence of a profitable strategy.",
    }
    (args.output_dir / "pairs_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nRobustness Summary (preset order)")
    columns = ["scenario", "status", "total_return_pct", "completed_trades",
               "selected_folds", "sharpe", "max_drawdown_pct", "transaction_cost"]
    print(summary.reindex(columns=columns).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nAn all-cash result means no trading opportunity under the screens.")
    print("Legacy fixed-split differences are not a one-factor causal comparison.")
    print(f"Saved summary, folds, log and metadata in {args.output_dir}.")
    if (summary.status == "failed").any():
        print("Some cases failed: review the error column before interpreting the summary.")


if __name__ == "__main__":
    main()
