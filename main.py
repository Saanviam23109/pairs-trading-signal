import warnings
warnings.filterwarnings("ignore")

import itertools
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm


# ============================================================
# 1. CONFIG
# ============================================================
tickers = [
    "XOM", "CVX", "KO", "PEP", "JPM", "BAC", "GS", "MS",
    "GOOG", "MSFT", "AAPL", "META", "WMT", "TGT", "MCD", "YUM"
]

start_date = "2020-01-01"
end_date = "2024-01-01"

initial_capital = 10_000
entry_z = 2.0
exit_z = 0.5
rolling_window = 60
transaction_cost_rate = 0.001   # 0.1% per position change
min_training_points = 252       # ~1 trading year for formation


# ============================================================
# 2. DOWNLOAD DATA
# ============================================================
print("Downloading price data...")
prices = yf.download(tickers, start=start_date, end=end_date)["Close"]
prices = prices.dropna(axis=1)
prices = prices.dropna()

print(f"Downloaded {prices.shape[1]} stocks, {prices.shape[0]} trading days.\n")


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================
split_idx = int(len(prices) * 0.6)  # 60% formation, 40% trading
train_prices = prices.iloc[:split_idx].copy()
test_prices = prices.iloc[split_idx:].copy()

if len(train_prices) < min_training_points:
    raise ValueError("Training sample is too short for robust pair selection.")

print(f"Formation period: {train_prices.index[0].date()} to {train_prices.index[-1].date()}")
print(f"Trading period:   {test_prices.index[0].date()} to {test_prices.index[-1].date()}\n")


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================
def estimate_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    """
    Estimate hedge ratio beta from OLS:
        y = alpha + beta * x + epsilon
    """
    x_const = sm.add_constant(x)
    model = sm.OLS(y, x_const).fit()
    return model.params[x.name]


def compute_spread(y: pd.Series, x: pd.Series, beta: float) -> pd.Series:
    """
    Construct hedge-ratio adjusted spread.
    """
    return y - beta * x


def compute_rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
    """
    Rolling z-score using only past information.
    """
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    z = (spread - rolling_mean) / rolling_std
    return z


def annualized_sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    """
    Annualized Sharpe ratio from daily returns.
    """
    returns = returns.dropna()
    if returns.std() == 0 or len(returns) < 2:
        return np.nan
    excess = returns - rf / 252
    return np.sqrt(252) * excess.mean() / excess.std()


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum drawdown from equity curve.
    """
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown.min()


def calculate_half_life(spread: pd.Series) -> float:
    """
    Estimate half-life of mean reversion using an AR(1)-style regression.
    """
    spread_lag = spread.shift(1).dropna()
    spread_ret = spread.diff().dropna()

    aligned = pd.concat([spread_lag, spread_ret], axis=1).dropna()
    if aligned.empty:
        return np.nan

    x = sm.add_constant(aligned.iloc[:, 0])
    y = aligned.iloc[:, 1]
    model = sm.OLS(y, x).fit()

    beta = model.params.iloc[1]
    if beta >= 0:
        return np.nan

    half_life = -np.log(2) / beta
    return float(half_life)


# ============================================================
# 5. PAIR SELECTION ON FORMATION PERIOD
# ============================================================
print("Scanning pairs in formation period for cointegration...\n")

pair_results = []

for s1, s2 in itertools.combinations(train_prices.columns, 2):
    y_train = train_prices[s1]
    x_train = train_prices[s2]

    try:
        coint_stat, pvalue, _ = coint(y_train, x_train)
        beta = estimate_hedge_ratio(y_train, x_train)
        spread_train = compute_spread(y_train, x_train, beta)

        adf_pvalue = adfuller(spread_train.dropna())[1]
        hl = calculate_half_life(spread_train.dropna())

        pair_results.append({
            "stock_1": s1,
            "stock_2": s2,
            "coint_pvalue": pvalue,
            "adf_pvalue": adf_pvalue,
            "hedge_ratio": beta,
            "half_life": hl
        })
    except Exception:
        continue

pairs_df = pd.DataFrame(pair_results)

# Filter for stronger candidates
filtered_pairs = pairs_df[
    (pairs_df["coint_pvalue"] < 0.05) &
    (pairs_df["adf_pvalue"] < 0.05) &
    (pairs_df["half_life"].notna()) &
    (pairs_df["half_life"] > 1) &
    (pairs_df["half_life"] < 60)
].copy()

if filtered_pairs.empty:
    raise ValueError("No suitable cointegrated pairs found.")

filtered_pairs = filtered_pairs.sort_values(["coint_pvalue", "adf_pvalue", "half_life"])
best_pair = filtered_pairs.iloc[0]

s1 = best_pair["stock_1"]
s2 = best_pair["stock_2"]
beta = best_pair["hedge_ratio"]

print("Top candidate pairs:")
print(filtered_pairs.head(10).to_string(index=False))
print("\nSelected best pair:")
print(f"{s1} / {s2}")
print(f"Cointegration p-value: {best_pair['coint_pvalue']:.4f}")
print(f"ADF p-value:           {best_pair['adf_pvalue']:.4f}")
print(f"Hedge ratio (beta):    {beta:.4f}")
print(f"Half-life:             {best_pair['half_life']:.2f} days\n")


# ============================================================
# 6. BUILD TEST-PERIOD SPREAD AND SIGNALS
# ============================================================
y_test = test_prices[s1]
x_test = test_prices[s2]

spread_test = compute_spread(y_test, x_test, beta)
zscore_test = compute_rolling_zscore(spread_test, rolling_window)

signals = pd.DataFrame(index=test_prices.index)
signals["y_price"] = y_test
signals["x_price"] = x_test
signals["spread"] = spread_test
signals["zscore"] = zscore_test

# Position convention:
# +1 = long spread  = long y, short beta*x
# -1 = short spread = short y, long beta*x
position = 0
positions = []

for z in signals["zscore"]:
    if pd.isna(z):
        positions.append(0)
        continue

    if position == 0:
        if z < -entry_z:
            position = 1
        elif z > entry_z:
            position = -1
    else:
        if abs(z) < exit_z:
            position = 0

    positions.append(position)

signals["position"] = positions


# ============================================================
# 7. STRATEGY RETURNS
# ============================================================
# Daily returns of each leg
signals["y_ret"] = signals["y_price"].pct_change().fillna(0)
signals["x_ret"] = signals["x_price"].pct_change().fillna(0)

# Hedge-ratio-adjusted spread return approximation
# long spread  = +y - beta*x
# short spread = -y + beta*x
signals["gross_strategy_return"] = (
    signals["position"].shift(1).fillna(0) *
    (signals["y_ret"] - beta * signals["x_ret"])
)

# Transaction costs when position changes
signals["position_change"] = signals["position"].diff().abs().fillna(0)
signals["transaction_cost"] = signals["position_change"] * transaction_cost_rate

signals["net_strategy_return"] = (
    signals["gross_strategy_return"] - signals["transaction_cost"]
)

signals["equity_curve"] = (1 + signals["net_strategy_return"]).cumprod()
signals["portfolio_value"] = initial_capital * signals["equity_curve"]


# ============================================================
# 8. PERFORMANCE METRICS
# ============================================================
total_return = signals["equity_curve"].iloc[-1] - 1
sharpe = annualized_sharpe(signals["net_strategy_return"])
mdd = max_drawdown(signals["equity_curve"])

n_days = len(signals)
annual_return = (signals["equity_curve"].iloc[-1]) ** (252 / n_days) - 1 if n_days > 0 else np.nan
calmar = annual_return / abs(mdd) if pd.notna(mdd) and mdd != 0 else np.nan

num_trades = int((signals["position_change"] > 0).sum())

print("Strategy Performance Summary")
print("-" * 40)
print(f"Selected pair:         {s1} / {s2}")
print(f"Formation hedge ratio: {beta:.4f}")
print(f"Trading days:          {n_days}")
print(f"Number of trades:      {num_trades}")
print(f"Total return:          {total_return * 100:.2f}%")
print(f"Annualized return:     {annual_return * 100:.2f}%")
print(f"Sharpe ratio:          {sharpe:.2f}")
print(f"Max drawdown:          {mdd * 100:.2f}%")
print(f"Calmar ratio:          {calmar:.2f}")
print(f"Starting value:        ${initial_capital:,.2f}")
print(f"Ending value:          ${signals['portfolio_value'].iloc[-1]:,.2f}")


# ============================================================
# 9. PLOTS
# ============================================================
plt.figure(figsize=(14, 5))
plt.plot(signals.index, signals["equity_curve"], linewidth=1.5)
plt.axhline(1.0, linestyle="--", linewidth=1)
plt.title(f"Pairs Trading Equity Curve ({s1} vs {s2})")
plt.ylabel("Equity Curve")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("improved_pairs_equity_curve.png", dpi=150)
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(signals.index, signals["zscore"], linewidth=1.2, label="Rolling Z-Score")
plt.axhline(entry_z, linestyle="--", linewidth=1, label=f"Short Entry (+{entry_z})")
plt.axhline(-entry_z, linestyle="--", linewidth=1, label=f"Long Entry (-{entry_z})")
plt.axhline(exit_z, linestyle=":", linewidth=1, label=f"Exit (+{exit_z})")
plt.axhline(-exit_z, linestyle=":", linewidth=1, label=f"Exit (-{exit_z})")

buy_idx = signals.index[(signals["position"].diff() == 1) | (signals["position"].diff() == 2)]
sell_idx = signals.index[(signals["position"].diff() == -1) | (signals["position"].diff() == -2)]

plt.scatter(buy_idx, signals.loc[buy_idx, "zscore"], marker="^", s=70, zorder=5, label="Long Spread")
plt.scatter(sell_idx, signals.loc[sell_idx, "zscore"], marker="v", s=70, zorder=5, label="Short Spread")

plt.title(f"Rolling Z-Score Signals ({s1} vs {s2})")
plt.ylabel("Z-Score")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("improved_pairs_zscore_signals.png", dpi=150)
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(signals.index, signals["portfolio_value"], linewidth=1.5)
plt.title(f"Portfolio Value Over Time ({s1} vs {s2})")
plt.ylabel("Portfolio Value ($)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("improved_pairs_portfolio_value.png", dpi=150)
plt.show()
