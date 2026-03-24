import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
import itertools

# ── 1. Define candidate stocks ──────────────────────────────────────
# A diverse universe of liquid US stocks across sectors
tickers = ["XOM", "CVX", "KO", "PEP", "JPM", "BAC", "GS", "MS",
           "GOOG", "MSFT", "AAPL", "META", "WMT", "TGT", "MCD", "YUM"]

start = "2020-01-01"
end = "2024-01-01"

print("Downloading data...")
data = yf.download(tickers, start=start, end=end)["Close"]
data.dropna(axis=1, inplace=True)
print(f"Downloaded {data.shape[1]} stocks, {data.shape[0]} days\n")

# ── 2. Scan all pairs for cointegration ─────────────────────────────
print("Scanning pairs for cointegration...")
pairs = []

for s1, s2 in itertools.combinations(data.columns, 2):
    score, pvalue, _ = coint(data[s1], data[s2])
    if pvalue < 0.05:
        pairs.append((s1, s2, round(pvalue, 4)))

# ── 3. Show results ──────────────────────────────────────────────────
if pairs:
    pairs_df = pd.DataFrame(pairs, columns=["Stock 1", "Stock 2", "P-Value"])
    pairs_df.sort_values("P-Value", inplace=True)
    print(f"✅ Found {len(pairs)} cointegrated pairs:\n")
    print(pairs_df.to_string(index=False))
else:
    print("❌ No cointegrated pairs found in this universe")

    # ── 4. Build the trading signal for best pair ───────────────────────
s1, s2 = "PEP", "XOM"

# Calculate the spread (difference between the two normalized prices)
spread = data[s1] - data[s2]

# Z-score tells us how far the spread is from its mean
# When z-score is extreme, we expect it to revert back to zero
zscore = (spread - spread.mean()) / spread.std()

# ── 5. Generate buy/sell signals ────────────────────────────────────
# When z-score > 2: spread is too wide → sell PEP, buy XOM
# When z-score < -2: spread is too narrow → buy PEP, sell XOM
# When z-score crosses 0: close the position

signals = pd.DataFrame(index=data.index)
signals["spread"] = spread
signals["zscore"] = zscore
signals["long_entry"]  = zscore < -2      # Buy signal
signals["short_entry"] = zscore > 2       # Sell signal
signals["exit"]        = abs(zscore) < 0.5  # Exit signal

print(f"\n📊 Signal Summary:")
print(f"Long entries (buy):  {signals['long_entry'].sum()}")
print(f"Short entries (sell): {signals['short_entry'].sum()}")
print(f"Exit signals:        {signals['exit'].sum()}")

# ── 6. Plot the results ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle("Pairs Trading Signal: PEP vs XOM", fontsize=14, fontweight="bold")

# ── Top chart: Stock prices ──────────────────────────────────────────
ax1.plot(data["PEP"], label="PEP", color="royalblue")
ax1.plot(data["XOM"], label="XOM", color="darkorange")
ax1.set_ylabel("Price (USD)")
ax1.legend()
ax1.set_title("Stock Prices")
ax1.grid(alpha=0.3)

# ── Bottom chart: Z-score + signals ─────────────────────────────────
ax2.plot(zscore, label="Z-Score", color="purple", linewidth=1)
ax2.axhline(2,    color="red",   linestyle="--", linewidth=1, label="Sell threshold (+2)")
ax2.axhline(-2,   color="green", linestyle="--", linewidth=1, label="Buy threshold (-2)")
ax2.axhline(0.5,  color="gray",  linestyle=":",  linewidth=1)
ax2.axhline(-0.5, color="gray",  linestyle=":",  linewidth=1, label="Exit zone (±0.5)")

# Plot buy signals
ax2.scatter(zscore.index[signals["long_entry"]],
            zscore[signals["long_entry"]],
            color="green", marker="^", s=80, zorder=5, label="Buy")

# Plot sell signals
ax2.scatter(zscore.index[signals["short_entry"]],
            zscore[signals["short_entry"]],
            color="red", marker="v", s=80, zorder=5, label="Sell")

ax2.set_ylabel("Z-Score")
ax2.set_title("Trading Signals (Z-Score of Spread)")
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("pairs_trading_signal.png", dpi=150)
plt.show()
print("\n✅ Chart saved as pairs_trading_signal.png")

# ── 7. Simple Backtest ──────────────────────────────────────────────
# Position: +1 = long PEP/short XOM, -1 = short PEP/long XOM, 0 = flat
position = 0
portfolio = [0]
returns = data["PEP"].pct_change().fillna(0) - data["XOM"].pct_change().fillna(0)

positions = []
for i in range(len(signals)):
    if signals["long_entry"].iloc[i]:
        position = 1
    elif signals["short_entry"].iloc[i]:
        position = -1
    elif signals["exit"].iloc[i]:
        position = 0
    positions.append(position)

signals["position"] = positions
signals["strategy_returns"] = signals["position"].shift(1) * returns
signals["cumulative_returns"] = (1 + signals["strategy_returns"]).cumprod()

# ── 8. Plot cumulative returns ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(signals["cumulative_returns"], color="green", linewidth=1.5)
ax.axhline(1, color="gray", linestyle="--", linewidth=1)
ax.set_title("Pairs Trading Strategy — Cumulative Returns (PEP vs XOM)", fontweight="bold")
ax.set_ylabel("Portfolio Growth (1 = starting value)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("cumulative_returns.png", dpi=150)
plt.show()

# ── 9. Print performance summary ────────────────────────────────────
total_return = signals["cumulative_returns"].iloc[-1] - 1
print(f"\n📈 Strategy Performance Summary:")
print(f"Total Return:        {total_return*100:.2f}%")
print(f"Starting value:      $10,000")
print(f"Ending value:        ${10000 * (1 + total_return):,.2f}")