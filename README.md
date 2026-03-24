# Pairs Trading Signal Detector

A quantitative trading strategy that identifies statistically cointegrated stock pairs and generates buy/sell signals using z-score analysis.

## What it does
- Scans a universe of 16 liquid US stocks for cointegrated pairs
- Uses the Engle-Granger cointegration test to identify valid pairs
- Calculates the spread z-score to generate mean-reversion signals
- Backtests the strategy on 4 years of real market data (2020–2024)

## Results
- **Best pair found:** PEP / XOM (p-value: 0.0131)
- **Total return:** 85.45% over 4 years
- **Starting value:** $10,000 → **Ending value:** $18,545

## Charts
### Trading Signals
![Signal Chart](pairs_trading_signal.png)

### Cumulative Returns
![Returns Chart](cumulative_returns.png)

## Tech Stack
- Python 3
- pandas, numpy, matplotlib
- yfinance (market data)
- statsmodels (cointegration testing)

## How to run
```bash
pip install yfinance pandas numpy matplotlib statsmodels
python main.py
```

## Key concepts
- **Cointegration:** Two stocks are cointegrated if their spread is stationary — meaning it reverts to the mean over time
- **Z-score:** Measures how far the spread has deviated from its historical mean
- **Mean reversion:** When z-score > 2, we short the spread. When z-score < -2, we go long.