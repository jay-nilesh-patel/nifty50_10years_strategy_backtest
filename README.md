# NIFTY50 EMA89 Strategy Backtest (Streamlit)

A Streamlit dashboard that backtests a personally used NIFTY50 intraday strategy based on an EMA band (EMA on High/Low) with pierce + confirmation entries, and reports synthetic futures P&L in ₹ after costs.

## Important note (EMA 89)
This is a **personal research** project and the strategy logic was designed and evaluated around **EMA length = 89**.  
The app allows changing the EMA length for experimentation, but the reported performance and conclusions in this repo are intended for the **EMA89 configuration**; other values should be treated as separate experiments (i.e., not directly comparable to the EMA89 backtest).

## Features
- One-click data loading from `./data` (no uploader UI)
- Timeframe selector: 5m / 15m / 30m
- Strategy controls: EMA length, Risk:Reward (1/2/3)
- Synthetic futures P&L in ₹ with sizing, slippage, and fees
- Charts/tables:
  - Price + EMA band + entry/exit markers
  - Equity curve + drawdown
  - Yearly summary
  - TF × RR grid (Net PnL and Accuracy)

## Strategy summary (high level)
- EMA band: `EMA(high, N)` and `EMA(low, N)`
- Entry signals require a pierce + confirmation sequence (see code in `generate_signals`)
- Target: `entry ± (RR × risk)`, where risk is measured vs the opposite EMA band at the signal bar
- Stop: 2 consecutive closes beyond the opposite EMA band (exit at same bar close)
- Forced time-exit: at the configured bar close per timeframe

## Costs model
The dashboard models:

- Slippage: points per side (adverse on entry and exit)
- Fees: ₹ per round-trip (total)
- PnL is computed as synthetic futures P&L using:
- Lot size × number of contracts
