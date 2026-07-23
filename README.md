# Bitget Quant Trading Bot - BTC / ETH / SOL

A regime-aware quant trading bot for Bitget USDT-M perpetual futures, built
around one idea a professional desk would insist on: **the risk manager
matters more than the entry signal.** Position sizing, stop-losses and
circuit breakers are not optional add-ons here - they're the core of the
system, and every trade goes through them before it's allowed to happen.

## Read this first

No trading strategy - here or anywhere - is "guaranteed profitable." Anyone
who tells you otherwise is selling something. Crypto futures are volatile
and leveraged; you can lose your entire deposit, faster than in spot
markets. What this bot *does* give you:

- A real, working strategy with a documented edge hypothesis (trend +
  mean-reversion regime switching), not a black box.
- Disciplined risk management: every position is sized so a stop-out costs a
  fixed, small percentage of equity, with portfolio-wide risk caps and
  daily-loss / max-drawdown circuit breakers that halt trading automatically.
- A backtester so you can see how the strategy would have performed on real
  historical Bitget data before risking a cent.
- A paper-trading mode (the default) that runs against live market data with
  simulated fills, so you can watch it operate with zero financial risk.
- A live-trading mode that is off by default and requires two explicit,
  separate confirmations to enable.

Start in `backtest`, move to `paper`, and only consider `live` after you've
watched it operate for a meaningful stretch of time and you understand
exactly what it will do with your money. Start with an amount you can afford
to lose completely.

## How it works

**Regime detection.** ADX on the execution timeframe (4h by default)
classifies the market as trending or ranging.

**Strategy selection.**
- *Trending* (ADX >= threshold): EMA fast/slow crossover confirmed by MACD
  histogram direction, with a Donchian-channel breakout as a secondary
  trigger. Both require price to be on the correct side of a long-period EMA
  (the higher-timeframe trend filter) - the bot never fights the dominant
  trend.
- *Ranging* (ADX < threshold): Bollinger Band extreme + RSI confirmation
  (buy oversold, sell overbought), the classic mean-reversion setup, which
  is exactly what tends to fail during strong trends - hence gating it on
  regime.

**Risk management, per trade.**
- Stop-loss placed at `ATR * atr_stop_multiplier` from entry (volatility-
  adaptive: wider stops in choppy markets, tighter in calm ones).
- Position size computed so that a stop-out costs exactly
  `risk_per_trade_pct` of equity (1% by default) - never a fixed contract
  count.
- Take-profit at a fixed R-multiple of the initial risk (2.5R by default).
- A trailing stop kicks in once a trade is up `trailing_activation_r` (1R by
  default), locking in gains as price moves further favorably.

**Risk management, portfolio-wide.**
- Max concurrent positions (default 3 - one per symbol).
- Max total open risk as a % of equity across all positions at once, so BTC/
  ETH/SOL's high correlation can't stack into an outsized bet.
- Daily loss circuit breaker: past `max_daily_loss_pct` (3% default), no new
  trades until the next UTC day.
- Max drawdown circuit breaker: past `max_drawdown_pct` (15% default) from
  the equity peak, the bot stops opening new trades entirely and needs a
  human to review before continuing.

All of these are config knobs in `settings.yaml`, not constants buried in
code - tune them to your own risk tolerance.

## Project layout

```
config.py        settings.yaml + .env loader
models.py         Position / ClosedTrade dataclasses
indicators.py      EMA/SMA/RSI/MACD/ATR/Bollinger/ADX/Donchian (pandas, Wilder smoothing)
strategies.py       regime detection + trend/mean-reversion/breakout signal logic
risk_manager.py     position sizing, stops/targets, trailing stops, circuit breakers
portfolio.py         equity/positions/drawdown tracking, trade log, state persistence
exchange.py          ccxt Bitget wrapper: public market data, PaperBroker, LiveBroker
backtester.py         event-driven historical replay with fees + slippage
bot.py                 live/paper trading loop
main.py                 CLI: backtest / run --mode paper|live
settings.yaml            strategy & risk parameters (safe to edit freely)
.env.example              API credential template (copy to .env, never commit .env)
tests/                     pytest suite for indicators, risk manager, backtester
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# only needed for live trading - paper mode and backtesting need no keys
```

## Usage

**Backtest** against real historical Bitget data (no API keys required):

```bash
python main.py backtest --days 365
python main.py backtest --days 365 --save-trades trades.jsonl
```

**Paper trade** (default, no API keys required) - runs continuously against
live market data with simulated fills:

```bash
python main.py run --mode paper
```

**Live trade** - requires `LIVE_TRADING_ENABLED=true` in `.env` *and* the
`--i-understand-the-risk` flag. Both are required on purpose:

```bash
python main.py run --mode live --i-understand-the-risk
```

Create your Bitget API key at Bitget -> API Management, scoped to **Futures
Trade only** - never enable withdrawal permissions on a bot's API key.

State (open positions, equity) is persisted to `state/<mode>/state.json` so
the bot resumes correctly after a restart. Closed trades are appended to
`state/<mode>/trades.jsonl` for your own record-keeping and later analysis.

## Testing

```bash
pytest tests/ -v
```

## Tuning

Everything strategy- and risk-related lives in `settings.yaml`: symbols,
timeframe, indicator periods/thresholds, risk-per-trade, leverage, drawdown
limits. Back-test after every change - a parameter set that looks great on
one window can be badly overfit; validate on multiple time periods before
trusting it, and re-validate periodically as market conditions change.

## Uploaded reference files

This repo replaces the sketches in the uploaded `claude_dev_framework.py` /
`strategy_experiments.py` (a backtest-only toy framework, no exchange
integration, no risk management, and a couple of indicator bugs - e.g. its
MACD signal line was computed by smoothing a single-element list) with a
complete, runnable system: real Bitget connectivity, proper indicator math,
and the risk controls a live bot actually needs.
