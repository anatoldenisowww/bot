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

**Strategy selection.** Two strategies are available via `strategy.mode` in
`settings.yaml` (or `--strategy-mode` on the CLI) - see "Strategy comparison"
below for why `ensemble` is the default and the recommendation.

- `ensemble` (default) routes on regime:
  - *Trending* (ADX >= threshold): EMA fast/slow crossover confirmed by MACD
    histogram direction, with a Donchian-channel breakout as a secondary
    trigger. Both require price to be on the correct side of a long-period EMA
    (the higher-timeframe trend filter) - the bot never fights the dominant
    trend.
  - *Ranging* (ADX < threshold): Bollinger Band extreme + RSI confirmation
    (buy oversold, sell overbought), the classic mean-reversion setup, which
    is exactly what tends to fail during strong trends - hence gating it on
    regime.
- `mean_reversion_scalp` fades short-term statistical extremes (RSI(2) deep
  oversold/overbought + a Bollinger Band touch) in any regime, skipping only
  when ADX shows a violently strong trend against the fade. It trades far
  more often on a smaller edge per trade - see below for why that didn't
  translate into either a higher win rate or better returns here.

**Risk management, per trade.**
- Stop-loss placed at `ATR * atr_stop_multiplier` from entry (volatility-
  adaptive: wider stops in choppy markets, tighter in calm ones).
- Position size computed so that a stop-out costs exactly
  `risk_per_trade_pct` of equity (1% by default) - never a fixed contract
  count.
- Take-profit at a fixed R-multiple of the initial risk (2R by default).
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
indicators.py      EMA/SMA/RSI (standard + fast)/MACD/ATR/Bollinger/ADX/Donchian (pandas, Wilder smoothing)
strategies.py       ensemble (regime-routed) + mean_reversion_scalp strategies, selected via build_strategy()
risk_manager.py     position sizing, stops/targets, trailing stops, circuit breakers
portfolio.py         equity/positions/drawdown tracking, trade log, state persistence
exchange.py          ccxt Bitget wrapper: public market data, PaperBroker, LiveBroker
backtester.py         event-driven historical replay with fees + slippage
optimize.py            walk-forward parameter search (train on the past, validate strictly out-of-sample)
bot.py                 live/paper trading loop
main.py                 CLI: backtest / optimize / run --mode paper|live
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
python main.py backtest --days 365 --strategy-mode mean_reversion_scalp
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

## Walk-forward optimization

`main.py backtest` answers "how would these parameters have performed?".
`main.py optimize` answers the harder, more honest question: "if I kept
re-optimizing this system on a schedule, using only data available at the
time, how would it actually have performed?" That distinction matters -
grid-searching a whole history and reporting whichever parameter set won is
how you produce an impressive number that's actually just overfit to noise.

```bash
python main.py optimize --days 700 --train-days 300 --test-days 100
```

For every rolling window it searches a small parameter grid (regime
threshold, stop distance, take-profit target - deliberately *not*
leverage/position size, since cranking those up always makes a backtest
look better right up until it doesn't) using Sharpe ratio with a
drawdown-and-trade-count guardrail as the objective, not raw return. The
chosen parameters are then run once, forward, on the following window they
were never allowed to see. Every fold's out-of-sample segment is chained
into one continuous curve - that chained curve is the number to trust, not
any individual fold's train-set number.

**Actual result on BTC/ETH/SOL, 4h candles, 700 days, 300d/100d walk-forward
(run 2026-07-23):**

| Metric | Value |
|---|---|
| Chained OOS period | ~400 days across 3 folds |
| Total return | +1.64% |
| Sharpe (annualized) | 0.20 |
| Max drawdown | 19.58% (see caveat below) |
| Win rate | 44.9% |
| Profit factor | 1.08 |
| Trades | 127 |

That is a modest, close-to-flat result, not an exciting one - and that's
the honest answer, not a failure of the tool. The fold-by-fold numbers show
why: fold 0 (Jul-Oct 2025) returned +17.5% out-of-sample, folds 1 and 2
gave most of it back (-6.0%, -8.1%). A system that's genuinely +1.64% net
of costs across regimes it wasn't fit to is a real, if small, edge - and a
system that claims much more than that from the same data almost always
got there by fitting noise. `settings.yaml`'s `adx_trend_threshold` and
`take_profit_r_multiple` were set to this run's most-frequently-reselected
parameters as a reasonable starting point.

**Caveat on that 19.58% max drawdown:** it's measured across the chained
curve, so fold 0's peak bleeding into fold 1's losses can produce a bigger
swing than the live bot would ever actually sit through - the live/paper
bot tracks its own equity peak continuously and halts new trades at
`max_drawdown_pct` (15% by default), which would have intervened before
this backtest's worst stretch played out in full. Treat the return/Sharpe/
win-rate numbers as the trustworthy output of this exercise, and the
drawdown number as directionally worse than reality, not better.

Re-run this periodically (monthly is reasonable) as new data comes in -
walk-forward validation is not a one-time exercise, and a parameter set
that was robust six months ago is not guaranteed to still be robust today.

## Strategy comparison: why win rate isn't the target

It's tempting to want a win rate over 50% - it feels like "the system is
usually right." Two separate experiments on this repo's real Bitget data
both say the same thing: chasing that number here made the system *less*
profitable, not more. Profitability is win_rate x avg_win vs (1-win_rate) x
avg_loss - pushing the first term up is not free if it costs more on the
other two, and in both experiments below it did.

**Experiment 1 - constrain the existing `ensemble` strategy's search to
only accept >=50% training win rate**, same walk-forward setup as above:

| | Unconstrained (default) | Win rate >=50% forced |
|---|---|---|
| Chained OOS return | **+1.64%** | -8.08% |
| Sharpe | 0.20 | -0.49 |
| Profit factor | 1.08 (profitable) | 0.93 (losing) |
| OOS win rate | 44.9% | 47.9% (missed its own target out-of-sample) |

The constraint pushed the search toward a 1R take-profit (as tight as the
stop-loss) to win more often. Win rate barely moved out-of-sample - and
didn't even clear 50% there - while every win got smaller, which was
strictly worse.

**Experiment 2 - build `mean_reversion_scalp`, a genuinely different
strategy** (RSI(2) extreme + Bollinger touch, trades in any regime, designed
to win more often on smaller moves) and walk-forward optimize it with the
same >=50%-win-rate constraint active:

| | `mean_reversion_scalp`, walk-forward |
|---|---|
| Chained OOS return | -2.83% |
| Sharpe | -0.06 |
| Profit factor | 1.03 |
| OOS win rate | 43.8% |

Every one of the 54 grid combinations tested, on every fold, failed to hit
50% win rate on its own training data - the search fell back to
`settings.yaml` defaults every time (reported in the CLI output, not
hidden). A single full-history backtest at those defaults (`main.py
backtest --strategy-mode mean_reversion_scalp --days 700`) came out to
-12.30%, worse still.

**Conclusion, stated plainly:** across two independently-designed
approaches, nothing tested clears 50% win rate on this data without
becoming unprofitable or staying unprofitable. `settings.yaml` stays on
`ensemble` with its unconstrained, walk-forward-validated parameters
(+1.64% chained OOS return, Sharpe 0.20, profit factor 1.08) because it is
the only configuration in this whole exercise that is both validated
out-of-sample and actually makes money. `mean_reversion_scalp` ships as a
selectable, tested, honestly-documented option for further experimentation
- e.g. a lower `MIN_WIN_RATE_PCT` in `optimize.py`, additional indicator
filters, or blending it with `ensemble` as a second sleeve - but it is not
currently recommended over the default.

## Uploaded reference files

This repo replaces the sketches in the uploaded `claude_dev_framework.py` /
`strategy_experiments.py` (a backtest-only toy framework, no exchange
integration, no risk management, and a couple of indicator bugs - e.g. its
MACD signal line was computed by smoothing a single-element list) with a
complete, runnable system: real Bitget connectivity, proper indicator math,
and the risk controls a live bot actually needs.
