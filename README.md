# Bitget Quant Trading Bot - multi-asset crypto perpetuals

A quant trading bot for Bitget USDT-M perpetual futures, built around one
idea a professional desk would insist on: **the risk manager matters more
than the entry signal.** Position sizing, stop-losses and circuit breakers
are the core of the system - every trade goes through them before it's
allowed to happen - and it's sized to run on a small account (a few hundred
euros) that you top up over time.

## Read this first - the honest headline

**No strategy here is a money printer, and the backtests say so plainly.**
Three separate, seriously-built strategies were each walk-forward validated
on real Bitget data (the honest way: parameters chosen only on past data,
scored only on unseen future data). Every one of them lands between
break-even and slightly positive **net of realistic fees and slippage**.
That is the truthful result of this exercise, and it's far more valuable
than a fake-impressive number that would evaporate live.

What that means concretely: the current default (`cross_sectional_momentum`)
showed a **real gross edge** (profit factor 1.07 - winners exceed losers)
with the best risk profile of anything built here (10.4% max drawdown across
10 diversified symbols), but trading costs ate the edge down to roughly
break-even (-0.34% over ~400 out-of-sample days). See "Strategy comparison
and honest results" below for the full numbers on all three.

**So why run it at all?** Because what you have is a professionally
structured, properly risk-managed, diversified system with disciplined
execution and a small real edge - the correct *foundation* to iterate on -
rather than a curve-fit fantasy. The honest path to net-profitable from here
is reducing trading costs (maker-only orders, a slower timeframe, fewer
trades), not cranking leverage. Treat live trading as an experiment with
money you can afford to lose entirely, starting small.

**Workflow: `backtest` -> `optimize` -> `paper` (for weeks) -> `live` small.**
Never skip paper trading.

## How it works

**Strategy selection.** Choose via `strategy.mode` in `settings.yaml` (or
`--strategy-mode` on the CLI):

- `cross_sectional_momentum` (**default**) - the rebuild for a broad
  universe. Each bar it ranks the most liquid ~15 Bitget USDT-M perpetuals
  (dynamically selected, see `universe.py`) by **risk-adjusted momentum**:
  `(price_now - price_N_bars_ago) / (ATR x sqrt(N))`. It goes long the
  `top_k` strongest and short the `bottom_k` weakest, and exits a name as
  soon as it drops out of that bucket. Normalizing by ATR is what makes a
  volatile micro-cap and a calm major comparable - ranking on raw return
  alone would just always pick the most volatile coin. This is the
  diversified, cross-sectional approach real systematic desks actually use,
  rather than one signal on one asset.
- `ensemble` - regime-routed single-symbol strategy on a fixed list:
  - *Trending* (ADX >= threshold): EMA crossover + MACD confirmation, plus a
    Donchian breakout, always on the correct side of a long-period EMA.
  - *Ranging* (ADX < threshold): Bollinger-band + RSI mean reversion.
- `mean_reversion_scalp` - fades short-term RSI(2)/Bollinger extremes in any
  regime, skipping only violently strong trends. Trades often on a thin edge.

**Dynamic universe (cross_sectional_momentum only).** Instead of a fixed
BTC/ETH/SOL list, the tradable set is the top-N most liquid eligible
USDT-M perpetuals by 24h quote volume, re-scanned every
`universe_refresh_hours` in live/paper mode. Tokenized real-world-asset
perpetuals (gold, tokenized stocks - Bitget lists these alongside crypto
and tags them `isRwa`) are filtered out; they have gap/corporate-action
risk this strategy isn't built for.

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
- Max concurrent positions (default 6 - matches top_k + bottom_k).
- Max total open risk as a % of equity across all positions at once.
- Daily loss circuit breaker: past `max_daily_loss_pct` (3% default), no new
  trades until the next UTC day.
- Max drawdown circuit breaker: past `max_drawdown_pct` (15% default) from
  the equity peak, the bot stops opening new trades entirely and needs a
  human to review before continuing.

**Small-account handling.** Every order's quantity is rounded to the
exchange's step size and skipped if it falls below Bitget's minimum order
size (~$5 notional) - a real concern at a few hundred euros that doesn't
exist at $10k. In live mode the bot reads your **actual exchange balance**
each cycle, so a monthly top-up is picked up automatically as more buying
power without editing any file. Leverage is deliberately capped low
(`max_leverage: 3`) and **not** raised for the small account - leverage is
how small accounts die, not how they grow.

All of these are config knobs in `settings.yaml`, not constants buried in
code - tune them to your own risk tolerance.

## Project layout

```
config.py        settings.yaml + .env loader
models.py         Position / ClosedTrade dataclasses
indicators.py      EMA/SMA/RSI (standard + fast)/MACD/ATR/Bollinger/ADX/Donchian (pandas, Wilder smoothing)
universe.py         liquidity-based dynamic asset selection (filter + rank Bitget perps, drop RWA tokens)
strategies.py       cross_sectional_momentum + ensemble + mean_reversion_scalp, selected via build_strategy()
risk_manager.py     position sizing, stops/targets, trailing stops, circuit breakers
portfolio.py         equity/positions/drawdown tracking, trade log, state persistence
exchange.py          ccxt Bitget wrapper: market data, universe fetch, order-size limits, PaperBroker, LiveBroker
backtester.py         event-driven historical replay with fees + slippage + exchange min-order-size handling
optimize.py            walk-forward parameter search (train on the past, validate strictly out-of-sample)
regime_hmm.py          Hidden Markov Model market-regime detector (moods + BUY/HOLD/SELL leans; analysis tool)
regime_report.py       colorful, theme-aware multi-symbol HTML dashboard for the HMM regime read
statarb.py             pairs-trading statistical-arbitrage screener (hedge ratio, half-life, z-score, backtest)
statarb_report.py      colorful, theme-aware HTML dashboard for the pairs screener
bot.py                 live/paper trading loop with dynamic universe refresh + live balance sync
main.py                 CLI: backtest / optimize / regime / pairs / run --mode paper|live
settings.yaml            strategy & risk parameters (safe to edit freely)
.env.example              API credential template (copy to .env, never commit .env)
tests/                     pytest suite (indicators, risk, portfolio, strategies, universe, sizing, backtester, optimizer)
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
python main.py backtest --days 365                                    # default strategy (cross_sectional_momentum)
python main.py backtest --days 700 --strategy-mode ensemble           # try another strategy
python main.py backtest --days 365 --save-trades trades.jsonl         # dump every trade to inspect
```

**Market-regime detector (Hidden Markov Model)** - an information tool, not a
trade signal. Fits a Gaussian HMM to log-returns + rolling volatility and
tells you which hidden regime (e.g. calm bull, choppy range, volatile
selloff) the market is most likely in now, how those regimes behave, and how
they transition:

```bash
python main.py regime --symbol BTC/USDT:USDT --timeframe 4h --states 3
python main.py regime --symbol ETH/USDT:USDT --html regime_eth.html   # + a visual dashboard
```

It prints a table of regimes (mean return, volatility, how often, how long
they last), the current regime with confidence, and the transition-
probability matrix. `--html` also writes a self-contained, theme-aware
dashboard you can open in any browser. **What it is not:** a price predictor
or a profit guarantee - an HMM describes the current statistical state and
historical dynamics; it cannot tell you tomorrow's return. Useful as context
(e.g. "high-volatility regime -> size down"), not as a buy/sell trigger.

**Pairs-trading screener (statistical arbitrage)** - screens which crypto
pairs are actually worth pairs-trading. For each pair it fits a hedge ratio,
measures how fast the spread mean-reverts (half-life), shows where the spread
sits now (z-score) and the current signal, and backtests fading the extremes
net of costs on all four legs:

```bash
python main.py pairs --symbols BTC/USDT:USDT ETH/USDT:USDT SOL/USDT:USDT
python main.py pairs --days 700 --html pairs.html    # + a colorful dashboard
```

Honest result on BTC/ETH/SOL: they're highly correlated but their spreads
have a **half-life of ~500 bars (~3 months)** - they trend together and don't
mean-revert fast enough to trade, and the backtests are negative. The tool
correctly flags them "not a good pair right now" (verified against a
synthetic mean-reverting pair, where it finds a short half-life and a
positive edge). Point it at other symbols/timeframes to screen for pairs that
actually revert. **Stat arb is not pure arbitrage:** no guaranteed
convergence, individual trades lose, relationships break - hence the hard
z-score stop.

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

For every rolling window it searches a small, **strategy-specific** parameter
grid (for momentum: lookback, signal-strength floor, stop distance -
deliberately *not* leverage/position size, since cranking those up always
makes a backtest look better right up until it doesn't) using Sharpe ratio
with a drawdown-and-trade-count guardrail as the objective, not raw return.
The chosen parameters are run once, forward, on the following window they
were never allowed to see. Every fold's out-of-sample segment is chained
into one continuous curve - that chained curve is the number to trust.

```bash
python main.py optimize --days 700 --train-days 300 --test-days 100
python main.py optimize --days 700 --strategy-mode ensemble
```

## Strategy comparison and honest results

Every number below is **chained out-of-sample** (walk-forward, ~400 days,
300d/100d folds) on real Bitget data, net of 0.06% taker fees + 0.05%
assumed slippage per fill. Run date 2026-07-23.

| Strategy | Universe / TF | OOS return | Sharpe | Profit factor | Max DD | Win rate | Trades |
|---|---|---|---|---|---|---|---|
| **cross_sectional_momentum** (default) | 10 liquid perps, 4h | **-0.34%** | 0.01 | **1.07** | **10.4%** | 37% | 265 |
| cross_sectional_momentum | 9 perps, **1d** | -1.35% | -0.49 | 0.73 | **4.3%** | 45% | **31** |
| ensemble | BTC/ETH/SOL, 4h | +1.64% | 0.20 | 1.08 | 19.6% | 45% | 127 |
| mean_reversion_scalp | BTC/ETH/SOL, 4h | -2.83% | -0.06 | 1.03 | ~20% | 44% | ~130 |

The 1d row was the theory-driven attempt to find real profit (momentum is
more robust at longer horizons, and 31 trades instead of 265 slashes cost
drag). It worked on cost and risk - drawdown fell to 4.3% - but still didn't
turn net-positive on this sample. That's the honest result: **no directional
configuration tested here shows a reliable net-of-cost edge.** The one that's
marginally positive (ensemble, +1.64%) is on 3 correlated coins and is well
within noise. This matches reality - persistent directional alpha, net of
costs, is genuinely hard for a retail account, and the intellectually honest
next step is a market-neutral **funding-rate carry** strategy (harvesting the
perpetual funding cash flow rather than predicting direction), which is a
different kind of edge entirely - not yet built here.

**How to read this honestly:**

- All three are essentially **break-even net of costs**. None is a money
  printer. That is the true finding, and it's worth more than a fabricated
  positive number that would fail live.
- `cross_sectional_momentum` (the broad-universe rebuild) has a **real gross
  edge** - profit factor 1.07 means its winning trades outweigh its losers -
  but 265 trades x ~0.11% round-trip cost is ~a third of the account in
  cumulative friction, which is what drags it to break-even. Its **10.4% max
  drawdown across 10 diversified names is the best risk profile here**, and
  its trending-regime fold posted +8.6% at Sharpe 2.47 before choppy folds
  gave it back - momentum works in trends and bleeds in chop, as expected.
- `ensemble` is the only one that edged out net-positive (+1.64%), but on
  just three highly-correlated coins with a deeper 19.6% drawdown - less
  diversified and more concentrated than the default.
- The path from "break-even gross edge" to "net profitable" is **lower
  trading costs**, not more leverage: maker-limit orders instead of market
  (turns the 0.06% fee into a rebate), a slower timeframe (fewer trades), or
  a higher `min_abs_momentum_score` (only the strongest signals). Those are
  the honest next experiments.

**On win rate (you asked earlier):** momentum structurally wins <50% of the
time - it makes money from a few large winners, not from being right often.
Forcing a >=50% win rate makes these strategies *less* profitable; an
earlier experiment on `ensemble` drove its OOS return from +1.64% to -8.08%
by doing exactly that. The `--min-win-rate` flag on `optimize` still lets
you explore that tradeoff, but it defaults to off for this reason.

**On timeframe (a 1h backtest lesson):** the default parameters are tuned for
the 4h timeframe. A 700-day backtest on **1h** candles at those same params
returned -15.23% (profit factor 0.41) - much worse, and not because the code
is broken but because `momentum_lookback_bars: 20` means ~3.3 days on 4h but
only 20 hours on 1h, and ultra-short-horizon "momentum" in crypto tends to
mean-revert rather than continue. The edge inverts. **Parameters do not
transfer across timeframes** - if you want to run 1h, re-run `optimize
--timeframe 1h` first (it would pick a much longer lookback). This is a
general rule, not a quirk of this bot.

**The multiple-comparisons caveat:** ~100+ parameter combinations were tried
across these strategies. Walk-forward guards against overfitting a single
window, but trying many things and reporting the best still bakes in some
optimism even out-of-sample. Treat these numbers as "roughly break-even with
a small real edge," not as precise forecasts.

Re-run `optimize` periodically (monthly is reasonable) as new data arrives -
a parameter set robust six months ago is not guaranteed robust today.

## First time using Bitget? Step-by-step guide

This walks you from zero to a bot trading a small live balance. **Do not
skip the paper-trading step** - it's how you find out what the bot does with
money before it's real money.

### 1. Create and fund a Bitget account
1. Sign up at bitget.com and complete identity verification (required for
   futures trading).
2. Deposit USDT (Bitget's quote currency). Your ~200 EUR becomes ~215 USDT.
   Deposit via card or by transferring crypto and converting to USDT.
3. Move the USDT into your **USDT-M Futures** wallet (Bitget keeps spot and
   futures balances separate - the bot trades futures).

### 2. Create an API key (the bot's login)
1. Bitget web -> profile icon -> **API Management** -> **Create API Key** ->
   choose **System-generated**.
2. Permissions: enable **read** and **Futures/Contract Trade** only. **Leave
   withdrawals DISABLED.** A trading bot never needs to withdraw; a key that
   can't withdraw can't drain your account if it leaks.
3. Set a passphrase when prompted (you choose it) - you'll need it below.
4. Optional but recommended: bind the key to your server's IP address.
5. Copy the three values it shows once: **API Key**, **Secret Key**,
   **Passphrase**. The secret is shown only once.

### 3. Install and configure the bot
```bash
git clone <your repo URL>
cd bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```
Open `.env` and paste your three keys:
```
BITGET_API_KEY=...
BITGET_API_SECRET=...
BITGET_API_PASSPHRASE=...
LIVE_TRADING_ENABLED=false
```
Set `starting_equity` in `settings.yaml` to your deposit in USDT (e.g. 215).

### 4. Backtest, then paper trade (no real money)
```bash
python main.py backtest --days 365          # see historical behavior
python main.py run --mode paper             # trade live prices, fake fills
```
Let paper mode run for **at least a few weeks**. Watch `logs/bot.log` and
`state/paper/trades.jsonl`. You're checking that it behaves sanely, not that
it's instantly profitable. Stop it anytime with Ctrl-C; it resumes state on
restart.

### 5. Go live - small, and only when ready
Only after paper trading has convinced you:
1. In `.env` set `LIVE_TRADING_ENABLED=true`.
2. Start with a small balance you can afford to lose entirely.
3. Run:
```bash
python main.py run --mode live --i-understand-the-risk
```
Both the `.env` flag and the `--i-understand-the-risk` flag are required on
purpose - two deliberate steps so you never start live by accident.

### 6. Keep it running and top up
- Run it on a machine that stays on (a cheap VPS, or a Raspberry Pi).
  Inside `tmux`/`screen` or as a `systemd` service so it survives logouts.
- Your planned monthly top-ups: just deposit more USDT into the futures
  wallet. In live mode the bot reads your real balance each cycle and sizes
  up automatically - no config edit, no restart needed.
- Re-run `python main.py optimize` every month or so and update
  `settings.yaml` if the walk-forward picks new parameters.

### Safety reminders
- Never enable withdrawal permission on the API key.
- Never commit your `.env` (it's gitignored - keep it that way).
- The bot can lose money, including all of it. Only trade what you can lose.
- If anything looks wrong, Ctrl-C stops it; the `max_drawdown_pct` breaker
  also halts new trades automatically past a 15% drawdown.

## Uploaded reference files

This repo replaces the sketches in the uploaded `claude_dev_framework.py` /
`strategy_experiments.py` (a backtest-only toy framework, no exchange
integration, no risk management, and a couple of indicator bugs - e.g. its
MACD signal line was computed by smoothing a single-element list) with a
complete, runnable system: real Bitget connectivity, proper indicator math,
and the risk controls a live bot actually needs.
