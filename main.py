"""CLI entrypoint.

    python main.py backtest --days 365
    python main.py run --mode paper
    python main.py run --mode live --i-understand-the-risk
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from tabulate import tabulate

from backtester import run_backtest
from bot import QuantTradingBot
from config import Config, load_config
from exchange import MarketDataFeed, MarketLimits
from logging_setup import setup_logging
from optimize import most_common_params, walk_forward_optimize

logger = logging.getLogger(__name__)


def _resolve_symbols(cfg: Config, feed: MarketDataFeed) -> List[str]:
    if cfg.strategy.mode != "cross_sectional_momentum":
        return list(cfg.exchange.symbols)

    symbols = feed.fetch_liquid_universe(cfg.strategy.universe_top_n, cfg.strategy.universe_min_quote_volume_24h)
    print(f"Cross-sectional universe - today's liquidity snapshot, {len(symbols)} symbols: {symbols}")
    print(
        "Note: backtest/optimize apply TODAY's liquidity ranking across the whole historical window, not a "
        "true point-in-time reconstruction (that would need historical daily volume per symbol). A symbol "
        "that only recently became liquid enough to rank will still show its full price history in this "
        "backtest even though it wouldn't have been selected back then - a real, known simplification, not "
        "survivorship-bias-free. Live/paper trading doesn't have this issue: it re-scans the actual current "
        "board every universe_refresh_hours.\n"
    )
    return symbols


def _fetch_price_data(cfg: Config, feed: MarketDataFeed, days: int, symbols: List[str]) -> dict:
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    price_data = {}
    for symbol in symbols:
        logger.info("fetching %s %s history since %s", symbol, cfg.exchange.timeframe, since_ms)
        df = feed.fetch_ohlcv_since(symbol, cfg.exchange.timeframe, since_ms)
        if df.empty:
            print(f"WARNING: no data returned for {symbol}, skipping it")
            continue
        price_data[symbol] = df
    return price_data


def _fetch_market_limits(feed: MarketDataFeed, symbols: List[str]) -> Dict[str, MarketLimits]:
    limits = {}
    for symbol in symbols:
        try:
            limits[symbol] = feed.get_market_limits(symbol)
        except Exception:
            logger.exception("failed to fetch market limits for %s, entries on it won't be size-clamped", symbol)
    return limits


def cmd_backtest(args: argparse.Namespace) -> None:
    cfg = load_config()
    setup_logging(cfg.runtime.log_dir)

    if args.timeframe:
        cfg.exchange.timeframe = args.timeframe
    if args.strategy_mode:
        cfg.strategy.mode = args.strategy_mode

    feed = MarketDataFeed(cfg)
    symbols = _resolve_symbols(cfg, feed)
    price_data = _fetch_price_data(cfg, feed, args.days, symbols)

    if len(price_data) < 1:
        print("No market data fetched for any symbol - aborting.")
        sys.exit(1)

    market_limits = _fetch_market_limits(feed, list(price_data.keys()))
    result = run_backtest(cfg, price_data, market_limits=market_limits)
    s = result.stats

    print("\n=== Backtest results ===")
    print(tabulate(
        [
            ["Strategy", cfg.strategy.mode],
            ["Symbols", ", ".join(price_data.keys())],
            ["Period", f"{args.days} days"],
            ["Starting equity", f"${cfg.risk.starting_equity:,.2f}"],
            ["Final equity", f"${s.final_equity:,.2f}"],
            ["Total return", f"{s.total_return_pct:+.2f}%"],
            ["Max drawdown", f"{s.max_drawdown_pct:.2f}%"],
            ["Sharpe (annualized)", f"{s.sharpe_ratio:.2f}"],
            ["Total trades", s.total_trades],
            ["Win rate", f"{s.win_rate:.1f}%"],
            ["Profit factor", f"{s.profit_factor:.2f}"],
            ["Avg R multiple", f"{s.avg_r_multiple:.2f}"],
        ],
        tablefmt="simple",
    ))
    print(
        "\nPast performance on historical data is not indicative of future results. "
        "This is a backtest, not a promise.\n"
    )

    if args.save_trades:
        import json
        with open(args.save_trades, "w") as f:
            for t in result.trades:
                record = {
                    "symbol": t.symbol, "side": t.side, "entry_price": t.entry_price,
                    "exit_price": t.exit_price, "qty": t.qty, "pnl": t.pnl, "fees": t.fees,
                    "reason": t.reason, "r_multiple": t.r_multiple,
                    "opened_at": t.opened_at.isoformat(), "closed_at": t.closed_at.isoformat(),
                }
                f.write(json.dumps(record) + "\n")
        print(f"Saved {len(result.trades)} trades to {args.save_trades}")


def cmd_optimize(args: argparse.Namespace) -> None:
    cfg = load_config()
    setup_logging(cfg.runtime.log_dir)

    if args.timeframe:
        cfg.exchange.timeframe = args.timeframe
    if args.strategy_mode:
        cfg.strategy.mode = args.strategy_mode

    feed = MarketDataFeed(cfg)
    symbols = _resolve_symbols(cfg, feed)
    price_data = _fetch_price_data(cfg, feed, args.days, symbols)
    if len(price_data) < 1:
        print("No market data fetched for any symbol - aborting.")
        sys.exit(1)

    market_limits = _fetch_market_limits(feed, list(price_data.keys()))

    print(
        f"Strategy: {cfg.strategy.mode}\n"
        f"Running walk-forward optimization: {args.train_days}d train / {args.test_days}d test "
        f"windows, stepping {args.step_days or args.test_days}d, over {args.days}d of {cfg.exchange.timeframe} "
        f"history. Every fold's parameters are chosen only from data before that fold's test window.\n"
    )

    if args.min_win_rate > 0:
        print(
            f"Win-rate floor active: only parameter sets with >={args.min_win_rate:.0f}% TRAINING win rate "
            "are eligible. Note we've shown this tends to REDUCE profitability (momentum structurally wins "
            "<50% of the time); it's here because you can ask for it, not because it's recommended.\n"
        )

    result = walk_forward_optimize(
        cfg, price_data, train_days=args.train_days, test_days=args.test_days, step_days=args.step_days,
        market_limits=market_limits, min_win_rate_pct=args.min_win_rate,
    )

    def _format_params(params: dict) -> str:
        # abbreviate whatever knobs this strategy's grid actually tuned
        abbrev = {
            "adx_trend_threshold": "adx>=", "atr_stop_multiplier": "atr_x",
            "take_profit_r_multiple": "tp", "momentum_lookback_bars": "lb",
            "min_abs_momentum_score": "minscore",
        }
        parts = []
        for key, value in params.items():
            label = abbrev.get(key, key + "=")
            parts.append(f"{label}{value:g}")
        return " ".join(parts)

    rows = []
    for f in result.folds:
        params_str = _format_params(f.best_params)
        rows.append([
            f.fold_index,
            f.test_start.strftime("%Y-%m-%d"),
            f.test_end.strftime("%Y-%m-%d"),
            params_str + ("" if f.guardrails_cleared else " (fallback*)"),
            f"{f.train_stats.win_rate:.0f}%",
            f.train_stats.total_trades,
            f"{f.train_stats.sharpe_ratio:.2f}",
            f.test_result.stats.total_trades,
            f"{f.test_result.stats.win_rate:.0f}%",
            f"{f.test_result.stats.total_return_pct:+.2f}%",
            f"{f.test_result.stats.sharpe_ratio:.2f}",
        ])

    print("=== Per-fold results (train chooses params, test is genuinely out-of-sample) ===")
    print(tabulate(
        rows,
        headers=["fold", "OOS start", "OOS end", "chosen params", "train WR", "train trades", "train sharpe", "OOS trades", "OOS WR", "OOS return", "OOS sharpe"],
        tablefmt="simple",
    ))

    fallback_folds = [f.fold_index for f in result.folds if not f.guardrails_cleared]
    if fallback_folds:
        constraint_desc = f"{args.min_win_rate:.0f}%+ training win rate together with the " if args.min_win_rate > 0 else "the "
        print(
            f"\n* fold(s) {fallback_folds}: no combination in the parameter grid cleared "
            f"{constraint_desc}drawdown/trade-count guardrails, so that fold fell back to "
            "settings.yaml's existing defaults instead of forcing a pick. This is being reported, not hidden."
        )

    s = result.chained_stats
    print("\n=== Chained out-of-sample performance (the honest headline number) ===")
    print(tabulate(
        [
            ["Symbols", ", ".join(price_data.keys())],
            ["Total OOS period", f"{args.days - args.train_days} days (approx)"],
            ["Starting equity", f"${cfg.risk.starting_equity:,.2f}"],
            ["Final equity", f"${s.final_equity:,.2f}"],
            ["Total return", f"{s.total_return_pct:+.2f}%"],
            ["Max drawdown", f"{s.max_drawdown_pct:.2f}%"],
            ["Sharpe (annualized)", f"{s.sharpe_ratio:.2f}"],
            ["Total trades", s.total_trades],
            ["Win rate", f"{s.win_rate:.1f}%"],
            ["Profit factor", f"{s.profit_factor:.2f}"],
        ],
        tablefmt="simple",
    ))
    print(
        "\nNote on max drawdown above: it's measured across the whole chained curve, so a peak set "
        "near the end of one fold's good run can be followed by losses at the start of the next fold's "
        "different parameter set - a bigger swing than the live bot would actually allow. The live/paper "
        f"bot tracks its own equity peak continuously and halts new trades at max_drawdown_pct "
        f"({cfg.risk.max_drawdown_pct:.0f}% in settings.yaml) - it would have intervened before this "
        "backtest's worst stretch played out in full."
    )

    recommended = most_common_params(result)
    print(
        f"\nMost frequently re-selected parameters across folds (a reasonable settings.yaml "
        f"starting point, not a guarantee): {recommended}\n"
    )
    print(
        "This is still a backtest, now validated the way a real desk would insist on - out-of-sample, "
        "walk-forward, re-optimized on a schedule. It is not a promise of future returns, and this "
        "system should keep being walk-forward re-validated periodically as markets change.\n"
    )

    if args.save_folds:
        import json
        with open(args.save_folds, "w") as f:
            for fold in result.folds:
                f.write(json.dumps({
                    "fold_index": fold.fold_index,
                    "test_start": fold.test_start.isoformat(),
                    "test_end": fold.test_end.isoformat(),
                    "best_params": fold.best_params,
                    "train_sharpe": fold.train_stats.sharpe_ratio,
                    "oos_return_pct": fold.test_result.stats.total_return_pct,
                    "oos_sharpe": fold.test_result.stats.sharpe_ratio,
                    "oos_trades": fold.test_result.stats.total_trades,
                }) + "\n")
        print(f"Saved {len(result.folds)} fold summaries to {args.save_folds}")


def cmd_regime(args: argparse.Namespace) -> None:
    from regime_hmm import fit_regimes

    cfg = load_config()
    setup_logging(cfg.runtime.log_dir)
    timeframe = args.timeframe or cfg.exchange.timeframe
    feed = MarketDataFeed(cfg)

    reports = []
    for symbol in args.symbols:
        logger.info("fetching %s %s history for HMM regime detection", symbol, timeframe)
        df = feed.fetch_ohlcv(symbol, timeframe, limit=args.bars)
        if len(df) < 60:
            print(f"Not enough data for {symbol} ({len(df)} bars) - skipping.")
            continue
        reports.append(fit_regimes(df, n_states=args.states, timeframe=timeframe, symbol=symbol))

    if not reports:
        print("No symbols had enough data - aborting.")
        sys.exit(1)

    print("\nThink of each market as having a few 'moods'. The model labels every candle with the")
    print("mood it was most likely in, then reads the plain-language lean that mood historically rewarded.")
    print("  BUY = prices historically rose · HOLD = choppy/sideways, wait · SELL = prices historically fell\n")

    for report in reports:
        print("=" * 74)
        print(f"{report.symbol}  ({timeframe}, last {len(report.closes)} candles)")
        print("=" * 74)
        print(tabulate(
            [
                [s.index, s.label, s.lean, f"{s.mean_return_pct:+.3f}%", f"{s.volatility_pct:.3f}%",
                 f"{s.frequency_pct:.0f}%",
                 ("inf" if s.expected_duration_bars == float("inf") else f"{s.expected_duration_bars:.0f}")]
                for s in report.states
            ],
            headers=["#", "market mood", "lean", "avg move/candle", "swinginess", "how often", "lasts"],
            tablefmt="simple",
        ))
        cur = report.current
        conf = report.current_state_probs[report.current_state] * 100
        print(f"\n>>> RIGHT NOW: {cur.lean} - '{cur.label}' ({conf:.0f}% confident)")
        print(f"    {cur.plain}")
        nxt = report.expected_next_state()
        if nxt.index != cur.index:
            print(f"    If the mood changes, it most often shifts to '{nxt.label}' (lean {nxt.lean}).\n")
        else:
            print("    This mood is 'sticky' - it most often continues rather than switching.\n")

    print(
        "IMPORTANT: this describes what HAPPENED in each mood in the past - it is NOT a prediction and NOT\n"
        "financial advice. Moods flip without warning and detecting the current one has lag. Use it as\n"
        "context (what kind of weather is it?), not a blind buy/sell button.\n"
    )

    if args.html:
        from regime_report import render_multi_regime_html
        with open(args.html, "w") as f:
            f.write(render_multi_regime_html(reports))
        print(f"Wrote colorful HTML dashboard ({len(reports)} symbols) to {args.html}")


def cmd_pairs(args: argparse.Namespace) -> None:
    from statarb import PairConfig, analyze_pair, all_pairs

    cfg = load_config()
    setup_logging(cfg.runtime.log_dir)
    timeframe = args.timeframe or cfg.exchange.timeframe
    feed = MarketDataFeed(cfg)

    since_ms = int((datetime.now(timezone.utc) - timedelta(days=args.days)).timestamp() * 1000)
    data = {}
    for sym in args.symbols:
        logger.info("fetching %s %s history for pairs analysis", sym, timeframe)
        df = feed.fetch_ohlcv_since(sym, timeframe, since_ms)
        if not df.empty:
            data[sym] = df

    pairs = [(a, b) for a, b in all_pairs(args.symbols) if a in data and b in data]
    if not pairs:
        print("Not enough symbols with data to form a pair - aborting.")
        sys.exit(1)

    pair_cfg = PairConfig(entry_z=args.entry_z, exit_z=args.exit_z, stop_z=args.stop_z, zscore_window=args.window)
    reports = []
    for a, b in pairs:
        try:
            reports.append(analyze_pair(data[a], data[b], a, b, pair_cfg))
        except ValueError as exc:
            print(f"skip {a}/{b}: {exc}")

    print(f"\n=== Pairs-trading screener ({timeframe}, {args.days}d) ===")
    print("'reverts in' = spread half-life (lower = mean-reverts faster = more tradeable).")
    print("A very long half-life means the coins trend together and the spread does NOT revert.\n")
    rows = []
    for r in reports:
        s, bt = r.stats, r.backtest
        hl = "inf" if s.half_life_bars == float("inf") else f"{s.half_life_bars:.0f}"
        pf = "inf" if bt.profit_factor == float("inf") else f"{bt.profit_factor:.2f}"
        rows.append([
            f"{s.symbol_a.split('/')[0]}-{s.symbol_b.split('/')[0]}",
            f"{s.correlation:.2f}", f"{hl}b", f"{s.current_z:+.2f}", s.current_signal,
            f"{bt.total_return_pct:+.1f}%", pf, f"{bt.win_rate:.0f}%", f"{bt.max_drawdown_pct:.0f}%",
        ])
    print(tabulate(
        rows,
        headers=["pair", "corr", "reverts in", "z now", "signal", "bt return", "PF", "win%", "maxDD"],
        tablefmt="simple",
    ))
    print(
        "\nStat arb is NOT pure arbitrage: no guaranteed convergence, individual trades lose, relationships "
        "break. Costs hit 4 legs per round trip. Validate out-of-sample before trusting any of this.\n"
    )

    if args.html:
        from statarb_report import render_statarb_html
        with open(args.html, "w") as f:
            f.write(render_statarb_html(reports, pair_cfg, timeframe))
        print(f"Wrote colorful pairs dashboard to {args.html}")


def cmd_carry(args: argparse.Namespace) -> None:
    from funding_carry import CarryConfig, backtest_carry, screen

    cfg = load_config()
    setup_logging(cfg.runtime.log_dir)
    feed = MarketDataFeed(cfg)

    symbols = args.symbols
    if args.scan_top:
        symbols = feed.fetch_liquid_universe(args.scan_top, cfg.strategy.universe_min_quote_volume_24h)
        print(f"Scanning the {len(symbols)} most liquid perps for persistent positive funding...\n")

    carry_cfg = CarryConfig(hold=args.hold)
    reports = []
    for sym in symbols:
        logger.info("fetching funding history for %s", sym)
        try:
            fh = feed.fetch_funding_history(sym, days=args.days)
            reports.append(backtest_carry(fh, carry_cfg, symbol=sym))
        except ValueError as exc:
            print(f"skip {sym}: {exc}")

    if not reports:
        print("No symbols had enough funding history - aborting.")
        sys.exit(1)

    ordered = screen(reports)
    print("=== Funding-carry screener (cash-and-carry: long spot + short perp, delta-neutral) ===")
    print("'persistence' = % of intervals funding was positive (the edge). 'carry/yr' held long-term.\n")
    print(tabulate(
        [
            [r.stats.symbol.split("/")[0], f"{r.stats.days_covered:.0f}d", f"{r.stats.persistence_pct:.0f}%",
             f"{r.stats.sustainable_annual_pct:+.1f}%", f"{r.stats.one_time_cost_pct:.2f}%",
             "yes" if r.stats.worth_it else "no"]
            for r in ordered
        ],
        headers=["coin", "sample", "persistence", "carry/yr", "1x cost", "worth it"],
        tablefmt="simple",
    ))
    print(
        "\nHONEST READ: this is the one real, structural edge here - but it's modest (single-digit %/yr) and\n"
        "market-neutral 'rent', not a jackpot. Funding can flip negative (already counted). Needs a spot AND a\n"
        "perp leg. Sample is only ~33 days (Bitget's limit), so treat annualized figures as indicative.\n"
    )

    if args.html:
        from funding_carry_report import render_carry_html
        with open(args.html, "w") as f:
            f.write(render_carry_html(reports, carry_cfg))
        print(f"Wrote colorful funding-carry dashboard to {args.html}")


def cmd_run(args: argparse.Namespace) -> None:
    cfg = load_config()
    setup_logging(cfg.runtime.log_dir)

    mode = args.mode
    if mode == "live":
        if not cfg.secrets.live_trading_enabled:
            print(
                "Refusing to start live trading: LIVE_TRADING_ENABLED is not 'true' in your .env file.\n"
                "This is an intentional safety interlock. Set it explicitly once you have paper-traded "
                "this bot and understand the risk."
            )
            sys.exit(1)
        if not args.i_understand_the_risk:
            print(
                "Refusing to start live trading: pass --i-understand-the-risk explicitly.\n"
                "Crypto trading bots can and do lose money, including all of the deposited capital. "
                "No strategy here is guaranteed to be profitable."
            )
            sys.exit(1)
        print("*** LIVE TRADING MODE - real orders will be placed on Bitget with real funds ***")

    bot = QuantTradingBot(cfg, mode=mode)

    def _handle_signal(signum, frame):
        logger.info("received signal %s, shutting down after current cycle", signum)
        bot.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    bot.run_forever()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bitget quant trading bot for BTC/ETH/SOL")
    sub = parser.add_subparsers(dest="command", required=True)

    strategy_choices = ["ensemble", "mean_reversion_scalp", "cross_sectional_momentum"]

    bt = sub.add_parser("backtest", help="backtest the strategy against historical Bitget data")
    bt.add_argument("--days", type=int, default=365, help="how many days of history to fetch")
    bt.add_argument("--timeframe", type=str, default=None, help="override the execution timeframe from settings.yaml, e.g. 1h")
    bt.add_argument("--strategy-mode", choices=strategy_choices, default=None, help="override strategy.mode from settings.yaml")
    bt.add_argument("--save-trades", type=str, default=None, help="optional path to save closed trades as JSONL")
    bt.set_defaults(func=cmd_backtest)

    opt = sub.add_parser("optimize", help="walk-forward parameter search - train on rolling windows, validate strictly out-of-sample")
    opt.add_argument("--days", type=int, default=700, help="total days of history to fetch")
    opt.add_argument("--timeframe", type=str, default=None, help="override the execution timeframe from settings.yaml")
    opt.add_argument("--strategy-mode", choices=strategy_choices, default=None, help="override strategy.mode from settings.yaml")
    opt.add_argument("--train-days", type=int, default=300, help="length of each training window")
    opt.add_argument("--test-days", type=int, default=100, help="length of each out-of-sample test window")
    opt.add_argument("--step-days", type=int, default=None, help="how far to roll forward between folds (defaults to --test-days)")
    opt.add_argument("--min-win-rate", type=float, default=0.0, help="optional training win-rate floor (%%). Off by default; we've shown it tends to reduce profitability")
    opt.add_argument("--save-folds", type=str, default=None, help="optional path to save per-fold summaries as JSONL")
    opt.set_defaults(func=cmd_optimize)

    reg = sub.add_parser("regime", help="fit a Hidden Markov Model and report each market's current regime (mood)")
    reg.add_argument("--symbols", nargs="+", default=["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"],
                     help="symbols to analyze (default: BTC ETH SOL)")
    reg.add_argument("--timeframe", type=str, default=None, help="candle timeframe (defaults to settings.yaml)")
    reg.add_argument("--states", type=int, default=3, help="number of hidden regimes/moods to fit (2-4 is typical)")
    reg.add_argument("--bars", type=int, default=1000, help="how many recent candles to fit on")
    reg.add_argument("--html", type=str, default=None, help="optional path to write a colorful HTML dashboard")
    reg.set_defaults(func=cmd_regime)

    pr = sub.add_parser("pairs", help="statistical-arbitrage screener: which crypto pairs are worth pairs-trading")
    pr.add_argument("--symbols", nargs="+", default=["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"],
                    help="symbols to form pairs from (default: BTC ETH SOL)")
    pr.add_argument("--timeframe", type=str, default=None, help="candle timeframe (defaults to settings.yaml)")
    pr.add_argument("--days", type=int, default=700, help="how many days of history to analyze")
    pr.add_argument("--window", type=int, default=60, help="rolling window (bars) for the spread z-score")
    pr.add_argument("--entry-z", type=float, default=2.0, help="open a trade when |z| exceeds this")
    pr.add_argument("--exit-z", type=float, default=0.5, help="close when |z| falls back inside this")
    pr.add_argument("--stop-z", type=float, default=4.0, help="bail out if |z| blows past this (relationship broke)")
    pr.add_argument("--html", type=str, default=None, help="optional path to write a colorful HTML dashboard")
    pr.set_defaults(func=cmd_pairs)

    ca = sub.add_parser("carry", help="funding-rate carry screener: the one real structural edge (collect perpetual funding, delta-neutral)")
    ca.add_argument("--symbols", nargs="+",
                    default=["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "DOGE/USDT:USDT",
                             "XRP/USDT:USDT", "LINK/USDT:USDT", "AVAX/USDT:USDT"],
                    help="symbols to screen for harvestable funding")
    ca.add_argument("--scan-top", type=int, default=0, help="instead of --symbols, scan the N most liquid perps")
    ca.add_argument("--days", type=int, default=365, help="days of funding history to request (Bitget serves ~33)")
    ca.add_argument("--hold", choices=["always", "positive_only"], default="always",
                    help="'always' = hold continuously (realistic); 'positive_only' = flip out on negative funding (usually worse)")
    ca.add_argument("--html", type=str, default=None, help="optional path to write a colorful HTML dashboard")
    ca.set_defaults(func=cmd_carry)

    run = sub.add_parser("run", help="run the bot continuously")
    run.add_argument("--mode", choices=["paper", "live"], default="paper")
    run.add_argument("--i-understand-the-risk", action="store_true", help="required to start live mode")
    run.set_defaults(func=cmd_run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
