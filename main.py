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

from tabulate import tabulate

from backtester import run_backtest
from bot import QuantTradingBot
from config import Config, load_config
from exchange import MarketDataFeed
from logging_setup import setup_logging
from optimize import MIN_WIN_RATE_PCT, most_common_params, walk_forward_optimize

logger = logging.getLogger(__name__)


def _fetch_price_data(cfg: Config, feed: MarketDataFeed, days: int) -> dict:
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    price_data = {}
    for symbol in cfg.exchange.symbols:
        logger.info("fetching %s %s history since %s", symbol, cfg.exchange.timeframe, since_ms)
        df = feed.fetch_ohlcv_since(symbol, cfg.exchange.timeframe, since_ms)
        if df.empty:
            print(f"WARNING: no data returned for {symbol}, skipping it")
            continue
        price_data[symbol] = df
    return price_data


def cmd_backtest(args: argparse.Namespace) -> None:
    cfg = load_config()
    setup_logging(cfg.runtime.log_dir)

    if args.timeframe:
        cfg.exchange.timeframe = args.timeframe
    if args.strategy_mode:
        cfg.strategy.mode = args.strategy_mode

    feed = MarketDataFeed(cfg)
    price_data = _fetch_price_data(cfg, feed, args.days)

    if len(price_data) < 1:
        print("No market data fetched for any symbol - aborting.")
        sys.exit(1)

    result = run_backtest(cfg, price_data)
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
    price_data = _fetch_price_data(cfg, feed, args.days)
    if len(price_data) < 1:
        print("No market data fetched for any symbol - aborting.")
        sys.exit(1)

    print(
        f"Strategy: {cfg.strategy.mode}\n"
        f"Running walk-forward optimization: {args.train_days}d train / {args.test_days}d test "
        f"windows, stepping {args.step_days or args.test_days}d, over {args.days}d of {cfg.exchange.timeframe} "
        f"history. Every fold's parameters are chosen only from data before that fold's test window.\n"
    )

    result = walk_forward_optimize(
        cfg, price_data, train_days=args.train_days, test_days=args.test_days, step_days=args.step_days,
    )

    rows = []
    for f in result.folds:
        rows.append([
            f.fold_index,
            f.test_start.strftime("%Y-%m-%d"),
            f.test_end.strftime("%Y-%m-%d"),
            f"adx>={f.best_params['adx_trend_threshold']:.0f} atr_x{f.best_params['atr_stop_multiplier']:.1f} tp{f.best_params['take_profit_r_multiple']:.1f}R" + ("" if f.guardrails_cleared else " (fallback*)"),
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
        print(
            f"\n* fold(s) {fallback_folds}: no combination in the parameter grid hit "
            f"{MIN_WIN_RATE_PCT:.0f}%+ training win rate together with the drawdown/trade-count "
            "guardrails, so that fold fell back to settings.yaml's existing defaults instead of "
            "forcing a pick. This is being reported, not hidden."
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

    strategy_choices = ["ensemble", "mean_reversion_scalp"]

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
    opt.add_argument("--save-folds", type=str, default=None, help="optional path to save per-fold summaries as JSONL")
    opt.set_defaults(func=cmd_optimize)

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
