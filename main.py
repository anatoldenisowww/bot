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
from config import load_config
from exchange import MarketDataFeed
from logging_setup import setup_logging

logger = logging.getLogger(__name__)


def cmd_backtest(args: argparse.Namespace) -> None:
    cfg = load_config()
    setup_logging(cfg.runtime.log_dir)

    if args.timeframe:
        cfg.exchange.timeframe = args.timeframe

    feed = MarketDataFeed(cfg)
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=args.days)).timestamp() * 1000)

    price_data = {}
    for symbol in cfg.exchange.symbols:
        logger.info("fetching %s %s history since %s", symbol, cfg.exchange.timeframe, since_ms)
        df = feed.fetch_ohlcv_since(symbol, cfg.exchange.timeframe, since_ms)
        if df.empty:
            print(f"WARNING: no data returned for {symbol}, skipping it from this backtest")
            continue
        price_data[symbol] = df

    if len(price_data) < 1:
        print("No market data fetched for any symbol - aborting.")
        sys.exit(1)

    result = run_backtest(cfg, price_data)
    s = result.stats

    print("\n=== Backtest results ===")
    print(tabulate(
        [
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

    bt = sub.add_parser("backtest", help="backtest the strategy against historical Bitget data")
    bt.add_argument("--days", type=int, default=365, help="how many days of history to fetch")
    bt.add_argument("--timeframe", type=str, default=None, help="override the execution timeframe from settings.yaml, e.g. 1h")
    bt.add_argument("--save-trades", type=str, default=None, help="optional path to save closed trades as JSONL")
    bt.set_defaults(func=cmd_backtest)

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
