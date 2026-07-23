"""Bitget connectivity via ccxt.

Two concerns are kept separate on purpose:

- MarketDataFeed: public endpoints only (candles, ticker price). Needs no API
  key, so paper trading works out of the box with zero credentials.
- LiveBroker: private endpoints (place orders, leverage). Only ever
  instantiated when the caller has explicitly enabled live trading.
- PaperBroker: implements the same execute_market_order() interface as
  LiveBroker but fills against real market data locally, applying the fee
  and slippage assumptions from config. This is the default broker.

bot.py and backtester.py talk to whichever broker they're given without
caring which one it is (duck typing on execute_market_order/get_price).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal, Optional

import ccxt
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)

OrderSide = Literal["buy", "sell"]


@dataclass
class Fill:
    price: float
    fee: float
    qty: float


def _make_ccxt_client(cfg: Config, with_credentials: bool) -> ccxt.Exchange:
    params = {
        "enableRateLimit": True,
        "options": {"defaultType": cfg.exchange.market_type},
        # Let the underlying requests session honor standard CA/proxy env vars
        # (REQUESTS_CA_BUNDLE, HTTPS_PROXY, ...) instead of ccxt's hardened
        # default of ignoring them.
        "requests_trust_env": True,
    }
    if with_credentials:
        params.update(
            apiKey=cfg.secrets.api_key,
            secret=cfg.secrets.api_secret,
            password=cfg.secrets.api_passphrase,
        )
    exchange_class = getattr(ccxt, cfg.exchange.id)
    return exchange_class(params)


class MarketDataFeed:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = _make_ccxt_client(cfg, with_credentials=False)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 300, since: Optional[int] = None) -> pd.DataFrame:
        raw = self.client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    def get_price(self, symbol: str) -> float:
        ticker = self.client.fetch_ticker(symbol)
        return float(ticker["last"])

    def fetch_ohlcv_since(self, symbol: str, timeframe: str, since_ms: int, page_limit: int = 200) -> pd.DataFrame:
        """Page through fetch_ohlcv from since_ms up to now. Used by the backtester
        to pull more history than a single request returns."""
        all_rows: list = []
        cursor = since_ms
        now_ms = self.client.milliseconds()

        stall_count = 0
        while cursor < now_ms:
            batch = self.client.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=page_limit)
            if not batch:
                break
            all_rows.extend(batch)
            last_ts = batch[-1][0]
            if last_ts <= cursor:
                # Some venues/timeframes occasionally return a page that doesn't
                # advance the cursor (e.g. a short page ending exactly on a
                # boundary). Don't treat that alone as "no more data" - only
                # bail out once it happens repeatedly, to avoid an infinite loop.
                stall_count += 1
                if stall_count >= 3:
                    break
                cursor += 1
            else:
                stall_count = 0
                cursor = last_ts + 1
            time.sleep(self.client.rateLimit / 1000.0)

        if not all_rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df


class LiveBroker:
    """Places real orders on Bitget. Only construct this when live trading
    has been explicitly enabled - see main.py's confirmation gate."""

    def __init__(self, cfg: Config, market_data: MarketDataFeed):
        if not cfg.secrets.api_key or not cfg.secrets.api_secret or not cfg.secrets.api_passphrase:
            raise RuntimeError("Live trading requires BITGET_API_KEY/SECRET/PASSPHRASE to be set in .env")
        self.cfg = cfg
        self.market_data = market_data
        self.client = _make_ccxt_client(cfg, with_credentials=True)
        self._leverage_set = set()

    def get_price(self, symbol: str) -> float:
        return self.market_data.get_price(symbol)

    def ensure_leverage(self, symbol: str) -> None:
        if symbol in self._leverage_set:
            return
        try:
            self.client.set_leverage(
                self.cfg.exchange.leverage,
                symbol,
                params={"marginMode": self.cfg.exchange.margin_mode},
            )
        except Exception as exc:  # pragma: no cover - exchange-side, needs live testing
            logger.warning("could not set leverage for %s (may already be set): %s", symbol, exc)
        self._leverage_set.add(symbol)

    def execute_market_order(self, symbol: str, side: OrderSide, qty: float, reduce_only: bool = False) -> Fill:
        self.ensure_leverage(symbol)
        order = self.client.create_order(
            symbol, "market", side, qty, params={"reduceOnly": reduce_only},
        )
        # ccxt market orders don't always return an immediate fill price;
        # poll briefly for the average fill price, then fall back to ticker.
        price = order.get("average") or order.get("price")
        if not price:
            for _ in range(5):
                time.sleep(1)
                fetched = self.client.fetch_order(order["id"], symbol)
                price = fetched.get("average") or fetched.get("price")
                if price:
                    break
        if not price:
            price = self.get_price(symbol)

        fee = 0.0
        if order.get("fee") and order["fee"].get("cost"):
            fee = float(order["fee"]["cost"])
        else:
            fee = price * qty * (self.cfg.exchange.taker_fee_bps / 10000.0)

        return Fill(price=float(price), fee=fee, qty=qty)

    def get_account_equity(self) -> float:
        balance = self.client.fetch_balance(params={"type": self.cfg.exchange.market_type})
        usdt = balance.get(self.cfg.exchange.quote, {})
        total = usdt.get("total")
        return float(total) if total is not None else 0.0


class PaperBroker:
    """Simulated execution against real market prices. Default and safe."""

    def __init__(self, cfg: Config, market_data: MarketDataFeed):
        self.cfg = cfg
        self.market_data = market_data

    def get_price(self, symbol: str) -> float:
        return self.market_data.get_price(symbol)

    def execute_market_order(self, symbol: str, side: OrderSide, qty: float, reduce_only: bool = False) -> Fill:
        price = self.get_price(symbol)
        slippage = self.cfg.exchange.slippage_bps / 10000.0
        fill_price = price * (1 + slippage) if side == "buy" else price * (1 - slippage)
        fee = fill_price * qty * (self.cfg.exchange.taker_fee_bps / 10000.0)
        return Fill(price=fill_price, fee=fee, qty=qty)
