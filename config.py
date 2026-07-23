"""Configuration loading: settings.yaml (strategy/risk knobs) + .env (secrets)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent


@dataclass
class ExchangeConfig:
    id: str
    market_type: str
    quote: str
    symbols: List[str]
    timeframe: str
    trend_timeframe: str
    leverage: int
    margin_mode: str
    taker_fee_bps: float
    slippage_bps: float


@dataclass
class IndicatorConfig:
    ema_fast: int
    ema_slow: int
    ema_trend_filter: int
    rsi_period: int
    rsi_oversold: float
    rsi_overbought: float
    bb_period: int
    bb_std: float
    atr_period: int
    adx_period: int
    adx_trend_threshold: float
    donchian_period: int
    rsi_fast_period: int
    rsi_fast_oversold: float
    rsi_fast_overbought: float


@dataclass
class StrategyConfig:
    mode: str  # "ensemble" (regime-routed trend/breakout/mean-reversion) | "mean_reversion_scalp"


@dataclass
class RiskConfig:
    starting_equity: float
    risk_per_trade_pct: float
    max_leverage: int
    atr_stop_multiplier: float
    take_profit_r_multiple: float
    trailing_activation_r: float
    trailing_atr_multiplier: float
    max_concurrent_positions: int
    max_portfolio_risk_pct: float
    max_daily_loss_pct: float
    max_drawdown_pct: float


@dataclass
class RuntimeConfig:
    poll_interval_seconds: int
    state_dir: str
    log_dir: str


@dataclass
class Secrets:
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""
    live_trading_enabled: bool = False


@dataclass
class Config:
    exchange: ExchangeConfig
    indicators: IndicatorConfig
    strategy: StrategyConfig
    risk: RiskConfig
    runtime: RuntimeConfig
    secrets: Secrets = field(default_factory=Secrets)


def load_config(settings_path: str | Path = ROOT / "settings.yaml", env_path: str | Path = ROOT / ".env") -> Config:
    load_dotenv(env_path, override=False)

    with open(settings_path, "r") as f:
        raw = yaml.safe_load(f)

    exchange = ExchangeConfig(**raw["exchange"])
    indicators = IndicatorConfig(**raw["indicators"])
    strategy = StrategyConfig(**raw["strategy"])
    risk = RiskConfig(**raw["risk"])
    runtime = RuntimeConfig(**raw["runtime"])

    secrets = Secrets(
        api_key=os.getenv("BITGET_API_KEY", ""),
        api_secret=os.getenv("BITGET_API_SECRET", ""),
        api_passphrase=os.getenv("BITGET_API_PASSPHRASE", ""),
        live_trading_enabled=os.getenv("LIVE_TRADING_ENABLED", "false").strip().lower() == "true",
    )

    return Config(exchange=exchange, indicators=indicators, strategy=strategy, risk=risk, runtime=runtime, secrets=secrets)
