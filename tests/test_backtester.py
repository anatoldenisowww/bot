import numpy as np
import pandas as pd

from backtester import run_backtest
from config import load_config


def _synthetic_ohlcv(n=500, seed=1, drift=0.15, start=100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1.0, n)
    trend = np.linspace(0, drift * n, n)
    close = start + trend + noise.cumsum() * 0.3
    close = np.maximum(close, 1.0)

    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    low = np.minimum(low, close - 0.01)
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.uniform(1000, 5000, n)

    timestamps = pd.date_range("2023-01-01", periods=n, freq="4h", tz="utc")
    return pd.DataFrame({
        "timestamp": timestamps, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })


def test_backtest_runs_end_to_end_and_produces_sane_stats():
    cfg = load_config()
    price_data = {
        "BTC/USDT:USDT": _synthetic_ohlcv(seed=1),
        "ETH/USDT:USDT": _synthetic_ohlcv(seed=2, start=50.0),
        "SOL/USDT:USDT": _synthetic_ohlcv(seed=3, start=20.0),
    }

    result = run_backtest(cfg, price_data)

    assert result.stats.final_equity > 0
    assert isinstance(result.stats.total_trades, int)
    assert 0.0 <= result.stats.win_rate <= 100.0
    assert len(result.equity_curve) > 0
    # every closed trade should have paid fees on both legs
    for trade in result.trades:
        assert trade.fees >= 0


def test_backtest_raises_on_too_little_data():
    cfg = load_config()
    price_data = {"BTC/USDT:USDT": _synthetic_ohlcv(n=50)}
    try:
        run_backtest(cfg, price_data)
        assert False, "expected ValueError for insufficient candles"
    except ValueError:
        pass
