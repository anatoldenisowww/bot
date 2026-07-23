import numpy as np
import pandas as pd

from config import load_config
from optimize import (
    walk_forward_optimize, most_common_params, _grid_combos, _param_grid_for,
    _apply_params, _CROSS_SECTIONAL_GRID,
)


def _synthetic_ohlcv(n, seed, drift=0.1, start=100.0) -> pd.DataFrame:
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

    timestamps = pd.date_range("2022-01-01", periods=n, freq="4h", tz="utc")
    return pd.DataFrame({
        "timestamp": timestamps, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })


def test_grid_combos_matches_expected_size_per_mode():
    for mode in ("ensemble", "mean_reversion_scalp", "cross_sectional_momentum"):
        grid = _param_grid_for(mode)
        combos = list(_grid_combos(mode))
        expected = 1
        for values in grid.values():
            expected *= len(values)
        assert len(combos) == expected
        assert all(set(c.keys()) == set(grid.keys()) for c in combos)


def test_cross_sectional_params_apply_to_strategy_config():
    cfg = load_config()
    cfg.strategy.mode = "cross_sectional_momentum"
    params = {"momentum_lookback_bars": 60, "min_abs_momentum_score": 1.5, "atr_stop_multiplier": 3.0}
    trial = _apply_params(cfg, params)
    assert trial.strategy.momentum_lookback_bars == 60
    assert trial.strategy.min_abs_momentum_score == 1.5
    assert trial.risk.atr_stop_multiplier == 3.0
    # the searched keys must all be real grid keys for this mode
    assert set(params.keys()) == set(_CROSS_SECTIONAL_GRID.keys())


def test_walk_forward_optimize_runs_and_produces_folds():
    cfg = load_config()
    cfg.strategy.mode = "ensemble"  # pin the mode so param keys are deterministic regardless of settings.yaml default
    # 4h bars: 6/day. Small train/test windows keep this test fast.
    price_data = {
        "BTC/USDT:USDT": _synthetic_ohlcv(n=1400, seed=1),
        "ETH/USDT:USDT": _synthetic_ohlcv(n=1400, seed=2, start=50.0),
    }

    result = walk_forward_optimize(cfg, price_data, train_days=60, test_days=20, step_days=20)

    assert len(result.folds) >= 1
    for fold in result.folds:
        assert set(fold.best_params.keys()) == {"adx_trend_threshold", "atr_stop_multiplier", "take_profit_r_multiple"}
        assert fold.test_start < fold.test_end

    assert result.chained_stats.final_equity > 0
    recommended = most_common_params(result)
    assert set(recommended.keys()) == {"adx_trend_threshold", "atr_stop_multiplier", "take_profit_r_multiple"}


def test_walk_forward_raises_when_no_fold_fits():
    cfg = load_config()
    price_data = {"BTC/USDT:USDT": _synthetic_ohlcv(n=300, seed=1)}
    try:
        walk_forward_optimize(cfg, price_data, train_days=300, test_days=100)
        assert False, "expected ValueError when history is too short for a single fold"
    except ValueError:
        pass
