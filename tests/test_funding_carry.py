import numpy as np
import pandas as pd

from funding_carry import CarryConfig, backtest_carry, screen
from funding_carry_report import render_carry_html


def _funding(rates, start="2026-01-01"):
    ts = pd.date_range(start, periods=len(rates), freq="8h", tz="utc")
    return pd.DataFrame({"timestamp": ts, "funding_rate": rates})


def test_positive_funding_produces_positive_carry():
    # steadily positive funding -> positive sustainable carry, high persistence
    rates = np.full(120, 0.0004)  # 0.04%/8h ~ 4.4%/yr
    report = backtest_carry(_funding(rates), CarryConfig(hold="always"), symbol="X/USDT:USDT")
    assert report.stats.persistence_pct == 100.0
    assert report.stats.sustainable_annual_pct > 3.0
    assert report.stats.worth_it


def test_mostly_negative_funding_is_not_worth_it():
    rates = np.full(120, -0.0003)  # you'd pay to hold -> not worth it
    report = backtest_carry(_funding(rates), CarryConfig(hold="always"), symbol="Y/USDT:USDT")
    assert report.stats.persistence_pct == 0.0
    assert not report.stats.worth_it


def test_persistence_measures_positive_fraction():
    rates = np.array([0.001] * 80 + [-0.001] * 20)  # 80% positive
    report = backtest_carry(_funding(rates), CarryConfig(hold="always"), symbol="Z/USDT:USDT")
    assert abs(report.stats.persistence_pct - 80.0) < 1e-6


def test_positive_only_pays_flip_costs():
    # alternating funding forces many flips in positive_only mode -> flip drag
    rates = np.array([0.001, -0.001] * 60)
    always = backtest_carry(_funding(rates), CarryConfig(hold="always"), symbol="A")
    flip = backtest_carry(_funding(rates), CarryConfig(hold="positive_only"), symbol="A")
    # positive_only churns and its net should be dragged below its own gross entries
    assert flip.stats.net_annual_pct < always.stats.sustainable_annual_pct + 1e-9


def test_screen_orders_worth_it_first():
    good = backtest_carry(_funding(np.full(100, 0.0005)), CarryConfig(), symbol="GOOD")
    bad = backtest_carry(_funding(np.full(100, -0.0002)), CarryConfig(), symbol="BAD")
    ordered = screen([bad, good])
    assert ordered[0].stats.symbol == "GOOD"


def test_carry_html_self_contained():
    report = backtest_carry(_funding(np.full(100, 0.0004)), CarryConfig(), symbol="X/USDT:USDT")
    doc = render_carry_html([report], CarryConfig())
    assert "<svg" in doc and "carry" in doc.lower()
    assert "http://" not in doc and "https://" not in doc and "<script" not in doc
    assert 'data-theme="dark"' in doc and doc.strip().startswith("<!doctype html>")
