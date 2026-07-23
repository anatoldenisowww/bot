from universe import is_eligible_perp_market, rank_by_liquidity, select_universe


def _market(symbol, **overrides):
    base = {
        "symbol": symbol, "swap": True, "linear": True, "active": True,
        "quote": "USDT", "settle": "USDT", "expiry": None, "info": {},
    }
    base.update(overrides)
    return base


def test_is_eligible_rejects_non_perp_and_rwa():
    assert is_eligible_perp_market(_market("BTC/USDT:USDT"))
    assert not is_eligible_perp_market(_market("BTC/USDT:USDT", swap=False))
    assert not is_eligible_perp_market(_market("BTC/USDT:USDT", active=False))
    assert not is_eligible_perp_market(_market("BTC/USDT:USDT", quote="USDC"))
    assert not is_eligible_perp_market(_market("BTC/USDT:USDT", expiry=1700000000000))
    # tokenized real-world assets (gold, stocks) are tagged isRwa in info
    assert not is_eligible_perp_market(_market("XAU/USDT:USDT", info={"isRwa": "YES"}))


def test_rank_by_liquidity_orders_descending_and_filters():
    markets = {
        "BTC/USDT:USDT": _market("BTC/USDT:USDT"),
        "ETH/USDT:USDT": _market("ETH/USDT:USDT"),
        "XAU/USDT:USDT": _market("XAU/USDT:USDT", info={"isRwa": "YES"}),
        "OLD/USDT:USDT": _market("OLD/USDT:USDT", active=False),
    }
    tickers = {
        "BTC/USDT:USDT": {"quoteVolume": 1_000_000},
        "ETH/USDT:USDT": {"quoteVolume": 5_000_000},
        "XAU/USDT:USDT": {"quoteVolume": 9_000_000},  # highest volume but RWA -> excluded
        "OLD/USDT:USDT": {"quoteVolume": 8_000_000},  # inactive -> excluded
        "GHOST/USDT:USDT": {"quoteVolume": 7_000_000},  # in tickers but not markets -> excluded
    }
    ranked = rank_by_liquidity(tickers, markets)
    assert [sym for sym, _ in ranked] == ["ETH/USDT:USDT", "BTC/USDT:USDT"]


def test_select_universe_applies_floor_and_top_n_and_exclude():
    markets = {f"C{i}/USDT:USDT": _market(f"C{i}/USDT:USDT") for i in range(5)}
    tickers = {
        "C0/USDT:USDT": {"quoteVolume": 100_000_000},
        "C1/USDT:USDT": {"quoteVolume": 50_000_000},
        "C2/USDT:USDT": {"quoteVolume": 20_000_000},
        "C3/USDT:USDT": {"quoteVolume": 1_000_000},   # below floor
        "C4/USDT:USDT": {"quoteVolume": 200_000},     # below floor
    }
    selected = select_universe(tickers, markets, top_n=2, min_quote_volume_24h=5_000_000)
    assert selected == ["C0/USDT:USDT", "C1/USDT:USDT"]

    selected_excl = select_universe(
        tickers, markets, top_n=2, min_quote_volume_24h=5_000_000, exclude=["C0/USDT:USDT"],
    )
    assert selected_excl == ["C1/USDT:USDT", "C2/USDT:USDT"]
