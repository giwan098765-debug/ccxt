import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanner.candles import detect_candles  # noqa: E402
from scanner.data import DemoDataProvider, StockInfo  # noqa: E402
from scanner.engine import ScanConfig, Scanner, week_return  # noqa: E402
from scanner.fundamentals import score_fundamentals  # noqa: E402
from scanner.indicators import add_indicators  # noqa: E402
from scanner.probability import SurgeModel, forward_stats, shrink, simulate_trades  # noqa: E402
from scanner.volume_profile import volume_profile  # noqa: E402


def _df(closes, highs=None, lows=None, opens=None, vol=None, start="2026-01-05"):
    n = len(closes)
    idx = pd.bdate_range(start, periods=n)
    c = np.asarray(closes, float)
    return pd.DataFrame({
        "Open": np.asarray(opens, float) if opens is not None else c,
        "High": np.asarray(highs, float) if highs is not None else c * 1.01,
        "Low": np.asarray(lows, float) if lows is not None else c * 0.99,
        "Close": c, "Volume": np.asarray(vol, float) if vol is not None else np.full(n, 1000.0),
    }, index=idx)


@pytest.fixture(scope="module")
def demo_df():
    return DemoDataProvider().ohlcv(StockInfo("005930", "삼성전자", "KOSPI"))


def test_indicator_ranges(demo_df):
    d = add_indicators(demo_df)
    rsi = d["rsi"].dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()
    assert (d["bb_up"].dropna() >= d["bb_low"].dropna()).all()
    assert (d["atr"].dropna() > 0).all()
    assert d["ma20"].iloc[-1] == pytest.approx(demo_df["Close"].iloc[-20:].mean())


def test_rsi_extremes():
    up = add_indicators(_df(np.linspace(100, 200, 60)))
    dn = add_indicators(_df(np.linspace(200, 100, 60)))
    assert up["rsi"].iloc[-1] > 95
    assert dn["rsi"].iloc[-1] < 5


def test_simulate_trades_win_and_loss():
    # 진입 후 바로 +10% 고가 → 성공
    closes = [100] * 5 + [100, 105, 111, 111, 111]
    highs = [101] * 5 + [101, 106, 112, 112, 112]
    lows = [99] * 5 + [99, 104, 110, 110, 110]
    d = _df(closes, highs, lows)
    mask = pd.Series(False, index=d.index)
    mask.iloc[5] = True
    r = simulate_trades(d, mask, target=0.10, stop=0.05, horizon=3)
    assert (r["n"], r["wins"], r["losses"]) == (1, 1, 0)
    # 손절이 먼저 → 실패
    lows2 = list(lows)
    lows2[6] = 94
    r2 = simulate_trades(_df(closes, highs, lows2), mask, target=0.10, stop=0.05, horizon=3)
    assert r2["losses"] == 1 and r2["wins"] == 0


def test_shrink_moves_toward_prior():
    assert shrink(1, 1, 0.3) < 0.5
    assert shrink(100, 100, 0.3) > 0.85


def test_forward_stats_excludes_today():
    d = _df([10, 10, 10, 20], highs=[30, 10, 10, 20])
    fw = forward_stats(d, (2,))
    assert fw["max2"].iloc[0] == pytest.approx(0.0)  # 다음 2일 고가 최대 10
    assert fw["max2"].iloc[1] == pytest.approx(1.0)


def test_candles_bull_engulf():
    opens = [10] * 20 + [10.5, 9.4]
    closes = [10] * 20 + [9.5, 10.8]
    d = _df(closes, highs=np.maximum(opens, closes) + 0.1, lows=np.minimum(opens, closes) - 0.1, opens=opens)
    assert detect_candles(d)["bull_engulf"].iloc[-1]


def test_volume_profile_poc():
    closes = [100] * 80 + [120] * 20
    d = _df(closes, highs=[c + 1 for c in closes], lows=[c - 1 for c in closes])
    vp = volume_profile(d, lookback=100)
    assert abs(vp["poc"] - 100) < 2
    assert vp["support"] is not None and vp["support"] < 120


def test_week_return_uses_previous_week_close():
    d = _df([100, 100, 100, 100, 100, 110, 120], start="2026-11-02")  # 월~금, 다음주 월·화
    wk = week_return(d)
    assert wk["ret"] == pytest.approx(0.2)
    assert wk["days"] == 2


def test_surge_model_learns_signal():
    rng = np.random.default_rng(0)
    from scanner.probability import FEATURES
    X = rng.normal(size=(4000, len(FEATURES)))
    y = (X[:, 1] + rng.normal(scale=0.5, size=4000) > 1.5).astype(float)
    m = SurgeModel().fit(X, y, np.arange(4000))
    assert m.metrics["trained"] and m.metrics["auc_test"] > 0.85


def test_fundamental_score():
    good = score_fundamentals({"per": 7, "pbr": 0.8, "roe": 0.2, "revenue_growth": 0.4})
    bad = score_fundamentals({"per": 80, "roe": -0.1, "debt_to_equity": 300})
    assert good["score"] > 80 > bad["score"]
    assert score_fundamentals({})["score"] is None


def test_end_to_end_demo_scan():
    res = Scanner(ScanConfig(demo=True, min_value=0, limit=40)).run()
    assert len(res.summary) > 20
    assert res.model_metrics["trained"]
    s = res.summary
    assert s["success_prob"].between(0, 1).all()
    assert (s["target_pct"] >= 1.5 * s["stop_pct"] - 1e-9).all()
    for code in res.categories["surge_week"].index:
        assert s.loc[code, "week_ret"] >= res.config.surge_min
        assert res.details[code]["surge_reasons"]


def test_ai_report_payload_is_json_serializable():
    import json
    from scanner.ai_report import build_payload
    res = Scanner(ScanConfig(demo=True, min_value=0, limit=30)).run()
    code = res.summary.index[0]
    json.dumps(build_payload(res.details[code]), ensure_ascii=False)
