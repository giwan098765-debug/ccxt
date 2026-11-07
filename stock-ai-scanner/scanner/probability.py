"""성공확률·기대수익률 산출.

'감'이 아니라 데이터로 계산한다:
1) 셋업 백테스트 — 지금과 같은 셋업이 과거에 발생했던 모든 날을 찾아,
   이후 N일 동안 '목표가 도달 전 손절가 이탈' 여부를 시뮬레이션 → 성공확률.
   종목 자체 표본이 적으면 스캔 유니버스 전체 표본(풀)으로 베이지안 축소(shrinkage).
2) 급등 AI 모델 — 유니버스 전체 과거 데이터로 학습한 로지스틱 회귀가
   '향후 5거래일 내 +X% 이상 급등' 확률을 예측 (시간순 검증 AUC 제공).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def simulate_trades(d: pd.DataFrame, mask: pd.Series, target: float, stop: float, horizon: int) -> dict:
    """mask 가 True 인 날 종가 진입 → horizon 일 내 목표(+target) / 손절(-stop) 선도달 시뮬레이션.
    같은 봉에서 둘 다 닿으면 보수적으로 실패 처리."""
    hi, lo, cl = d["High"].to_numpy(), d["Low"].to_numpy(), d["Close"].to_numpy()
    idx = np.flatnonzero(mask.to_numpy())
    idx = idx[idx < len(cl) - horizon]
    wins = losses = timeouts = 0
    max_gains, end_rets = [], []
    last_entry = -10**9
    for i in idx:
        if i - last_entry < 3:  # 연속 신호 중복 카운트 방지
            continue
        last_entry = i
        e = cl[i]
        up, dn = e * (1 + target), e * (1 - stop)
        result = 0
        for j in range(i + 1, i + horizon + 1):
            if lo[j] <= dn:
                result = -1
                break
            if hi[j] >= up:
                result = 1
                break
        wins += result == 1
        losses += result == -1
        timeouts += result == 0
        max_gains.append(hi[i + 1:i + horizon + 1].max() / e - 1)
        end_rets.append(cl[i + horizon] / e - 1)
    n = wins + losses + timeouts
    return {
        "n": int(n), "wins": int(wins), "losses": int(losses), "timeouts": int(timeouts),
        "max_gains": max_gains, "end_rets": end_rets,
    }


def shrink(wins: int, n: int, prior_p: float, prior_n: float = 20.0) -> float:
    """베이지안 축소: 표본이 적을수록 사전확률(유니버스 평균)에 가깝게."""
    return (wins + prior_p * prior_n) / (n + prior_n)


def forward_stats(d: pd.DataFrame, horizons=(5, 10, 20)) -> pd.DataFrame:
    """각 날짜 기준 향후 N일 최대 상승률/종가 수익률 (학습 라벨·통계용)."""
    out = pd.DataFrame(index=d.index)
    hi, cl = d["High"], d["Close"]
    for h in horizons:
        fwd_max = hi[::-1].rolling(h, min_periods=h).max()[::-1].shift(-1)
        out[f"max{h}"] = fwd_max / cl - 1
        out[f"ret{h}"] = cl.shift(-h) / cl - 1
    return out


# ───────────────────────── 급등 AI 모델 ─────────────────────────

FEATURES = [
    "rsi", "vol_ratio_log", "ret5", "ret20", "dist_ma20", "dist_ma60", "bb_pos", "bb_width_pct", "adx",
    "macd_hist_n", "obv_slope20", "atr_pct", "from_hh60", "from_hh250", "close_pos", "mfi", "ma_align",
    "upper_wick", "vol_trend", "value_log",
]


def make_features(d: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=d.index)
    c = d["Close"]
    f["rsi"] = d["rsi"] / 100
    f["vol_ratio_log"] = np.log1p(d["vol_ratio"].clip(0, 50))
    f["ret5"] = d["ret5"].clip(-0.5, 1)
    f["ret20"] = d["ret20"].clip(-0.7, 2)
    f["dist_ma20"] = (c / d["ma20"] - 1).clip(-0.5, 1)
    f["dist_ma60"] = (c / d["ma60"] - 1).clip(-0.7, 2)
    f["bb_pos"] = d["bb_pos"].clip(-1, 2)
    f["bb_width_pct"] = d["bb_width_pct"]
    f["adx"] = d["adx"] / 100
    f["macd_hist_n"] = (d["macd_hist"] / c).clip(-0.1, 0.1) * 10
    f["obv_slope20"] = d["obv_slope20"].clip(-3, 3)
    f["atr_pct"] = d["atr_pct"].clip(0, 0.3)
    f["from_hh60"] = (c / d["hh60"] - 1).clip(-0.8, 0.5)
    f["from_hh250"] = (c / d["hh250"] - 1).clip(-0.9, 0.5)
    f["close_pos"] = ((c - d["Low"]) / (d["High"] - d["Low"]).replace(0, np.nan)).fillna(0.5)
    f["mfi"] = d["mfi"] / 100
    f["ma_align"] = ((d["ma5"] > d["ma20"]).astype(float) + (d["ma20"] > d["ma60"]).astype(float) + (d["ma60"] > d["ma120"]).astype(float)) / 3
    f["upper_wick"] = ((d["High"] - np.maximum(d["Open"], c)) / c).clip(0, 0.3)
    f["vol_trend"] = np.log((d["Volume"].rolling(5).mean() / d["vol_ma20"].replace(0, np.nan)).clip(0.05, 20))
    f["value_log"] = np.log10((d["value"].rolling(20).mean()).clip(lower=1))
    return f


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@dataclass
class SurgeModel:
    """numpy 로지스틱 회귀 (L2). 라벨: 향후 horizon 일 고가 기준 +threshold 이상 도달."""

    threshold: float = 0.15
    horizon: int = 5
    l2: float = 1e-3
    w: np.ndarray | None = None
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    metrics: dict = field(default_factory=dict)

    def _prep(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu) / self.sd
        return np.hstack([Z, np.ones((len(Z), 1))])

    def fit(self, X: np.ndarray, y: np.ndarray, dates: np.ndarray | None = None, epochs: int = 300, lr: float = 0.5):
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[ok], y[ok]
        dates = dates[ok] if dates is not None else np.arange(len(y))
        if len(y) < 200 or y.sum() < 10:
            self.metrics = {"trained": False, "reason": "학습 표본 부족", "n": int(len(y)), "positives": int(y.sum())}
            return self
        # 시간순 분할: 과거 80% 학습, 최근 20% 검증 (미래 정보 누설 방지)
        cut = np.quantile(dates.astype("datetime64[ns]").astype(np.int64) if dates.dtype.kind == "M" else dates, 0.8)
        dnum = dates.astype("datetime64[ns]").astype(np.int64) if dates.dtype.kind == "M" else dates
        tr, te = dnum <= cut, dnum > cut
        self.mu = X[tr].mean(axis=0)
        self.sd = X[tr].std(axis=0) + 1e-9
        A = self._prep(X[tr])
        yt = y[tr]
        # 클래스 불균형 보정 없이 학습 → 확률이 실제 빈도에 맞게 보정됨
        w = np.zeros(A.shape[1])
        w[-1] = np.log(yt.mean() / (1 - yt.mean()))
        for _ in range(epochs):
            p = 1 / (1 + np.exp(-A @ w))
            g = A.T @ (p - yt) / len(yt) + self.l2 * np.r_[w[:-1], 0]
            w -= lr * g
        self.w = w
        pt = self.predict_proba(X[te]) if te.sum() else np.array([])
        base = float(y.mean())
        top_dec = float("nan")
        if te.sum() > 50:
            thr = np.quantile(pt, 0.9)
            sel = pt >= thr
            top_dec = float(y[te][sel].mean()) if sel.any() else float("nan")
        self.metrics = {
            "trained": True, "n": int(len(y)), "positives": int(y.sum()), "base_rate": base,
            "auc_test": _auc(y[te], pt) if te.sum() else float("nan"),
            "top10_hit_rate_test": top_dec, "n_test": int(te.sum()),
        }
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            return np.full(len(X), np.nan)
        X = np.where(np.isfinite(X), X, self.mu)
        return 1 / (1 + np.exp(-self._prep(X) @ self.w))

    def contributions(self, x: np.ndarray) -> list[tuple[str, float]]:
        """개별 종목 예측에 대한 피처별 기여도 (로짓 단위) — '왜 급등 후보인가' 설명용."""
        if self.w is None:
            return []
        x = np.where(np.isfinite(x), x, self.mu)
        z = (x - self.mu) / self.sd
        contrib = z * self.w[:-1]
        return sorted(zip(FEATURES, contrib.tolist()), key=lambda t: -abs(t[1]))


FEATURE_KR = {
    "rsi": "RSI", "vol_ratio_log": "거래량 배수", "ret5": "5일 수익률", "ret20": "20일 수익률",
    "dist_ma20": "20일선 이격", "dist_ma60": "60일선 이격", "bb_pos": "볼린저 위치", "bb_width_pct": "밴드폭(수렴도)",
    "adx": "ADX 추세강도", "macd_hist_n": "MACD 히스토그램", "obv_slope20": "OBV 매집 기울기", "atr_pct": "변동성(ATR)",
    "from_hh60": "60일 고점 대비", "from_hh250": "52주 고점 대비", "close_pos": "당일 종가 위치", "mfi": "자금흐름(MFI)",
    "ma_align": "이평 정배열도", "upper_wick": "윗꼬리", "vol_trend": "최근 5일 거래량 추세", "value_log": "거래대금 규모",
}
