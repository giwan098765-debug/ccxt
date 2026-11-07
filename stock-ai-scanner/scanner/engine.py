"""스캔 엔진 — 유니버스 전체를 분석해 유형별 추천 / 이번주 급등주 / AI 급등 후보를 만든다."""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .candles import PATTERN_INFO, detect_candles, recent_patterns
from .data import StockInfo, get_provider
from .fundamentals import score_fundamentals
from .indicators import add_indicators
from .patterns import SETUPS, chart_shape, setup_signals
from .probability import FEATURE_KR, FEATURES, SurgeModel, forward_stats, make_features, shrink, simulate_trades
from .volume_profile import volume_profile

CATEGORIES = {
    "surge_week": "🚀 이번주 급등주",
    "ai_surge": "🤖 AI 급등 후보",
    "breakout": "📈 돌파형",
    "rebound": "↩️ 반등형",
    "trend": "📊 추세추종형",
    "pullback": "🌀 눌림목형",
    "squeeze": "🎯 수렴(스퀴즈)형",
    "volume": "🔋 거래량 급증형",
    "value": "💎 재무우량형",
}


@dataclass
class ScanConfig:
    market: str = "KRX"
    limit: int = 300
    surge_min: float = 0.15          # 이번주 급등주 최소 상승률
    ai_threshold: float = 0.15       # AI 모델 라벨: 5일 내 +15% 도달
    horizon: int = 10                # 성공확률 판정 기간(거래일)
    min_value: float = 3e8           # 최소 20일 평균 거래대금 (원/달러) — 유동성 필터
    workers: int = 8
    fundamentals_top: int = 60
    category_top: int = 20
    demo: bool = False


@dataclass
class ScanResult:
    config: ScanConfig
    summary: pd.DataFrame
    details: dict
    categories: dict
    model_metrics: dict
    continuation: dict
    provider_name: str
    errors: list = field(default_factory=list)


def _f(x, default=float("nan")) -> float:
    try:
        x = float(x)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def week_return(d: pd.DataFrame) -> dict:
    """이번 주(마지막 봉이 속한 주) 수익률 = 직전 주 마지막 종가 대비."""
    last = d.index[-1]
    monday = (last - pd.Timedelta(days=last.weekday())).normalize()
    prev = d[d.index < monday]
    week = d[d.index >= monday]
    if len(prev) == 0 or len(week) == 0:
        return {"ret": float("nan")}
    base = float(prev["Close"].iloc[-1])
    rets = week["Close"] / week["Close"].shift(1).fillna(base) - 1
    return {
        "ret": float(week["Close"].iloc[-1] / base - 1),
        "max_ret": float(week["High"].max() / base - 1),
        "base_date": prev.index[-1], "base_close": base,
        "best_day": rets.idxmax(), "best_day_ret": float(rets.max()),
        "limit_up_days": int((rets >= 0.295).sum()),
        "days": len(week),
    }


class Scanner:
    def __init__(self, config: ScanConfig, provider=None):
        self.cfg = config
        self.provider = provider or get_provider(config.demo)

    # ------------------------------------------------------------------ helpers
    def _load(self, s: StockInfo):
        df = self.provider.ohlcv(s)
        if df is None or len(df) < 150:
            return None
        if _f(df["Close"].iloc[-20:].mul(df["Volume"].iloc[-20:]).mean(), 0) < self.cfg.min_value:
            return None
        d = add_indicators(df)
        return {
            "info": s, "d": d, "sig": setup_signals(d), "candles": detect_candles(d),
            "feat": make_features(d), "fwd": forward_stats(d, (5, 10, 20)),
        }

    @staticmethod
    def _target_stop(d: pd.DataFrame, vp: dict) -> tuple[float, float, float | None]:
        last = d.iloc[-1]
        c, atr = float(last["Close"]), _f(last["atr"], float(last["Close"]) * 0.03)
        sup = vp.get("support") if vp else None
        stop_px = c - 2 * atr
        if sup and c * 0.85 < sup < c:
            stop_px = max(stop_px, sup * 0.99)
        stop = float(np.clip(1 - stop_px / c, 0.03, 0.15))
        res = vp.get("resistance") if vp else None
        tgt = 3 * atr / c
        if res and res > c:
            tgt = max(tgt, res / c - 1)
        tgt = float(np.clip(max(tgt, 1.5 * stop), 0.04, 0.6))
        return tgt, stop, res

    # --------------------------------------------------------------------- run
    def run(self, progress: Callable[[float, str], None] | None = None) -> ScanResult:
        cfg = self.cfg
        prog = progress or (lambda p, m: None)
        universe = self.provider.universe(cfg.market, cfg.limit)
        prog(0.02, f"유니버스 {len(universe)}종목 시세 수집 중…")
        loaded, errors = [], []
        with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
            futs = {ex.submit(self._load, s): s for s in universe}
            for k, fu in enumerate(as_completed(futs)):
                s = futs[fu]
                try:
                    r = fu.result()
                    if r:
                        loaded.append(r)
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{s.code} {s.name}: {e}")
                if k % 10 == 0:
                    prog(0.02 + 0.5 * (k + 1) / max(len(universe), 1), f"시세·지표 계산 {k+1}/{len(universe)}")
        if not loaded:
            raise RuntimeError("분석 가능한 종목이 없습니다. 네트워크/데이터 소스를 확인하세요. " + "; ".join(errors[:3]))

        # ── 1) 급등 AI 모델 학습 (유니버스 풀 데이터) ──
        prog(0.55, "AI 급등 예측 모델 학습 중…")
        Xs, ys, ds = [], [], []
        for r in loaded:
            f, fw = r["feat"], r["fwd"]
            valid = fw["max5"].notna() & (np.arange(len(f)) >= 130)
            sub = f[valid].iloc[::2]
            Xs.append(sub[FEATURES].to_numpy(float))
            ys.append((fw.loc[sub.index, "max5"] >= cfg.ai_threshold).to_numpy(float))
            ds.append(sub.index.to_numpy())
        model = SurgeModel(threshold=cfg.ai_threshold, horizon=5)
        model.fit(np.vstack(Xs), np.concatenate(ys), np.concatenate(ds))

        # ── 2) 급등 이후 흐름 통계 (풀) ──
        cont_n, cont_up10, cont_ret = 0, 0, []
        for r in loaded:
            d, fw = r["d"], r["fwd"]
            m = (d["ret5"] >= cfg.surge_min) & fw["ret10"].notna()
            m = m & ~m.shift(1, fill_value=False)
            cont_n += int(m.sum())
            cont_up10 += int((fw.loc[m, "max10"] >= 0.10).sum())
            cont_ret += fw.loc[m, "ret10"].tolist()
        continuation = {
            "n": cont_n,
            "p_more10": cont_up10 / cont_n if cont_n else float("nan"),
            "median_ret10": float(np.median(cont_ret)) if cont_ret else float("nan"),
            "p_positive10": float(np.mean(np.array(cont_ret) > 0)) if cont_ret else float("nan"),
        }

        # ── 3) 종목별 목표/손절 + 셋업 백테스트 ──
        prog(0.62, "셋업별 과거 성공확률 백테스트 중…")
        pooled = {k: [0, 0] for k in SETUPS}
        for r in loaded:
            d = r["d"]
            r["vp"] = volume_profile(d)
            r["target"], r["stop"], r["resistance"] = self._target_stop(d, r["vp"])
            r["sims"] = {}
            for key in SETUPS:
                sim = simulate_trades(d, r["sig"][key], r["target"], r["stop"], cfg.horizon)
                r["sims"][key] = sim
                pooled[key][0] += sim["wins"]
                pooled[key][1] += sim["n"]
            base = simulate_trades(d, pd.Series(True, index=d.index), r["target"], r["stop"], cfg.horizon)
            r["base_sim"] = base
        priors = {k: (w / n if n else 0.3) for k, (w, n) in pooled.items()}

        # ── 4) 종목별 종합 분석 ──
        prog(0.72, "차트·캔들·매물대·보조지표 종합 분석 중…")
        details, rows = {}, []
        for r in loaded:
            a = self._analyze(r, model, priors, continuation)
            details[a["code"]] = a
            rows.append(a["row"])
        summary = pd.DataFrame(rows).set_index("code")

        # ── 5) 상위 후보 + 급등주 재무 분석 ──
        prog(0.82, "재무제표 분석 중…")
        need = set(summary.sort_values("composite", ascending=False).head(cfg.fundamentals_top).index)
        need |= set(summary[summary["week_ret"] >= cfg.surge_min].index)
        def _fund(code):
            a = details[code]
            return code, self.provider.fundamentals(a["info"])
        with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
            for fu in as_completed([ex.submit(_fund, c) for c in need]):
                try:
                    code, fdat = fu.result()
                except Exception as e:  # noqa: BLE001
                    errors.append(str(e))
                    continue
                fs = score_fundamentals(fdat)
                details[code]["fundamentals"] = fdat
                details[code]["fund_score"] = fs
                summary.loc[code, "value_score"] = fs["score"] if fs["score"] is not None else np.nan
        summary["value_score"] = summary.get("value_score", pd.Series(np.nan, index=summary.index))
        vs = summary["value_score"].fillna(50)
        summary["composite"] = (summary["composite"] * 0.85 + vs * 0.15).round(1)
        for code in summary.index:
            details[code]["row"]["composite"] = float(summary.loc[code, "composite"])
            details[code]["row"]["value_score"] = _f(summary.loc[code, "value_score"])
            if code in need:
                details[code]["surge_reasons"] = self._explain_surge(details[code]) if details[code]["week"].get("ret", 0) >= cfg.surge_min else []

        # ── 6) 유형별 분류 ──
        prog(0.95, "유형별 분류 중…")
        cats = {}
        top = cfg.category_top
        cats["surge_week"] = summary[summary["week_ret"] >= cfg.surge_min].sort_values("week_ret", ascending=False)
        cats["ai_surge"] = summary[summary["ai_lift"] >= 1.5].sort_values("ai_prob", ascending=False).head(top)
        for key in ("breakout", "rebound", "trend", "pullback", "squeeze", "volume"):
            col = f"s_{key}"
            cats[key] = summary[summary[col] >= 60].sort_values([col, "composite"], ascending=False).head(top)
        cats["value"] = summary[summary["value_score"] >= 70].sort_values("value_score", ascending=False).head(top)
        prog(1.0, "완료")
        return ScanResult(cfg, summary.sort_values("composite", ascending=False), details, cats, model.metrics,
                          continuation, self.provider.name, errors)

    # --------------------------------------------------------------- analysis
    def _analyze(self, r: dict, model: SurgeModel, priors: dict, continuation: dict) -> dict:
        cfg = self.cfg
        s: StockInfo = r["info"]
        d, sig = r["d"], r["sig"]
        last = d.iloc[-1]
        c = float(last["Close"])
        vp = r["vp"]
        shape = chart_shape(d)
        cpat = recent_patterns(r["candles"], 3)
        x_now = r["feat"][FEATURES].iloc[-1].to_numpy(float)
        p_ai = float(model.predict_proba(x_now[None, :])[0]) if model.w is not None else float("nan")
        base_rate = model.metrics.get("base_rate", float("nan"))
        lift = p_ai / base_rate if base_rate and base_rate > 0 and math.isfinite(p_ai) else float("nan")
        contrib = [(FEATURE_KR[k], v) for k, v in model.contributions(x_now)[:6]]

        # 셋업 점수(0~100)
        recent = sig.iloc[-3:].any()
        today = sig.iloc[-1]
        rsi, vr, adx = _f(last["rsi"], 50), _f(last["vol_ratio"], 1), _f(last["adx"], 0)
        bull_c = sum(1 for p in cpat if p["bias"] > 0)
        bear_c = sum(1 for p in cpat if p["bias"] < 0)
        overhead = vp.get("overhead_supply", 0.5) if vp else 0.5
        hh60p = _f(d["hh60"].iloc[-2], c)
        sc = {}
        sc["breakout"] = (70 if today["breakout"] else 55 if recent["breakout"] else 40 * max(0, 1 - (hh60p / c - 1) / 0.05) if c >= hh60p * 0.95 else 0)
        sc["breakout"] += 15 * min(vr / 5, 1) + 15 * (1 - overhead) if sc["breakout"] else 0
        reb = 0.0
        if recent["rebound"]:
            reb = 60
        elif _f(d["rsi"].iloc[-10:].min(), 50) < 35 and rsi > _f(d["rsi"].iloc[-2], rsi):
            reb = 45
        if reb:
            reb += 10 * min(bull_c, 2)
            if vp and vp.get("support") and abs(c / vp["support"] - 1) < 0.05:
                reb += 10
            if shape.get("double_bottom"):
                reb += 10
        sc["rebound"] = reb
        tr = 0.0
        if today["trend"]:
            tr = 60 + min(adx - 22, 25) + (10 if shape.get("ichimoku", "").startswith("구름대 위") else 0)
        sc["trend"] = tr
        pb = 0.0
        if recent["pullback"]:
            pb = 65 + 15 * max(0, 1 - abs(c / _f(last["ma20"], c) - 1) / 0.03) + (10 if bull_c else 0)
        sc["pullback"] = pb
        sq = 0.0
        bwp = _f(last["bb_width_pct"], 1)
        if c > _f(last["ma60"], c * 2) and bwp < 0.25:
            sq = 50 + 50 * (0.25 - bwp) / 0.25
        sc["squeeze"] = sq
        sc["volume"] = (55 + 45 * min((vr - 3) / 5, 1)) if today["volume"] else (45 if recent["volume"] else 0)
        sc = {k: float(np.clip(v, 0, 100)) for k, v in sc.items()}

        # 성공확률: 현재 활성(최근 3봉) 셋업 중 가장 좋은 것, 없으면 전체 기준선
        active = [k for k in SETUPS if recent[k]]
        probs = {}
        for k in SETUPS:
            sim = r["sims"][k]
            p = shrink(sim["wins"], sim["n"], priors[k])
            probs[k] = {"p": p, "n": sim["n"], "wins": sim["wins"], "losses": sim["losses"],
                        "median_max_gain": float(np.median(sim["max_gains"])) if sim["max_gains"] else float("nan"),
                        "median_end_ret": float(np.median(sim["end_rets"])) if sim["end_rets"] else float("nan")}
        base = r["base_sim"]
        base_p = base["wins"] / base["n"] if base["n"] else float("nan")
        if active:
            best = max(active, key=lambda k: probs[k]["p"])
            succ = probs[best]["p"]
            exp_gain = probs[best]["median_max_gain"]
        else:
            best = None
            succ = base_p
            exp_gain = float(np.median(base["max_gains"])) if base["max_gains"] else float("nan")
        # 기대값 (EV): 성공 시 +target, 실패 시 -stop (타임아웃 포함 보수적)
        ev = succ * r["target"] - (1 - succ) * r["stop"] if math.isfinite(succ) else float("nan")

        wk = week_return(d)
        tech_best = max(sc.values()) if sc else 0
        ai_score = float(np.clip((lift if math.isfinite(lift) else 1) / 4, 0, 1) * 100)
        composite = (0.30 * ai_score + 0.30 * (succ if math.isfinite(succ) else 0.3) * 100
                     + 0.25 * tech_best + 0.15 * float(np.clip((ev if math.isfinite(ev) else 0) / 0.08 + 0.5, 0, 1) * 100))

        row = {
            "code": s.code, "name": s.name, "market": s.market, "close": c,
            "chg1": _f(last["ret1"]), "week_ret": _f(wk.get("ret")), "ret20": _f(last["ret20"]),
            "vol_ratio": vr, "rsi": rsi, "ai_prob": p_ai, "ai_lift": lift,
            "success_prob": succ, "best_setup": SETUPS[best][0] if best else "-",
            "target_pct": r["target"], "stop_pct": r["stop"], "target_px": c * (1 + r["target"]),
            "stop_px": c * (1 - r["stop"]), "exp_max_gain": exp_gain, "ev": ev,
            "limit_up": wk.get("limit_up_days", 0), "composite": composite,
            **{f"s_{k}": v for k, v in sc.items()},
            "candles": ", ".join(p["name"] for p in cpat if p["days_ago"] == 0),
        }
        a = {
            "code": s.code, "info": s, "d": d, "row": row, "vp": vp, "shape": shape, "candle_patterns": cpat,
            "setup_scores": sc, "active_setups": active, "probs": probs, "base_success": base_p,
            "ai": {"prob": p_ai, "base_rate": base_rate, "lift": lift, "contrib": contrib},
            "week": wk, "continuation": continuation, "target": r["target"], "stop": r["stop"],
            "resistance": r["resistance"], "fundamentals": {}, "fund_score": None, "surge_reasons": [],
            "model": model,
        }
        a["summary_text"] = build_summary(a)
        return a

    def _explain_surge(self, a: dict) -> list[str]:
        """이번 주 급등의 근거를 데이터로 분해."""
        d, wk = a["d"], a["week"]
        out = []
        if not wk.get("best_day") is None:
            bd = wk["best_day"]
            row = d.loc[bd]
            out.append(f"📅 {bd:%m/%d} 하루 {wk['best_day_ret']*100:+.1f}% — 거래량 평균의 {_f(row['vol_ratio'], 0):.1f}배")
            if wk.get("limit_up_days"):
                out.append(f"🔥 이번 주 상한가(+29.5%↑) {wk['limit_up_days']}회")
            pos = d.index.get_loc(bd)
            if pos > 0:
                prev_hh60 = _f(d["hh60"].iloc[pos - 1])
                prev_hh250 = _f(d["hh250"].iloc[pos - 1])
                if math.isfinite(prev_hh250) and row["Close"] > prev_hh250:
                    out.append("🏔 52주 신고가 돌파 — 위에 물린 매물이 없는 구간 진입")
                elif math.isfinite(prev_hh60) and row["Close"] > prev_hh60:
                    out.append("📈 60일 고점(박스권 상단) 돌파")
            body = abs(row["Close"] - row["Open"]) / row["Close"]
            if row["Close"] > row["Open"] and body > 0.08:
                out.append(f"🕯 몸통 {body*100:.0f}% 장대양봉 — 강한 매수 주체 유입")
        vp = a["vp"]
        if vp:
            if vp["overhead_supply"] < 0.1:
                out.append(f"🧱 현재가 위 매물 비중 {vp['overhead_supply']*100:.0f}% — 매물대 상단을 뚫어 저항이 거의 없음")
            elif vp.get("resistance"):
                out.append(f"🧱 다음 매물 저항대 {vp['resistance']:,.0f} (현재가 대비 {vp['resistance']/vp['price']*100-100:+.1f}%)")
        base_date = wk.get("base_date")
        if base_date is not None:
            pos = d.index.get_loc(base_date)
            pre = d.iloc[max(0, pos - 10):pos + 1]
            if (pre["bb_width_pct"] < 0.2).any():
                out.append("🎯 급등 직전 볼린저밴드 극단 수렴 — 에너지 응축 후 분출 패턴")
            if _f(pre["obv_slope20"].iloc[-1], 0) > 0.3:
                out.append("🧲 급등 전 OBV 상승 — 가격 횡보 중 물량 매집 흔적")
            if (pre["rsi"] < 35).any():
                out.append("↩️ 과매도 구간에서 출발한 반등형 급등")
            m: SurgeModel = a["model"]
            if m.w is not None:
                feat = make_features(d).loc[[base_date], FEATURES].to_numpy(float)
                p0 = float(m.predict_proba(feat)[0])
                br = m.metrics.get("base_rate", float("nan"))
                out.append(f"🤖 급등 시작 전날 AI 예측 급등확률 {p0*100:.1f}% (평균 {br*100:.1f}% 대비 {p0/br:.1f}배)")
        fs = a.get("fund_score")
        if fs and fs.get("points"):
            out.append("💎 재무: " + ", ".join(fs["points"][:3]))
        if fs and fs.get("warnings"):
            out.append("⚠️ 재무 리스크: " + ", ".join(fs["warnings"][:2]))
        c = a["continuation"]
        if c.get("n"):
            out.append(f"📊 과거 유니버스에서 5일 +{self.cfg.surge_min*100:.0f}%↑ 급등 후 10일 내 추가 +10% 도달 확률 "
                       f"{c['p_more10']*100:.0f}% (n={c['n']}), 10일 뒤 중앙 수익률 {c['median_ret10']*100:+.1f}%")
        return out


def build_summary(a: dict) -> str:
    """규칙 기반 한 줄 요약 (AI 리포트 없이도 이해 가능한 근거)."""
    r, sh = a["row"], a["shape"]
    parts = []
    if a["active_setups"]:
        parts.append("셋업: " + "·".join(SETUPS[k][0] for k in a["active_setups"]))
    if sh.get("alignment"):
        parts.append(sh["alignment"])
    bulls = [p["name"] for p in a["candle_patterns"] if p["bias"] > 0]
    if bulls:
        parts.append("캔들: " + "·".join(dict.fromkeys(bulls)))
    if math.isfinite(r["ai_lift"]):
        parts.append(f"AI 급등확률 {r['ai_prob']*100:.1f}% (평균 {r['ai_lift']:.1f}배)")
    if math.isfinite(r["success_prob"]):
        parts.append(f"목표 +{r['target_pct']*100:.1f}% 성공확률 {r['success_prob']*100:.0f}%")
    return " | ".join(parts)


__all__ = ["Scanner", "ScanConfig", "ScanResult", "CATEGORIES", "PATTERN_INFO", "week_return"]
