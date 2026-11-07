"""AI 주식 스캐너 — Streamlit UI.

실행:  streamlit run app.py
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from scanner import ai_report
from scanner.engine import CATEGORIES, ScanConfig, Scanner
from scanner.patterns import SETUPS

st.set_page_config(page_title="AI 주식 스캐너", page_icon="📈", layout="wide")

COLS = {
    "name": "종목명", "close": "현재가", "chg1": "등락률", "week_ret": "이번주", "vol_ratio": "거래량배수",
    "ai_prob": "AI급등확률", "ai_lift": "평균대비", "success_prob": "성공확률", "best_setup": "셋업",
    "target_pct": "목표수익", "stop_pct": "손절폭", "ev": "기대값", "composite": "종합점수", "candles": "오늘 캔들",
}
PCT = ["chg1", "week_ret", "ai_prob", "success_prob", "target_pct", "stop_pct", "ev"]


def fmt_table(df: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    cols = [c for c in COLS if c in df.columns]
    if extra:
        cols = cols[:4] + [c for c in extra if c in df.columns] + cols[4:]
    t = df[cols].copy()
    for c in PCT:
        if c in t:
            t[c] = t[c] * 100
    t = t.rename(columns={**COLS, "s_breakout": "돌파점수", "s_rebound": "반등점수", "s_trend": "추세점수",
                          "s_pullback": "눌림점수", "s_squeeze": "수렴점수", "s_volume": "수급점수",
                          "value_score": "재무점수", "limit_up": "상한가"})
    return t


def show_table(df: pd.DataFrame, key: str, extra: list[str] | None = None):
    if df is None or df.empty:
        st.info("조건을 만족하는 종목이 없습니다.")
        return
    t = fmt_table(df, extra)
    cfg = {c: st.column_config.NumberColumn(c, format="%.1f%%") for c in
           ["등락률", "이번주", "AI급등확률", "성공확률", "목표수익", "손절폭", "기대값"] if c in t}
    cfg["현재가"] = st.column_config.NumberColumn("현재가", format="%,.0f")
    cfg["평균대비"] = st.column_config.NumberColumn("평균대비", format="%.1f배")
    cfg["거래량배수"] = st.column_config.NumberColumn("거래량배수", format="%.1f배")
    cfg["종합점수"] = st.column_config.ProgressColumn("종합점수", min_value=0, max_value=100, format="%.0f")
    ev = st.dataframe(t, column_config=cfg, width="stretch", key=key, on_select="rerun",
                      selection_mode="single-row")
    rows = ev.selection.rows if ev and hasattr(ev, "selection") else []
    if rows:
        st.session_state["selected"] = t.index[rows[0]]


def chart(a: dict) -> go.Figure:
    d = a["d"].iloc[-180:]
    vp = a["vp"]
    fig = make_subplots(rows=4, cols=2, shared_xaxes=True, column_widths=[0.86, 0.14], horizontal_spacing=0.01,
                        row_heights=[0.55, 0.15, 0.15, 0.15], vertical_spacing=0.02,
                        specs=[[{}, {}], [{"colspan": 2}, None], [{"colspan": 2}, None], [{"colspan": 2}, None]])
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="가격",
                                 increasing_line_color="#e03131", decreasing_line_color="#1c7ed6"), 1, 1)
    for n, col in ((5, "#f59f00"), (20, "#d6336c"), (60, "#37b24d"), (120, "#7048e8")):
        fig.add_trace(go.Scatter(x=d.index, y=d[f"ma{n}"], name=f"MA{n}", line=dict(width=1, color=col)), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["bb_up"], name="BB상단", line=dict(width=0.8, dash="dot", color="#868e96")), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["bb_low"], name="BB하단", line=dict(width=0.8, dash="dot", color="#868e96"),
                             fill="tonexty", fillcolor="rgba(134,142,150,0.06)"), 1, 1)
    c = float(d["Close"].iloc[-1])
    fig.add_hline(y=a["row"]["target_px"], line=dict(color="#e03131", dash="dash"), annotation_text=f"목표 {a['row']['target_px']:,.0f}", row=1, col=1)
    fig.add_hline(y=a["row"]["stop_px"], line=dict(color="#1c7ed6", dash="dash"), annotation_text=f"손절 {a['row']['stop_px']:,.0f}", row=1, col=1)
    if vp:
        colors = ["#e03131" if x > c else "#1c7ed6" for x in vp["centers"]]
        fig.add_trace(go.Bar(x=vp["volumes"], y=vp["centers"], orientation="h", marker_color=colors, opacity=0.55,
                             name="매물대", showlegend=False), 1, 2)
        fig.add_hline(y=vp["poc"], line=dict(color="#fab005", width=1), annotation_text="POC", row=1, col=2)
    vcol = np.where(d["Close"] >= d["Open"], "#e03131", "#1c7ed6")
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], marker_color=vcol, name="거래량", showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["rsi"], name="RSI", line=dict(color="#7048e8")), 3, 1)
    fig.add_hline(y=70, line=dict(color="#e03131", width=0.6, dash="dot"), row=3, col=1)
    fig.add_hline(y=30, line=dict(color="#1c7ed6", width=0.6, dash="dot"), row=3, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d["macd_hist"], name="MACD hist", marker_color=np.where(d["macd_hist"] >= 0, "#e03131", "#1c7ed6"), showlegend=False), 4, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["macd"], name="MACD", line=dict(color="#f59f00", width=1)), 4, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["macd_signal"], name="Signal", line=dict(color="#495057", width=1)), 4, 1)
    fig.update_layout(height=820, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.03))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(matches="y", row=1, col=2, showticklabels=False)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    return fig


def pct(x, d=1):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x*100:.{d}f}%"


def detail(res, code: str):
    a = res.details[code]
    r = a["row"]
    st.subheader(f"{r['name']} ({code} · {r['market']})")
    m = st.columns(6)
    m[0].metric("현재가", f"{r['close']:,.0f}", pct(r["chg1"]))
    m[1].metric("이번주", pct(r["week_ret"]))
    m[2].metric("AI 5일 급등확률", pct(r["ai_prob"]), f"평균의 {r['ai_lift']:.1f}배" if math.isfinite(r["ai_lift"]) else None)
    m[3].metric("성공확률", pct(r["success_prob"], 0), f"{r['best_setup']}")
    m[4].metric("목표 / 손절", f"+{r['target_pct']*100:.1f}% / -{r['stop_pct']*100:.1f}%")
    m[5].metric("기대값(EV)", pct(r["ev"]), f"종합 {r['composite']:.0f}점")
    st.caption(a["summary_text"])
    st.plotly_chart(chart(a), width="stretch")

    if a.get("surge_reasons"):
        st.markdown("#### 🚀 이번주 급등 근거")
        for line in a["surge_reasons"]:
            st.markdown(f"- {line}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🕯 캔들 · 차트 모양")
        for p in a["candle_patterns"]:
            icon = "🔴" if p["bias"] > 0 else ("🔵" if p["bias"] < 0 else "⚪")
            st.markdown(f"{icon} **{p['name']}** ({'오늘' if p['days_ago'] == 0 else str(p['days_ago']) + '일 전'}) — {p['desc']}")
        sh = a["shape"]
        for k, label in (("alignment", "이평 배열"), ("ichimoku", "일목균형표"), ("box", "박스권")):
            if sh.get(k):
                st.markdown(f"- {label}: {sh[k]}")
        if sh.get("from_52w_high") is not None and math.isfinite(sh["from_52w_high"]):
            st.markdown(f"- 52주 고점 대비: {sh['from_52w_high']*100:+.1f}%")
        if sh.get("double_bottom"):
            db = sh["double_bottom"]
            st.markdown(f"- 쌍바닥(W) 감지: 넥라인 {db['neckline']:,.0f} {'돌파 ✅' if db['breakout'] else '미돌파'} · 패턴 목표 {db['target']:,.0f}")
    with c2:
        st.markdown("#### 🧱 매물대 · 보조지표")
        vp = a["vp"]
        if vp:
            st.markdown(f"- POC(최대 매물): **{vp['poc']:,.0f}** ({'위' if vp['above_poc'] else '아래'})")
            st.markdown(f"- 가치영역 70%: {vp['va_low']:,.0f} ~ {vp['va_high']:,.0f}")
            st.markdown(f"- 위 저항대: {vp['resistance']:,.0f}" if vp.get("resistance") else "- 위 저항대: 뚜렷한 매물벽 없음")
            st.markdown(f"- 아래 지지대: {vp['support']:,.0f}" if vp.get("support") else "- 아래 지지대: 없음")
            st.markdown(f"- 현재가 위 매물 비중: {vp['overhead_supply']*100:.0f}%")
        last = a["d"].iloc[-1]
        st.markdown(f"- RSI {last['rsi']:.0f} · 스토캐스틱 {last['stoch_k']:.0f} · MFI {last['mfi']:.0f} · CCI {last['cci']:.0f}")
        st.markdown(f"- ADX {last['adx']:.0f} (+DI {last['plus_di']:.0f} / -DI {last['minus_di']:.0f}) · MACD {'상승' if last['macd_hist'] > 0 else '하락'}")
        st.markdown(f"- 밴드폭 120일 백분위 {last['bb_width_pct']*100:.0f}% · 거래량 {last['vol_ratio']:.1f}배 · OBV 기울기 {last['obv_slope20']:+.2f}")
    with c3:
        st.markdown("#### 💎 재무")
        fs = a.get("fund_score")
        fd = a.get("fundamentals") or {}
        if fs and fs.get("score") is not None:
            st.markdown(f"재무 점수 **{fs['score']}** / 100")
            for p in fs["points"]:
                st.markdown(f"- ✅ {p}")
            for w in fs["warnings"]:
                st.markdown(f"- ⚠️ {w}")
            keys = {"per": "PER", "pbr": "PBR", "roe": "ROE", "op_margin": "영업이익률", "revenue_growth": "매출성장", "debt_to_equity": "부채비율"}
            st.caption(" · ".join(f"{v} {fd[k]*100:.1f}%" if k in ("roe", "op_margin", "revenue_growth") else f"{v} {fd[k]:.2f}"
                                  for k, v in keys.items() if isinstance(fd.get(k), (int, float))))
        else:
            st.caption("재무 데이터 없음 (상위 후보·급등주만 조회)")

    st.markdown("#### 🎲 셋업별 과거 백테스트 (이 종목, 같은 목표/손절 · 10거래일)")
    pr = pd.DataFrame([
        {"셋업": SETUPS[k][0], "현재 발생": "✅" if k in a["active_setups"] else "", "과거 발생": v["n"],
         "성공": v["wins"], "손절": v["losses"], "성공확률(보정)": v["p"] * 100,
         "최대상승 중앙값": v["median_max_gain"] * 100 if math.isfinite(v["median_max_gain"]) else None,
         "10일후 중앙수익": v["median_end_ret"] * 100 if math.isfinite(v["median_end_ret"]) else None,
         "설명": SETUPS[k][1]}
        for k, v in a["probs"].items()])
    st.dataframe(pr, width="stretch", hide_index=True, column_config={
        c: st.column_config.NumberColumn(c, format="%.1f%%") for c in ("성공확률(보정)", "최대상승 중앙값", "10일후 중앙수익")})
    st.caption(f"아무 신호 없이 매일 진입했을 때 기준 성공확률: {pct(a['base_success'], 0)} — 셋업 확률이 이보다 높아야 의미가 있습니다.")

    if a["ai"]["contrib"]:
        st.markdown("#### 🤖 AI 모델이 본 핵심 요인")
        st.markdown(" · ".join(f"{n} {'▲' if v > 0 else '▼'}{abs(v):.2f}" for n, v in a["ai"]["contrib"]))

    st.markdown("#### 🧠 Claude 종합 리포트")
    if ai_report.available():
        if st.button("AI 리포트 생성", key=f"rep_{code}"):
            with st.spinner("Claude 가 분석 중…"):
                try:
                    st.session_state[f"report_{code}"] = ai_report.generate_report(a)
                except Exception as e:  # noqa: BLE001
                    st.error(f"리포트 생성 실패: {e}")
        if st.session_state.get(f"report_{code}"):
            st.markdown(st.session_state[f"report_{code}"])
    else:
        st.caption("ANTHROPIC_API_KEY 를 설정하면 위 모든 수치를 근거로 Claude 가 애널리스트 리포트를 작성합니다.")


def main():
    st.title("📈 AI 주식 스캐너")
    st.caption("차트 · 캔들 · 상단지표 · 보조지표 · 매물대 · 재무제표를 종합 분석해 유형별로 종목을 찾고, "
               "과거 데이터 백테스트로 성공확률과 기대수익을 계산합니다.")
    with st.sidebar:
        st.header("스캔 설정")
        market = st.selectbox("시장", ["KRX", "KOSPI", "KOSDAQ", "US"], help="KRX = 코스피+코스닥")
        limit = st.slider("스캔 종목 수 (시가총액 상위)", 50, 3000, 400, 50)
        surge_min = st.slider("이번주 급등주 기준 (%)", 5, 50, 15) / 100
        ai_thr = st.slider("AI 급등 정의: 5일 내 상승 (%)", 8, 30, 15) / 100
        min_value = st.number_input("최소 20일 평균 거래대금", value=3e8 if market != "US" else 5e6, step=1e8, format="%.0f")
        demo = st.toggle("데모 데이터 (오프라인 · 실제 시세 아님)", value=False)
        go_btn = st.button("🔍 스캔 시작", type="primary", width="stretch")
        st.divider()
        st.caption("⚠️ 본 앱의 확률·목표가는 과거 통계 기반 추정치이며 수익을 보장하지 않습니다. 투자 판단과 책임은 본인에게 있습니다.")

    if go_btn:
        bar = st.progress(0.0, "준비 중…")
        cfg = ScanConfig(market=market, limit=limit, surge_min=surge_min, ai_threshold=ai_thr,
                         min_value=min_value, demo=demo)
        try:
            st.session_state["result"] = Scanner(cfg).run(lambda p, m: bar.progress(min(p, 1.0), m))
            st.session_state.pop("selected", None)
        except Exception as e:  # noqa: BLE001
            st.error(f"스캔 실패: {e}")
        bar.empty()

    res = st.session_state.get("result")
    if res is None:
        st.info("왼쪽에서 시장을 고르고 **🔍 스캔 시작** 을 누르세요.")
        return
    mm = res.model_metrics
    info = st.columns(4)
    info[0].metric("분석 종목", f"{len(res.summary)}")
    info[1].metric("이번주 급등주", f"{len(res.categories['surge_week'])}")
    if mm.get("trained"):
        info[2].metric("AI 모델 검증 AUC", f"{mm['auc_test']:.2f}", help="시간순 최근 20% 구간 검증. 0.5=무작위, 1.0=완벽")
        info[3].metric("AI 상위10% 적중률", pct(mm["top10_hit_rate_test"]), f"평균 {pct(mm['base_rate'])}")
    st.caption(f"데이터: {res.provider_name}" + (" — ⚠️ 데모 모드입니다. 실제 시세가 아닙니다." if "데모" in res.provider_name else ""))

    tabs = st.tabs(list(CATEGORIES.values()) + ["🏆 전체 랭킹"])
    extra = {"surge_week": ["limit_up"], "breakout": ["s_breakout"], "rebound": ["s_rebound"], "trend": ["s_trend"],
             "pullback": ["s_pullback"], "squeeze": ["s_squeeze"], "volume": ["s_volume"], "value": ["value_score"]}
    for tab, key in zip(tabs, list(CATEGORIES.keys()) + ["all"]):
        with tab:
            if key == "surge_week":
                c = res.continuation
                st.markdown(f"이번 주 **+{res.config.surge_min*100:.0f}% 이상** 오른 종목. 행을 클릭하면 급등 근거를 분석합니다.")
                if c.get("n"):
                    st.caption(f"과거 통계: 5일 +{res.config.surge_min*100:.0f}%↑ 급등 후 10일 내 추가 +10% 도달 {pct(c['p_more10'], 0)}, "
                               f"10일 뒤 상승 마감 {pct(c['p_positive10'], 0)}, 중앙 수익률 {pct(c['median_ret10'])} (표본 {c['n']}건)")
            elif key == "ai_surge":
                st.markdown("유니버스 전체 과거 데이터로 학습한 AI 모델이 **향후 5거래일 내 급등 확률**이 평균의 1.5배 이상이라고 본 종목")
            elif key in SETUPS:
                st.markdown(SETUPS[key][1])
            elif key == "value":
                st.markdown("재무 점수 70점 이상 (저평가 · 고수익성 · 성장성 · 안정성)")
            df = res.summary if key == "all" else res.categories[key]
            show_table(df.head(100), key=f"t_{key}", extra=extra.get(key))

    st.divider()
    options = list(res.summary.index)
    sel = st.session_state.get("selected")
    idx = options.index(sel) if sel in options else 0
    code = st.selectbox("📌 종목 상세 분석", options, index=idx,
                        format_func=lambda c: f"{res.summary.loc[c, 'name']} ({c}) · 종합 {res.summary.loc[c, 'composite']:.0f}")
    detail(res, code)


if __name__ == "__main__":
    main()
