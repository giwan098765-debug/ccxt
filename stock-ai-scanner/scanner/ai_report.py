"""Claude 기반 애널리스트 리포트 (선택 기능).

ANTHROPIC_API_KEY(또는 `ant auth login` 프로필)가 있으면, 엔진이 계산한 모든 수치를
Claude 에게 넘겨 사람이 읽기 좋은 종합 투자 의견으로 정리한다.
숫자는 엔진이 계산한 것만 사용하도록 지시해 환각을 줄인다.
"""
from __future__ import annotations

import json
import math
import os

MODEL = os.environ.get("STOCK_SCANNER_MODEL", "claude-opus-5-5")

SYSTEM = """너는 한국/미국 주식 퀀트 애널리스트다. 사용자가 준 JSON 은 스캐너 엔진이 실제 시세로 계산한 결과다.
규칙:
- JSON 에 있는 숫자만 인용하고, 새로운 수치나 뉴스·재료를 지어내지 마라. 모르면 '데이터 없음'이라고 써라.
- 성공확률/기대수익은 과거 통계 기반 추정치이며 보장이 아님을 분명히 하라.
- 한국어 마크다운으로, 다음 섹션을 순서대로: ## 한줄 결론, ## 왜 지금인가 (차트·캔들·매물대·보조지표·재무 근거),
  ## 매매 시나리오 (진입·목표가·손절가·성공확률·기대값), ## 리스크, ## 체크포인트(다음 며칠 무엇을 볼지).
- 간결하게, 600단어 이내."""


def available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.path.isdir(os.path.expanduser("~/.config/anthropic")))


def _clean(x):
    if isinstance(x, float):
        return None if not math.isfinite(x) else round(x, 4)
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "isoformat"):
        return x.isoformat()[:10]
    if hasattr(x, "item"):
        return _clean(x.item())
    return x


def build_payload(a: dict) -> dict:
    from .patterns import SETUPS
    d = a["d"]
    last = d.iloc[-1]
    vp = {k: v for k, v in (a["vp"] or {}).items() if k not in ("centers", "volumes", "edges")}
    return _clean({
        "종목": {"코드": a["code"], "이름": a["info"].name, "시장": a["info"].market, "기준일": d.index[-1]},
        "가격": {"종가": float(last["Close"]), "1일": float(last["ret1"]), "이번주": a["week"].get("ret"),
                 "20일": float(last["ret20"])},
        "보조지표": {k: float(last[k]) for k in ("rsi", "macd", "macd_signal", "stoch_k", "adx", "mfi", "cci", "vol_ratio", "bb_pos", "bb_width_pct", "atr_pct", "obv_slope20") if k in last},
        "이동평균": {f"ma{n}": float(last[f"ma{n}"]) for n in (5, 20, 60, 120)},
        "차트모양": a["shape"],
        "최근캔들": [{"패턴": p["name"], "의미": p["desc"], "며칠전": p["days_ago"]} for p in a["candle_patterns"]],
        "매물대": vp,
        "활성셋업": [SETUPS[k][0] for k in a["active_setups"]],
        "셋업별_과거성공확률": {SETUPS[k][0]: v for k, v in a["probs"].items() if v["n"] > 0},
        "매매계획": {"목표수익률": a["target"], "손절률": a["stop"], "목표가": a["row"]["target_px"],
                   "손절가": a["row"]["stop_px"], "성공확률": a["row"]["success_prob"], "기대값": a["row"]["ev"],
                   "판정기간_거래일": 10},
        "AI급등모델": {"5일내급등확률": a["ai"]["prob"], "평균확률": a["ai"]["base_rate"], "배수": a["ai"]["lift"],
                     "주요기여요인": a["ai"]["contrib"]},
        "이번주급등근거": a.get("surge_reasons", []),
        "급등후흐름통계": a["continuation"],
        "재무": a.get("fundamentals", {}),
        "재무평가": a.get("fund_score"),
    })


def generate_report(a: dict) -> str:
    import anthropic

    client = anthropic.Anthropic()
    payload = json.dumps(build_payload(a), ensure_ascii=False, default=str)
    kwargs = dict(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": f"다음 분석 데이터를 바탕으로 리포트를 작성해줘.\n```json\n{payload}\n```"}],
    )
    try:
        # 안전 분류기가 거절할 경우 서버측 자동 대체 모델로 재시도
        resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    except anthropic.BadRequestError:
        resp = client.messages.create(**kwargs)
    if resp.stop_reason == "refusal":
        return "AI 리포트를 생성하지 못했습니다 (요청이 거절됨)."
    return "\n".join(b.text for b in resp.content if b.type == "text")
