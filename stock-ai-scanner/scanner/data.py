"""시세·종목·재무 데이터 공급자.

RealDataProvider : FinanceDataReader(KRX/Naver) + yfinance 로 실제 시장 데이터를 가져온다.
DemoDataProvider : 인터넷이 없을 때 UI/로직을 확인하기 위한 합성 데이터(실제 시세 아님).
"""
from __future__ import annotations

import hashlib
import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(os.environ.get("STOCK_SCANNER_CACHE", Path.home() / ".cache" / "stock_ai_scanner"))
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# 네트워크로 종목 목록을 못 받을 때 쓰는 최소 유니버스
FALLBACK_KR = [
    ("005930", "삼성전자", "KOSPI"), ("000660", "SK하이닉스", "KOSPI"), ("373220", "LG에너지솔루션", "KOSPI"),
    ("207940", "삼성바이오로직스", "KOSPI"), ("005380", "현대차", "KOSPI"), ("000270", "기아", "KOSPI"),
    ("068270", "셀트리온", "KOSPI"), ("035420", "NAVER", "KOSPI"), ("105560", "KB금융", "KOSPI"),
    ("005490", "POSCO홀딩스", "KOSPI"), ("035720", "카카오", "KOSPI"), ("051910", "LG화학", "KOSPI"),
    ("012330", "현대모비스", "KOSPI"), ("028260", "삼성물산", "KOSPI"), ("055550", "신한지주", "KOSPI"),
    ("006400", "삼성SDI", "KOSPI"), ("012450", "한화에어로스페이스", "KOSPI"), ("042660", "한화오션", "KOSPI"),
    ("329180", "HD현대중공업", "KOSPI"), ("034020", "두산에너빌리티", "KOSPI"), ("010130", "고려아연", "KOSPI"),
    ("086790", "하나금융지주", "KOSPI"), ("015760", "한국전력", "KOSPI"), ("032830", "삼성생명", "KOSPI"),
    ("009150", "삼성전기", "KOSPI"), ("066570", "LG전자", "KOSPI"), ("003670", "포스코퓨처엠", "KOSPI"),
    ("018260", "삼성에스디에스", "KOSPI"), ("011200", "HMM", "KOSPI"), ("017670", "SK텔레콤", "KOSPI"),
    ("064350", "현대로템", "KOSPI"), ("267260", "HD현대일렉트릭", "KOSPI"), ("010140", "삼성중공업", "KOSPI"),
    ("247540", "에코프로비엠", "KOSDAQ"), ("086520", "에코프로", "KOSDAQ"), ("196170", "알테오젠", "KOSDAQ"),
    ("028300", "HLB", "KOSDAQ"), ("277810", "레인보우로보틱스", "KOSDAQ"), ("141080", "리가켐바이오", "KOSDAQ"),
    ("058470", "리노공업", "KOSDAQ"), ("214150", "클래시스", "KOSDAQ"), ("145020", "휴젤", "KOSDAQ"),
    ("357780", "솔브레인", "KOSDAQ"), ("039030", "이오테크닉스", "KOSDAQ"), ("240810", "원익IPS", "KOSDAQ"),
    ("263750", "펄어비스", "KOSDAQ"), ("293490", "카카오게임즈", "KOSDAQ"), ("035900", "JYP Ent.", "KOSDAQ"),
    ("041510", "에스엠", "KOSDAQ"), ("095340", "ISC", "KOSDAQ"), ("403870", "HPSP", "KOSDAQ"),
    ("348370", "엔켐", "KOSDAQ"), ("065350", "신성델타테크", "KOSDAQ"), ("091990", "셀트리온헬스케어", "KOSDAQ"),
    ("112040", "위메이드", "KOSDAQ"), ("036930", "주성엔지니어링", "KOSDAQ"), ("098460", "고영", "KOSDAQ"),
    ("222800", "심텍", "KOSDAQ"), ("078600", "대주전자재료", "KOSDAQ"), ("950140", "잉글우드랩", "KOSDAQ"),
]
FALLBACK_US = [
    ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "NVIDIA"), ("AMZN", "Amazon"), ("GOOGL", "Alphabet"),
    ("META", "Meta"), ("TSLA", "Tesla"), ("AVGO", "Broadcom"), ("AMD", "AMD"), ("NFLX", "Netflix"),
    ("PLTR", "Palantir"), ("SMCI", "Super Micro"), ("MU", "Micron"), ("COIN", "Coinbase"), ("MSTR", "MicroStrategy"),
    ("CRWD", "CrowdStrike"), ("SNOW", "Snowflake"), ("UBER", "Uber"), ("SHOP", "Shopify"), ("ARM", "Arm"),
    ("INTC", "Intel"), ("QCOM", "Qualcomm"), ("ORCL", "Oracle"), ("CRM", "Salesforce"), ("ADBE", "Adobe"),
    ("JPM", "JPMorgan"), ("V", "Visa"), ("LLY", "Eli Lilly"), ("UNH", "UnitedHealth"), ("XOM", "Exxon"),
    ("SOFI", "SoFi"), ("RIVN", "Rivian"), ("IONQ", "IonQ"), ("RKLB", "Rocket Lab"), ("HOOD", "Robinhood"),
    ("APP", "AppLovin"), ("DELL", "Dell"), ("ANET", "Arista"), ("TSM", "TSMC"), ("ASML", "ASML"),
]

MARKETS = ["KRX", "KOSPI", "KOSDAQ", "US"]


@dataclass
class StockInfo:
    code: str
    name: str
    market: str
    marcap: float = float("nan")


def _cache_get(key: str, ttl_sec: int):
    path = CACHE_DIR / (hashlib.md5(key.encode()).hexdigest() + ".pkl")
    try:
        if path.exists() and time.time() - path.stat().st_mtime < ttl_sec:
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception:
        return None
    return None


def _cache_put(key: str, value) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / (hashlib.md5(key.encode()).hexdigest() + ".pkl")
        with open(path, "wb") as f:
            pickle.dump(value, f)
    except Exception:
        pass


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=OHLCV_COLS)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df[[c for c in OHLCV_COLS if c in df.columns]].copy()
    if len(df.columns) < 5:
        return pd.DataFrame(columns=OHLCV_COLS)
    df.index = pd.to_datetime(df.index).tz_localize(None) if getattr(df.index, "tz", None) else pd.to_datetime(df.index)
    df = df.astype(float)
    # 거래정지일(시가=0) 등 비정상 봉 제거
    df = df[(df["Close"] > 0) & (df["High"] > 0) & (df["Low"] > 0)]
    df.loc[df["Open"] <= 0, "Open"] = df["Close"]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


class RealDataProvider:
    """실제 시장 데이터. 한국: FinanceDataReader(KRX/네이버), 미국/재무: yfinance."""

    name = "실제 시세 (FinanceDataReader / yfinance)"
    is_demo = False

    def universe(self, market: str = "KRX", limit: int = 300) -> list[StockInfo]:
        key = f"universe:{market}:{datetime.now():%Y%m%d}"
        cached = _cache_get(key, 12 * 3600)
        if cached is None:
            cached = self._fetch_universe(market)
            if cached:
                _cache_put(key, cached)
        return cached[:limit] if limit else cached

    def _fetch_universe(self, market: str) -> list[StockInfo]:
        try:
            import FinanceDataReader as fdr
            if market == "US":
                df = fdr.StockListing("S&P500")
                return [StockInfo(str(r.Symbol), str(r.Name), "US") for r in df.itertuples()]
            df = fdr.StockListing(market)
            code_col = "Code" if "Code" in df.columns else "Symbol"
            df = df.rename(columns={code_col: "Code"})
            if "Marcap" in df.columns:
                df = df.sort_values("Marcap", ascending=False)
            df = df[df["Code"].astype(str).str.fullmatch(r"\d{5}0")]  # 우선주 제외
            df = df[~df["Name"].astype(str).str.contains("스팩|ETN|리츠")]
            out = []
            for r in df.itertuples():
                m = r.Market if "Market" in df.columns else market
                m = "KOSDAQ" if "KOSDAQ" in str(m) else ("KOSPI" if "KOSPI" in str(m) else str(m))
                out.append(StockInfo(str(r.Code), str(r.Name), m, float(getattr(r, "Marcap", float("nan")))))
            return out
        except Exception:
            if market == "US":
                return [StockInfo(c, n, "US") for c, n in FALLBACK_US]
            return [StockInfo(c, n, m) for c, n, m in FALLBACK_KR if market in ("KRX", m)]

    def ohlcv(self, stock: StockInfo, days: int = 750) -> pd.DataFrame:
        start = (datetime.now() - timedelta(days=int(days * 1.5))).strftime("%Y-%m-%d")
        key = f"ohlcv:{stock.code}:{start}:{datetime.now():%Y%m%d%H}"
        cached = _cache_get(key, 3 * 3600)
        if cached is not None:
            return cached
        df = None
        if stock.market != "US":
            try:
                import FinanceDataReader as fdr
                df = fdr.DataReader(stock.code, start)
            except Exception:
                df = None
        if df is None or len(df) == 0:
            try:
                import yfinance as yf
                sym = self.yahoo_symbol(stock)
                df = yf.download(sym, start=start, progress=False, auto_adjust=False, threads=False)
            except Exception:
                df = None
        df = _clean_ohlcv(df)
        _cache_put(key, df)
        return df

    @staticmethod
    def yahoo_symbol(stock: StockInfo) -> str:
        if stock.market == "US":
            return stock.code
        return stock.code + (".KQ" if stock.market == "KOSDAQ" else ".KS")

    def fundamentals(self, stock: StockInfo) -> dict:
        key = f"fund:{stock.code}:{datetime.now():%Y%m%d}"
        cached = _cache_get(key, 24 * 3600)
        if cached is not None:
            return cached
        out: dict = {}
        try:
            import yfinance as yf
            t = yf.Ticker(self.yahoo_symbol(stock))
            info = t.info or {}
            fields = {
                "per": "trailingPE", "forward_per": "forwardPE", "pbr": "priceToBook", "roe": "returnOnEquity",
                "op_margin": "operatingMargins", "net_margin": "profitMargins", "revenue_growth": "revenueGrowth",
                "earnings_growth": "earningsGrowth", "debt_to_equity": "debtToEquity", "current_ratio": "currentRatio",
                "dividend_yield": "dividendYield", "market_cap": "marketCap", "sector": "sector", "industry": "industry",
            }
            for k, v in fields.items():
                if info.get(v) is not None:
                    out[k] = info.get(v)
            try:
                q = t.quarterly_income_stmt
                if q is not None and not q.empty:
                    for row, name in (("Total Revenue", "revenue_q"), ("Operating Income", "op_income_q"), ("Net Income", "net_income_q")):
                        if row in q.index:
                            s = q.loc[row].dropna().sort_index()
                            out[name] = [float(x) for x in s.values[-5:]]
            except Exception:
                pass
        except Exception:
            pass
        _cache_put(key, out)
        return out


class DemoDataProvider:
    """합성 데이터 — 네트워크 없이 기능을 시연/테스트하기 위한 용도. 투자 판단에 사용 금지."""

    name = "데모 (합성 데이터 · 실제 시세 아님)"
    is_demo = True

    def __init__(self, n_stocks: int = 80, seed: int = 7):
        self.n = n_stocks
        self.seed = seed

    def universe(self, market: str = "KRX", limit: int = 300) -> list[StockInfo]:
        base = [StockInfo(c, n, m) for c, n, m in FALLBACK_KR]
        out = list(base)
        i = 0
        while len(out) < self.n:
            i += 1
            out.append(StockInfo(f"9{i:04d}0", f"데모종목{i}", "KOSDAQ"))
        out = out[: self.n]
        if market in ("KOSPI", "KOSDAQ"):
            out = [s for s in out if s.market == market]
        return out[:limit] if limit else out

    def ohlcv(self, stock: StockInfo, days: int = 750) -> pd.DataFrame:
        rng = np.random.default_rng(int(hashlib.md5((stock.code + str(self.seed)).encode()).hexdigest()[:8], 16))
        n = days
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
        drift = rng.normal(0.0003, 0.0006)
        vol = rng.uniform(0.012, 0.035)
        rets = rng.normal(drift, vol, n)
        # 급등 이벤트 주입 (사전 스퀴즈/거래량 증가 후 급등)
        vmult = np.ones(n)
        for _ in range(rng.integers(2, 6)):
            t = int(rng.integers(60, n - 3))
            rets[t - 15:t] *= 0.35
            vmult[t - 5:t] *= 1.6
            k = int(rng.integers(1, 4))
            rets[t:t + k] = np.log1p(rng.uniform(0.06, 0.2999, k))
            vmult[t:t + k] *= rng.uniform(3, 8)
        if rng.random() < 0.35:  # 이번 주 급등 종목
            k = int(rng.integers(1, 3))
            rets[-k:] = np.log1p(rng.uniform(0.08, 0.2999, k))
            vmult[-k:] *= rng.uniform(4, 9)
        close = 10000 * rng.uniform(0.3, 8) * np.exp(np.cumsum(rets))
        prev = np.concatenate([[close[0]], close[:-1]])
        openp = prev * (1 + rng.normal(0, vol / 3, n))
        hi = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
        lo = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
        volume = rng.lognormal(12, 0.4, n) * vmult
        return pd.DataFrame({"Open": openp, "High": hi, "Low": lo, "Close": close, "Volume": volume}, index=idx)

    def fundamentals(self, stock: StockInfo) -> dict:
        rng = np.random.default_rng(int(hashlib.md5(stock.code.encode()).hexdigest()[:8], 16))
        return {
            "per": float(rng.uniform(4, 60)), "pbr": float(rng.uniform(0.3, 6)), "roe": float(rng.uniform(-0.1, 0.3)),
            "op_margin": float(rng.uniform(-0.1, 0.3)), "revenue_growth": float(rng.uniform(-0.2, 0.6)),
            "earnings_growth": float(rng.uniform(-0.5, 1.0)), "debt_to_equity": float(rng.uniform(10, 250)),
            "sector": "데모",
        }


def get_provider(demo: bool = False):
    return DemoDataProvider() if demo else RealDataProvider()
