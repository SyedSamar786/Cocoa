"""
Cocoa Procurement Agent - core module
AI Agents in Business | Case 1: the cocoa price shock (2023-2026)

A fictional UK chocolate maker asks: "Given what we know today, what share of the
next N months' cocoa need should we buy forward now (hedge), and why?"

Used by both the Colab notebook and the Streamlit app.
Teaching demo with a fictional company - not investment advice.
"""

# %% STEP 0 | Imports and settings
import os
import io
import re
import json
import time
import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import requests

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (cocoa-agent teaching demo)"}

# Cocoa-belt weather points (latitude, longitude)
LOCATIONS = {
    "Daloa (Cote d'Ivoire)": (6.877, -6.450),
    "Soubre (Cote d'Ivoire)": (5.785, -6.594),
    "San-Pedro (Cote d'Ivoire)": (4.748, -6.636),
    "Kumasi (Ghana)": (6.688, -1.624),
}
WEATHER_BASELINE_YEARS = (2000, 2020)

PRICE_URLS = {
    "fred": "https://fred.stlouisfed.org/series/PCOCOUSDM",
    "yahoo": "https://finance.yahoo.com/quote/CC=F",
}
WEATHER_URL = "https://open-meteo.com/en/docs/historical-weather-api"


# %% STEP 1A | Price data: FRED (monthly) and Yahoo Finance (daily futures)
def fetch_fred_cocoa() -> pd.Series:
    """IMF global cocoa price via FRED. USD per tonne, monthly."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PCOCOUSDM"
    r = requests.get(url, headers=HTTP_HEADERS, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "price"]
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna()
    s = pd.Series(df["price"].values,
                  index=pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M"),
                  name="price_usd_t")
    return s


def fetch_futures_daily(start: str = "2014-01-01") -> pd.Series:
    """ICE cocoa futures (continuous front month) from Yahoo Finance. USD per tonne, daily."""
    import yfinance as yf
    df = yf.download("CC=F", start=start, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError("Yahoo Finance returned no data for CC=F")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna().astype(float)
    idx = pd.to_datetime(close.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    close.index = idx
    close.name = "close_usd_t"
    return close


def build_monthly_price(fred: Optional[pd.Series], daily: Optional[pd.Series]):
    """One monthly price series for the agent and the backtest.
    Prefer the daily futures (more up to date); fall back to FRED."""
    if daily is not None and len(daily) > 250:
        m = daily.groupby(daily.index.to_period("M")).mean()
        m.name = "price_usd_t"
        return m, "Yahoo Finance CC=F (ICE cocoa futures, monthly mean of daily closes)"
    if fred is not None and len(fred) > 0:
        return fred, "FRED PCOCOUSDM (IMF global cocoa price, monthly)"
    raise RuntimeError("No price data available from Yahoo Finance or FRED")


# %% STEP 1B | Weather data: Open-Meteo historical archive for the cocoa belt
def fetch_weather(start: str = "2000-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Daily rain and max temperature for each location, aggregated to months."""
    end = end or (dt.date.today() - dt.timedelta(days=7)).isoformat()
    frames = []
    for name, (lat, lon) in LOCATIONS.items():
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params=dict(latitude=lat, longitude=lon, start_date=start, end_date=end,
                        daily="precipitation_sum,temperature_2m_max", timezone="UTC"),
            headers=HTTP_HEADERS, timeout=90)
        r.raise_for_status()
        d = r.json()["daily"]
        f = pd.DataFrame({"date": pd.to_datetime(d["time"]),
                          "rain_mm": d["precipitation_sum"],
                          "tmax_c": d["temperature_2m_max"]})
        f["location"] = name
        frames.append(f)
        time.sleep(0.5)  # be polite to the free API
    daily = pd.concat(frames, ignore_index=True)
    daily["month"] = daily["date"].dt.to_period("M")
    monthly = (daily.groupby(["location", "month"])
               .agg(rain_mm=("rain_mm", "sum"), tmax_c=("tmax_c", "mean"))
               .reset_index())
    return add_weather_anomalies(monthly)


def add_weather_anomalies(w: pd.DataFrame) -> pd.DataFrame:
    """Compare each month with its 2000-2020 normal for the same location and calendar month."""
    w = w.copy()
    w["cal_month"] = w["month"].dt.month
    years = w["month"].dt.year
    y0, y1 = WEATHER_BASELINE_YEARS
    base = (w[(years >= y0) & (years <= y1)]
            .groupby(["location", "cal_month"])
            .agg(rain_base=("rain_mm", "mean"), tmax_base=("tmax_c", "mean"))
            .reset_index())
    w = w.drop(columns=[c for c in ["rain_base", "tmax_base"] if c in w.columns])
    w = w.merge(base, on=["location", "cal_month"], how="left")
    w["tmax_anom_c"] = w["tmax_c"] - w["tmax_base"]
    return w


# %% STEP 1C | News and supply facts (curated, dated, with sources)
# Headlines are used as dated evidence. Summaries are short paraphrases, not article text.
# date_note = "approximate" where the exact publication day could not be confirmed.
NEWS_ROWS = [
    ("N01", "2024-02-07", "exact", "AFP via The Peninsula", "Cocoa prices smash records on weather woes",
     "Record prices in London and New York as bad weather damages crops in Ghana and Ivory Coast; Ivory Coast port arrivals down about 35% y/y for October-January.",
     "https://thepeninsulaqatar.com/article/07/02/2024/cocoa-prices-smash-records-on-weather-woes"),
    ("N02", "2024-03-02", "exact", "Seeking Alpha", "Why chocolate prices have risen to historically high levels",
     "Weather and disease problems in Ivory Coast, Ghana and Nigeria; ageing trees and disease weigh on Ghana's cocoa sector.",
     "https://seekingalpha.com/article/4675482-chocolate-prices-risen-to-historically-high-levels"),
    ("N03", "2024-03-22", "exact", "DP Tribune", "Cocoa prices rise to fresh record, nears $9,000 per metric ton",
     "New record highs as weather conditions worsen in Ivory Coast.",
     "https://business.dptribune.com/dptribune/news/read/43987845/cocoa_prices_rise_to_fresh_record"),
    ("N04", "2024-04-15", "approximate", "Reuters via Checkout.ie", "Cocoa futures hit record highs amid low rainfall and high temperatures",
     "London and New York futures at records; heat and low rainfall threaten the development of mid-crop pods.",
     "https://www.checkout.ie/supply-chain/cocoa-futures-hit-record-highs-amid-low-rainfall-and-high-temperatures-209532"),
    ("N05", "2024-06-28", "exact", "Bloomberg via BNN Bloomberg", "Ivory Coast expects cocoa rebound next season on better weather",
     "Ivory Coast eyes about 2 million tons for 2024-25 versus about 1.8 million this season; prices have eased from the peak.",
     "https://bnnbloomberg.ca/ivory-coast-expects-cocoa-rebound-next-season-on-better-weather-1.2090895"),
    ("N06", "2024-07-31", "exact", "Kapital.kz via DairyNews", "Cocoa bean prices fall to 4-month low",
     "New York futures drop to about $6,650/t on expectations of more supply and weaker demand; analysts expect a surplus in 2024-25.",
     "https://dairynews.today/global/news/tseny-na-kakao-boby-upali-do-4-mesyachnogo-minimuma.html"),
    ("N07", "2024-12-18", "approximate", "Anadolu via Africa-Press", "Cocoa hits record price of nearly $13K per ton",
     "Record near $12,931/t, up over 200% in 2024; swollen shoot virus and weather hit West African output; a 2024-25 deficit is expected.",
     "https://www.africa-press.net/mauritius/?p=49385"),
    ("N08", "2025-03-24", "exact", "Selina Wamucii", "Raw cocoa sticks to $8,000 amid sinking demand",
     "An ICCO surplus forecast turned the market bearish; high prices are cutting chocolate demand.",
     "https://www.selinawamucii.com/news/?p=22115"),
    ("N09", "2025-04-15", "exact", "XTB", "Cocoa tumbles 4%",
     "Futures fall on rain forecasts and weak grinding data; Barry Callebaut cut guidance after lower sales volumes.",
     "https://www.xtb.com/en/market-analysis/cocoa-tumbles-4"),
    ("N10", "2025-07-20", "approximate", "J.P. Morgan Global Research", "Cocoa prices: demand destruction confirmed by grind data",
     "Q2 2025 grindings fell in Europe, Asia and North America; production expected to rise in 2025/26.",
     "https://cws-main-ndc.jpmorgan.com/insights/global-research/commodities/cocoa-prices"),
    ("N11", "2025-10-12", "approximate", "Komersant (citing FT)", "After a record drop in cocoa prices, chocolate may become more affordable",
     "Cocoa at a 20-month low near $6,150/t in New York; 2025-26 supply expected to exceed demand.",
     "https://komersant.ua/en/pislia-rekordnoho-padinnia-tsin-na-kakao-shokolad-mozhe-staty-dostupnishym"),
    ("N12", "2026-03-26", "exact", "Factually", "Current developments in cocoa prices",
     "Prices collapsed from 2024-25 peaks to roughly $3,400-4,400/t in early 2026 as surplus forecasts grew; retail chocolate prices stay high.",
     "https://factually.co/fact-checks/finance/current-development-cocoa-prices-2026-c81a20"),
    ("N13", "2026-04-29", "exact", "CropGPT (World Bank outlook)", "Cocoa prices forecast to halve in 2026, edge up in 2027",
     "World Bank projects about a 51% fall in 2026 prices after historic highs, with a small rise in 2027.",
     "https://cropgpt.ai/cocoa-prices-forecast-to-halve-in-2026-edge-up-in-2027"),
    ("N14", "2026-06-12", "approximate", "Barchart via Nasdaq", "Ivory Coast weather concerns push cocoa prices higher",
     "Heavy rains disrupt farm and port access; Ivory Coast arrivals up about 19% y/y; StoneX cut its 2026/27 surplus estimate citing El Nino risk.",
     "https://www.nasdaq.com/articles/ivory-coast-weather-concerns-push-cocoa-prices-higher"),
    ("N15", "2026-07-06", "exact", "Trading Economics", "Cocoa hits 24-week high",
     "Cocoa rises to about $5,360/t, up 39% in four weeks, on worries about the 2026/27 West African crop.",
     "https://tradingeconomics.com/commodity/cocoa/news/564465"),
    ("N16", "2026-08-04", "exact", "Barchart via Webull", "Why have cocoa prices recovered?",
     "Futures bottomed near $2,846 in March 2026 and rallied to about $6,478 in July on West African floods and recovering demand.",
     "https://www.webull.ca/news-detail/15346137946858496"),
    ("N17", "2026-08-05", "approximate", "Barchart via Webull", "Cocoa recovers some ground after sharp drop on beneficial West African weather",
     "Good flowering conditions and ICE stocks at a 2-year high weigh on prices; Ghana expects lower 2026/27 output.",
     "https://www.webull.com.au/news-detail/15390846648656896"),
    ("N18", "2026-09-02", "exact", "Business Today (BMI)", "Cocoa prices could hit US$4,990 as weather risks threaten supply",
     "BMI raises its 2026 average forecast and expects the 2026/27 surplus to shrink sharply on El Nino risk.",
     "https://www.businesstoday.com.my/2026/09/02/cocoa-prices-could-hit-us4990-as-weather-risks-threaten-supply-bmi/"),
    ("N19", "2026-09-15", "approximate", "Barchart via Webull", "Cocoa prices rebound on forecasts for dry weather in the Ivory Coast",
     "Higher Ivory Coast output had pressured prices, but dry-weather forecasts, black pod disease and weak early crop surveys support them.",
     "https://www.webull.com/news/15617221168071680"),
]

SUPPLY_ROWS = [
    ("S01", "2024-02-07", "exact", "ICCO (via AFP)",
     "Ivory Coast and Ghana produced almost 60% of the 2022/23 world cocoa harvest.",
     "https://thepeninsulaqatar.com/article/07/02/2024/cocoa-prices-smash-records-on-weather-woes"),
    ("S02", "2024-02-07", "exact", "Industry estimates (via AFP)",
     "Ivory Coast port arrivals fell about 35% y/y between October 2023 and end-January 2024.",
     "https://thepeninsulaqatar.com/article/07/02/2024/cocoa-prices-smash-records-on-weather-woes"),
    ("S03", "2024-06-28", "exact", "Bloomberg",
     "Ivory Coast expects about 2.0 Mt in 2024-25 vs ICCO's 1.8 Mt estimate for 2023-24 (pre-2023 five-year average 2.2 Mt).",
     "https://bnnbloomberg.ca/ivory-coast-expects-cocoa-rebound-next-season-on-better-weather-1.2090895"),
    ("S04", "2024-12-18", "approximate", "Anadolu",
     "Ghana planned to defer up to 350,000 t of bean deliveries to next season because of poor crops.",
     "https://www.africa-press.net/mauritius/?p=49385"),
    ("S05", "2025-02-28", "approximate", "ICCO (via Selina Wamucii)",
     "ICCO forecasts a 2024/25 global surplus of about 142,000 t.",
     "https://www.selinawamucii.com/news/?p=22115"),
    ("S06", "2025-04-15", "exact", "Malaysian Cocoa Board / AIPC (via XTB)",
     "Q1 2025 grindings fell about 15% y/y in Malaysia and 13% in Brazil.",
     "https://www.xtb.com/en/market-analysis/cocoa-tumbles-4"),
    ("S07", "2025-07-20", "approximate", "J.P. Morgan",
     "Q2 2025 grindings: Europe -7.2%, Asia -16%, North America -2.8% y/y.",
     "https://cws-main-ndc.jpmorgan.com/insights/global-research/commodities/cocoa-prices"),
    ("S08", "2026-04-29", "exact", "World Bank outlook (via CropGPT)",
     "Cocoa price forecast to fall about 51% in 2026 to about $3.80/kg, then about $4.20/kg in 2027.",
     "https://cropgpt.ai/cocoa-prices-forecast-to-halve-in-2026-edge-up-in-2027"),
    ("S09", "2026-06-12", "approximate", "Barchart (via Nasdaq)",
     "Ivory Coast arrivals 1.95 Mt (1 Oct 2025 - 7 Jun 2026), +18.9% y/y; StoneX cut its 2026/27 surplus estimate to 149,000 t from 267,000 t.",
     "https://www.nasdaq.com/articles/ivory-coast-weather-concerns-push-cocoa-prices-higher"),
    ("S10", "2026-07-23", "exact", "Transgraph (via Barchart)",
     "2026/27 global surplus forecast to shrink to 80,000 t from 415,000 t; production 4.87 Mt vs 5.11 Mt.",
     "https://www.webull.com.au/news-detail/15390846648656896"),
    ("S11", "2026-07-30", "exact", "COCOBOD (via Barchart)",
     "Ghana 2026/27 production projected at 450,000-550,000 t vs 750,000 t for 2025/26 (swollen shoot, ageing farms, El Nino).",
     "https://www.webull.com.au/news-detail/15390846648656896"),
    ("S12", "2026-08-05", "approximate", "ICE (via Barchart)",
     "ICE cocoa inventories at a 2-year high of about 3.38 million bags.",
     "https://www.webull.com.au/news-detail/15390846648656896"),
    ("S13", "2026-09-02", "exact", "BMI",
     "2026 average price forecast raised to $4,990/t; 2026/27 surplus seen at 82,000 t vs 442,000 t in 2025/26.",
     "https://www.businesstoday.com.my/2026/09/02/cocoa-prices-could-hit-us4990-as-weather-risks-threaten-supply-bmi/"),
    ("S14", "2026-09-15", "approximate", "Conseil du Cafe-Cacao (via Barchart)",
     "Ivory Coast harvested 2.06 Mt Jun 2025 - Jun 2026, +30% y/y; early 2026/27 surveys show below-average cherelle formation.",
     "https://www.webull.com/news/15617221168071680"),
]


def load_curated_news() -> pd.DataFrame:
    df = pd.DataFrame(NEWS_ROWS, columns=["id", "date", "date_note", "source", "title", "summary", "url"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_supply_facts() -> pd.DataFrame:
    df = pd.DataFrame(SUPPLY_ROWS, columns=["id", "date", "date_note", "source", "fact", "url"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_gdelt_headlines(query: str = "cocoa (prices OR harvest OR grindings) sourcelang:english",
                          max_records: int = 25) -> pd.DataFrame:
    """Live headlines from the GDELT DOC API (covers roughly the last 3 months only)."""
    r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc",
                     params=dict(query=query, mode="artlist", maxrecords=max_records,
                                 format="json", timespan="3months", sort="datedesc"),
                     headers=HTTP_HEADERS, timeout=60)
    r.raise_for_status()
    arts = r.json().get("articles", [])
    rows = []
    for i, a in enumerate(arts, start=1):
        rows.append({"id": f"G{i:02d}",
                     "date": pd.to_datetime(a.get("seendate", ""), format="%Y%m%dT%H%M%SZ", errors="coerce"),
                     "date_note": "exact", "source": a.get("domain", "GDELT"),
                     "title": a.get("title", ""), "summary": "(live headline - title only)",
                     "url": a.get("url", "")})
    df = pd.DataFrame(rows, columns=["id", "date", "date_note", "source", "title", "summary", "url"])
    return df.dropna(subset=["date"]).drop_duplicates(subset=["title"])


# %% STEP 1D | Load everything once, with a local cache for reproducible recordings
@dataclass
class DataBundle:
    monthly_price: pd.Series
    price_source: str
    daily: Optional[pd.Series]
    weather: Optional[pd.DataFrame]
    news: pd.DataFrame
    supply: pd.DataFrame
    live_news: Optional[pd.DataFrame] = None
    status: dict = field(default_factory=dict)


def _fresh(path: str, max_age_hours: float) -> bool:
    return os.path.exists(path) and (time.time() - os.path.getmtime(path)) < max_age_hours * 3600


def load_all_data(cache_dir: str = "cache", max_age_hours: float = 12,
                  include_live_news: bool = True) -> DataBundle:
    os.makedirs(cache_dir, exist_ok=True)
    status = {}

    # --- daily futures
    p = os.path.join(cache_dir, "futures_daily.csv")
    daily = None
    try:
        if _fresh(p, max_age_hours):
            raise FileExistsError
        daily = fetch_futures_daily()
        daily.to_frame().to_csv(p, index_label="date")
        status["futures_daily"] = f"fetched ({len(daily)} days)"
    except Exception as e:
        if os.path.exists(p):
            d = pd.read_csv(p, parse_dates=["date"])
            daily = pd.Series(d["close_usd_t"].values, index=pd.DatetimeIndex(d["date"]), name="close_usd_t")
            status["futures_daily"] = "loaded from cache" if isinstance(e, FileExistsError) else f"fetch failed, cache used ({e})"
        else:
            status["futures_daily"] = f"unavailable ({e})"

    # --- FRED monthly
    p = os.path.join(cache_dir, "fred_monthly.csv")
    fred = None
    try:
        if _fresh(p, max_age_hours):
            raise FileExistsError
        fred = fetch_fred_cocoa()
        pd.DataFrame({"month": fred.index.astype(str), "price_usd_t": fred.values}).to_csv(p, index=False)
        status["fred_monthly"] = f"fetched ({len(fred)} months)"
    except Exception as e:
        if os.path.exists(p):
            d = pd.read_csv(p)
            fred = pd.Series(d["price_usd_t"].values, index=pd.PeriodIndex(d["month"], freq="M"), name="price_usd_t")
            status["fred_monthly"] = "loaded from cache" if isinstance(e, FileExistsError) else f"fetch failed, cache used ({e})"
        else:
            status["fred_monthly"] = f"unavailable ({e})"

    monthly, price_source = build_monthly_price(fred, daily)

    # --- weather
    p = os.path.join(cache_dir, "weather_monthly.csv")
    weather = None
    try:
        if _fresh(p, max_age_hours * 4):
            raise FileExistsError
        weather = fetch_weather()
        out = weather.copy()
        out["month"] = out["month"].astype(str)
        out.to_csv(p, index=False)
        status["weather"] = f"fetched ({weather['month'].nunique()} months x {weather['location'].nunique()} locations)"
    except Exception as e:
        if os.path.exists(p):
            weather = pd.read_csv(p)
            weather["month"] = pd.PeriodIndex(weather["month"], freq="M")
            status["weather"] = "loaded from cache" if isinstance(e, FileExistsError) else f"fetch failed, cache used ({e})"
        else:
            status["weather"] = f"unavailable ({e})"

    # --- live news (optional)
    live = None
    if include_live_news:
        p = os.path.join(cache_dir, "gdelt_live.csv")
        try:
            if _fresh(p, 3):
                raise FileExistsError
            live = fetch_gdelt_headlines()
            live.to_csv(p, index=False)
            status["live_news"] = f"fetched ({len(live)} headlines)"
        except Exception as e:
            if os.path.exists(p):
                live = pd.read_csv(p, parse_dates=["date"])
                status["live_news"] = "loaded from cache" if isinstance(e, FileExistsError) else f"fetch failed, cache used ({e})"
            else:
                status["live_news"] = f"unavailable ({e})"

    return DataBundle(monthly_price=monthly, price_source=price_source, daily=daily, weather=weather,
                      news=load_curated_news(), supply=load_supply_facts(), live_news=live, status=status)


# %% STEP 2 | The fictional company
@dataclass
class CompanyProfile:
    name: str = "Bramble & Bean Chocolate Ltd (fictional)"
    monthly_cocoa_t: float = 120.0           # tonnes of cocoa beans needed per month
    monthly_revenue_usd: float = 3_000_000   # sales per month
    monthly_other_costs_usd: float = 1_500_000  # everything except cocoa
    current_hedge_pct: float = 25.0          # share of the next months already bought forward


# %% STEP 3 | Point-in-time snapshot: the agent only sees data available on the decision date
@dataclass
class Snapshot:
    as_of: pd.Timestamp
    monthly_price: pd.Series
    daily: Optional[pd.Series]
    weather: Optional[pd.DataFrame]
    news: pd.DataFrame
    supply: pd.DataFrame
    price_source: str


def get_snapshot(data: DataBundle, as_of) -> Snapshot:
    as_of = pd.Timestamp(as_of).normalize()
    this_month = as_of.to_period("M")
    monthly = data.monthly_price[data.monthly_price.index < this_month]   # complete months only
    daily = data.daily[data.daily.index <= as_of] if data.daily is not None else None
    weather = data.weather[data.weather["month"] < this_month] if data.weather is not None else None
    news = data.news[data.news["date"] <= as_of]
    if data.live_news is not None and len(data.live_news):
        live = data.live_news[data.live_news["date"] <= as_of]
        news = pd.concat([news, live], ignore_index=True).drop_duplicates(subset=["url"])
    supply = data.supply[data.supply["date"] <= as_of]
    return Snapshot(as_of, monthly, daily, weather, news, supply, data.price_source)


# %% STEP 4 | Agent tools (each returns compact JSON plus an evidence id)
class AgentTools:
    def __init__(self, snap: Snapshot, profile: CompanyProfile, horizon_months: int = 6,
                 enabled=("price", "weather", "news", "supply", "margin")):
        self.snap = snap
        self.profile = profile
        self.horizon = horizon_months
        self.enabled = set(enabled)
        self.evidence = {}   # evidence id -> {id, type, detail, url}
        self.calls = []      # (tool name, args)

    def _add(self, eid, kind, detail, url=""):
        self.evidence[eid] = {"id": eid, "type": kind, "detail": detail, "url": url}

    def current_price(self):
        """Today's price: last daily futures close, else last monthly average."""
        if self.snap.daily is not None and len(self.snap.daily):
            return (float(self.snap.daily.iloc[-1]), self.snap.daily.index[-1].date().isoformat(),
                    "Yahoo Finance CC=F daily close")
        mp = self.snap.monthly_price
        return float(mp.iloc[-1]), str(mp.index[-1]), self.snap.price_source

    # ---- tool 1
    def get_price_trend(self) -> dict:
        mp = self.snap.monthly_price.dropna()
        if len(mp) < 13:
            return {"error": "not enough price history before this date"}

        def chg(n):
            return round((mp.iloc[-1] / mp.iloc[-1 - n] - 1) * 100, 1)

        last10y = mp[mp.index > mp.index[-1] - 120]
        price, pdate, psrc = self.current_price()
        out = {
            "latest_complete_month": str(mp.index[-1]),
            "monthly_avg_usd_t": round(float(mp.iloc[-1])),
            "change_3m_pct": chg(3), "change_6m_pct": chg(6), "change_12m_pct": chg(12),
            "percentile_vs_last_10y": int(round((last10y < mp.iloc[-1]).mean() * 100)),
            "current_price_usd_t": round(price), "current_price_date": pdate,
        }
        d = self.snap.daily
        if d is not None and len(d) > 70:
            last_year = d[d.index > d.index[-1] - pd.Timedelta(days=365)]
            rets = np.log(d).diff().dropna().iloc[-63:]
            out.update({"high_52w_usd_t": round(float(last_year.max())),
                        "low_52w_usd_t": round(float(last_year.min())),
                        "volatility_3m_annualised_pct": round(float(rets.std() * np.sqrt(252) * 100), 1)})
        url = PRICE_URLS["yahoo"] if "Yahoo" in psrc else PRICE_URLS["fred"]
        self._add("PRICE", "market data", f"{psrc} up to {pdate}; monthly series: {self.snap.price_source}", url)
        out["evidence_id"] = "PRICE"
        return out

    # ---- tool 2
    def get_weather_anomaly(self, months: int = 3) -> dict:
        w = self.snap.weather
        if w is None or w.empty:
            return {"error": "weather data unavailable"}
        months = int(max(1, min(int(months), 12)))
        last = sorted(w["month"].unique())[-months:]
        sub = w[w["month"].isin(last)]
        g = sub.groupby("location").agg(rain=("rain_mm", "sum"), base=("rain_base", "sum"),
                                        tmax_anom=("tmax_anom_c", "mean"))
        by_loc = {loc: {"rain_mm": round(float(r.rain)), "normal_mm": round(float(r.base)),
                        "rain_anomaly_pct": round(float((r.rain / r.base - 1) * 100), 1) if r.base else None,
                        "tmax_anomaly_c": round(float(r.tmax_anom), 2)}
                  for loc, r in g.iterrows()}
        out = {"window": f"{last[0]} to {last[-1]}",
               "avg_rain_anomaly_pct": round(float((g.rain.sum() / g.base.sum() - 1) * 100), 1),
               "avg_tmax_anomaly_c": round(float(g.tmax_anom.mean()), 2),
               "baseline": f"{WEATHER_BASELINE_YEARS[0]}-{WEATHER_BASELINE_YEARS[1]} normals for the same months",
               "by_location": by_loc, "evidence_id": "WEATHER"}
        self._add("WEATHER", "weather data",
                  f"Open-Meteo archive, {out['window']}: rain {out['avg_rain_anomaly_pct']}% vs normal, "
                  f"max temperature {out['avg_tmax_anomaly_c']:+} C", WEATHER_URL)
        return out

    # ---- tool 3
    def get_news_headlines(self, days: int = 180, max_items: int = 8) -> dict:
        n = self.snap.news.sort_values("date", ascending=False)
        if n.empty:
            return {"headlines": [], "note": "no headlines available before this date"}
        cutoff = self.snap.as_of - pd.Timedelta(days=int(days))
        recent = n[n["date"] >= cutoff].head(int(max_items))
        note = None
        if recent.empty:
            recent = n.head(3)
            note = f"no headlines in the last {days} days; showing the 3 most recent older ones"
        items = []
        for _, r in recent.iterrows():
            items.append({"id": r["id"], "date": r["date"].date().isoformat(), "source": r["source"],
                          "title": r["title"], "summary": r["summary"]})
            self._add(r["id"], "news", f"{r['date'].date()} {r['source']}: {r['title']}", r["url"])
        out = {"headlines": items}
        if note:
            out["note"] = note
        return out

    # ---- tool 4
    def get_supply_facts(self, max_items: int = 8) -> dict:
        s = self.snap.supply.sort_values("date", ascending=False).head(int(max_items))
        items = []
        for _, r in s.iterrows():
            items.append({"id": r["id"], "date": r["date"].date().isoformat(), "source": r["source"], "fact": r["fact"]})
            self._add(r["id"], "supply fact", f"{r['date'].date()} {r['source']}: {r['fact']}", r["url"])
        return {"facts": items} if items else {"facts": [], "note": "no supply facts available before this date"}

    # ---- tool 5
    def calc_margin_impact(self, scenario_price_usd_t: float, hedge_pct: Optional[float] = None) -> dict:
        p = self.profile
        locked, _, _ = self.current_price()
        h = float(p.current_hedge_pct if hedge_pct is None else hedge_pct)
        h = max(0.0, min(100.0, h)) / 100
        scen = float(scenario_price_usd_t)

        def margin(hh):
            eff = hh * locked + (1 - hh) * scen
            m = p.monthly_revenue_usd - p.monthly_other_costs_usd - p.monthly_cocoa_t * eff
            return eff, m

        eff, m = margin(h)
        _, m0 = margin(0.0)
        out = {"locked_price_usd_t": round(locked), "scenario_price_usd_t": round(scen),
               "hedge_pct": round(h * 100), "effective_cocoa_price_usd_t": round(eff),
               "monthly_gross_margin_usd": round(m),
               "gross_margin_pct": round(m / p.monthly_revenue_usd * 100, 1),
               "margin_vs_no_hedge_usd_per_month": round(m - m0), "evidence_id": "CALC"}
        self._add("CALC", "calculation", "Margin model of the fictional company (calc_margin_impact)")
        return out

    # ---- plumbing
    TOOL_MAP = {"get_price_trend": "price", "get_weather_anomaly": "weather",
                "get_news_headlines": "news", "get_supply_facts": "supply",
                "calc_margin_impact": "margin"}

    def schemas(self) -> list:
        all_schemas = [
            {"type": "function", "function": {
                "name": "get_price_trend",
                "description": "Cocoa price level and trend up to today: monthly average, 3/6/12-month change, 10-year percentile, current price, 52-week range, volatility.",
                "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {
                "name": "get_weather_anomaly",
                "description": "Rainfall and temperature in the West African cocoa belt over recent complete months, compared with 2000-2020 normals.",
                "parameters": {"type": "object", "properties": {
                    "months": {"type": "integer", "description": "Number of recent months (1-12), default 3"}}}}},
            {"type": "function", "function": {
                "name": "get_news_headlines",
                "description": "Dated cocoa market headlines published on or before today, newest first, each with an evidence id.",
                "parameters": {"type": "object", "properties": {
                    "days": {"type": "integer", "description": "Look-back window in days, default 180"},
                    "max_items": {"type": "integer", "description": "Maximum headlines, default 8"}}}}},
            {"type": "function", "function": {
                "name": "get_supply_facts",
                "description": "Dated supply and demand facts (production, arrivals, grindings, surplus forecasts) published on or before today.",
                "parameters": {"type": "object", "properties": {
                    "max_items": {"type": "integer", "description": "Maximum facts, default 8"}}}}},
            {"type": "function", "function": {
                "name": "calc_margin_impact",
                "description": "Company monthly gross margin if cocoa averages scenario_price_usd_t, with hedge_pct of need locked at today's price.",
                "parameters": {"type": "object", "properties": {
                    "scenario_price_usd_t": {"type": "number", "description": "Scenario average cocoa price, USD per tonne"},
                    "hedge_pct": {"type": "number", "description": "Share hedged, 0-100"}},
                    "required": ["scenario_price_usd_t"]}}},
        ]
        return [s for s in all_schemas if self.TOOL_MAP[s["function"]["name"]] in self.enabled]

    def dispatch(self, name: str, args: dict) -> dict:
        if name not in self.TOOL_MAP or self.TOOL_MAP[name] not in self.enabled:
            return {"error": f"tool '{name}' is not available"}
        self.calls.append((name, args))
        try:
            return getattr(self, name)(**(args or {}))
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except Exception as e:
            return {"error": f"{name} failed: {e}"}


# %% STEP 5A | LLM router: free Gemini and Groq endpoints with failover
PROVIDERS = {
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "key_env": "GEMINI_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1",
             "key_env": "GROQ_API_KEY",
             # each Groq model has its own free rate limit, so several candidates = more capacity
             "prefer": ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b",
                        "qwen/qwen3-32b", "meta-llama/llama-4-scout-17b-16e-instruct"]},
}
_EXCLUDE = ("embed", "tts", "image", "vision", "audio", "whisper", "guard", "imagen", "veo",
            "live", "native", "aqa", "gemma", "learnlm", "robotics", "computer-use", "compound")
_DEAD_SIGNS = ("404", "no longer available", "not found", "decommissioned", "does not exist",
               "deprecated", "model_not_found")


def _gemini_rank(model_id: str):
    """Newest Gemini first; flash before pro (higher free limits); stable before preview; lite last."""
    m = re.match(r"^gemini-(\d+(?:\.\d+)?)-(flash|pro)(-lite)?(-preview)?(-[\w.-]+)?$", model_id)
    if not m:
        return None
    version, family, lite, preview, suffix = m.groups()
    return (-float(version), family != "flash", bool(lite), bool(preview), bool(suffix), model_id)


def _retry_seconds(err: str) -> Optional[float]:
    """Read 'try again in 7.5s' (Groq) or 'retryDelay': '20s' (Gemini) from a rate-limit error."""
    m = (re.search(r"try again in (?:(\d+)m)?([\d.]+)s", err) or
         re.search(r"retry(?:Delay| in)['\": ]+([\d.]+)s", err))
    if not m:
        return None
    if len(m.groups()) == 2:
        return float(m.group(1) or 0) * 60 + float(m.group(2))
    return float(m.group(1))


def flatten_tool_turns(messages: list) -> list:
    """Turn earlier tool calls and results into plain text, so a different model can continue
    the conversation without needing the first model's tool-call ids or thought signatures."""
    out, names = [], {}
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            calls = []
            for tc in msg["tool_calls"]:
                names[tc["id"]] = tc["function"]["name"]
                calls.append(f"{tc['function']['name']}({tc['function'].get('arguments') or '{}'})")
            text = (msg.get("content") or "") + "\n[Called tools: " + "; ".join(calls) + "]"
            out.append({"role": "assistant", "content": text.strip()})
        elif msg.get("role") == "tool":
            out.append({"role": "user",
                        "content": f"[Result of {names.get(msg.get('tool_call_id'), 'tool')}]: {msg['content']}"})
        else:
            out.append(msg)
    return out


class LLMRouter:
    """Free Gemini and Groq models with failover.
    - picks candidates from each provider's live model list (newest Gemini first)
    - remembers models that are retired (404) and never calls them again
    - keeps one agent run on one model ('owner'); if it must switch, earlier tool turns are
      flattened to text so the new model can continue
    - waits the time a rate-limit error asks for, then moves to the next model"""

    def __init__(self, order=("gemini", "groq"), api_keys: Optional[dict] = None,
                 model_overrides: Optional[dict] = None, verbose: bool = False,
                 min_interval: float = 1.5, max_wait: float = 45):
        from openai import OpenAI
        self.verbose = verbose
        self.min_interval = min_interval     # seconds between calls (free-tier requests per minute)
        self.max_wait = max_wait             # longest rate-limit wait before trying another model
        self.clients = []
        self.errors = []
        self.dead = set()
        self.last_used = None
        self._last_call = 0.0
        for p in order:
            key = (api_keys or {}).get(p) or os.environ.get(PROVIDERS[p]["key_env"])
            if not key:
                continue
            client = OpenAI(api_key=key, base_url=PROVIDERS[p]["base_url"], max_retries=0, timeout=90)
            override = (model_overrides or {}).get(p)
            models = ([override] if isinstance(override, str) else list(override)) if override \
                else self._pick_models(p, client)
            self.clients.append((p, client, models))
            if verbose:
                print(f"{p}: candidate models {models}")
        if not self.clients:
            raise RuntimeError("No API key found. Set GEMINI_API_KEY and/or GROQ_API_KEY.")

    @staticmethod
    def _pick_models(provider, client, k=4):
        try:
            ids = [m.id.replace("models/", "") for m in client.models.list().data]
        except Exception:
            ids = []
        ids = [i for i in ids if not any(x in i.lower() for x in _EXCLUDE)]
        if provider == "gemini":
            ranked = sorted((r for r in map(_gemini_rank, ids) if r), key=lambda r: r[:-1])
            picked = [r[-1] for r in ranked][:k - 1]
            if "gemini-flash-latest" in ids or not ids:
                picked.append("gemini-flash-latest")
            return picked[:k]
        prefer = PROVIDERS[provider]["prefer"]
        picked = [m for m in prefer if m in ids] if ids else list(prefer)
        return (picked or [i for i in ids if "gpt-oss" in i or "llama" in i])[:k]

    def _candidates(self, owner=None):
        cands = [(p, c, m) for p, c, models in self.clients for m in models if f"{p}:{m}" not in self.dead]
        if owner:   # try the model that started this conversation first
            cands.sort(key=lambda x: f"{x[0]}:{x[2]}" != owner)
        return cands

    def chat(self, messages, tools=None, temperature=0.2, owner: Optional[str] = None) -> dict:
        cands = self._candidates(owner)
        if not cands:
            raise RuntimeError("No working models left:\n" + "\n".join(self.errors[-6:]))
        for provider, client, model in cands:
            name = f"{provider}:{model}"
            msgs = messages if (owner is None or owner == name) else flatten_tool_turns(messages)
            for attempt in range(3):
                wait = self.min_interval - (time.time() - self._last_call)
                if wait > 0:
                    time.sleep(wait)
                self._last_call = time.time()
                try:
                    kw = dict(model=model, messages=msgs, temperature=temperature)
                    if tools:
                        kw["tools"] = tools
                        kw["tool_choice"] = "auto"
                    resp = client.chat.completions.create(**kw)
                    msg = resp.choices[0].message
                    calls = []
                    for i, tc in enumerate(msg.tool_calls or []):
                        # model_dump keeps extra fields such as Gemini 3's extra_content.thought_signature,
                        # which must be sent back unchanged in the next request
                        raw = tc.model_dump(exclude_none=True)
                        raw["id"] = raw.get("id") or f"call_{int(time.time() * 1000)}_{i}"
                        raw["type"] = "function"
                        raw["function"]["arguments"] = raw["function"].get("arguments") or "{}"
                        calls.append(raw)
                    self.last_used = name
                    return {"content": msg.content or "", "tool_calls": calls, "model": name,
                            "flattened": msgs is not messages}
                except Exception as e:
                    err = str(e)
                    low = err.lower()
                    self.errors.append(f"{name}: {err[:200]}")
                    if self.verbose:
                        print(f"  ! {name} failed: {err[:120]}")
                    if any(s in low for s in _DEAD_SIGNS):
                        self.dead.add(name)
                        break
                    if "thought_signature" in low and msgs is messages:
                        msgs = flatten_tool_turns(messages)   # retry once without tool-call history
                        continue
                    if "429" in err or "rate limit" in low or "quota" in low or "resource_exhausted" in low:
                        secs = _retry_seconds(err)
                        if secs is None or secs > self.max_wait or attempt == 2:
                            break   # long wait or daily quota: try the next model instead
                        if self.verbose:
                            print(f"    waiting {secs + 1:.0f}s for rate limit")
                        time.sleep(secs + 1)
                        continue
                    break
        raise RuntimeError("All models failed:\n" + "\n".join(self.errors[-6:]))


# %% STEP 5B | The agent loop (reason -> call tools -> observe -> decide)
@dataclass
class AgentConfig:
    as_of: str
    horizon_months: int = 6
    risk_appetite: str = "balanced"          # conservative | balanced | aggressive
    weather_view: str = "Let the agent decide from the data"
    temperature: float = 0.2
    max_steps: int = 10
    enabled_tools: tuple = ("price", "weather", "news", "supply", "margin")


RISK_TEXT = {
    "conservative": "protect margins first; prefer locking in prices when upside price risk is material",
    "balanced": "balance margin protection against the cost of locking in a price that may fall",
    "aggressive": "accept more exposure to spot prices when prices are more likely to fall",
}

OUTPUT_SCHEMA = """{
  "situation": "2-3 sentences on where the cocoa market is today",
  "key_drivers": [{"driver": "short text", "direction": "up|down|neutral", "evidence_ids": ["PRICE"]}],
  "scenarios": [
    {"name": "Base case", "avg_price_usd_t": 0, "probability": 0.5, "rationale": "short text"},
    {"name": "Supply shock (e.g. El Nino / disease)", "avg_price_usd_t": 0, "probability": 0.25, "rationale": "short text"},
    {"name": "Good weather / surplus", "avg_price_usd_t": 0, "probability": 0.25, "rationale": "short text"}
  ],
  "recommended_hedge_pct": 0,
  "confidence": "low|medium|high",
  "recommendation_rationale": "2-4 sentences for the CFO",
  "risks_and_limits": ["short text"],
  "evidence_ids": ["PRICE", "WEATHER"]
}"""


def build_system_prompt(profile: CompanyProfile, cfg: AgentConfig) -> str:
    risk = cfg.risk_appetite if cfg.risk_appetite in RISK_TEXT else "balanced"
    return (
        f"You are a cocoa procurement analyst AI agent working for {profile.name}, a fictional UK chocolate maker.\n"
        f"Today's date is {pd.Timestamp(cfg.as_of).date()}. Treat it as the present: your tools only return "
        f"information published on or before today, and you must NOT use any knowledge of later events.\n\n"
        f"Company profile:\n"
        f"- Cocoa need: {profile.monthly_cocoa_t:,.0f} tonnes per month\n"
        f"- Monthly revenue: ${profile.monthly_revenue_usd:,.0f}; other monthly costs: ${profile.monthly_other_costs_usd:,.0f}\n"
        f"- Already hedged: {profile.current_hedge_pct:.0f}% of the next {cfg.horizon_months} months\n"
        f"- Risk appetite: {risk} ({RISK_TEXT[risk]})\n"
        f"- Weather planning assumption from the user: {cfg.weather_view}\n\n"
        f"Task: decide what share (0-100%) of the next {cfg.horizon_months} months' cocoa need should be bought "
        f"forward now at today's price (the hedge), and explain why to the CFO.\n\n"
        f"How to work:\n"
        f"1. Call the data tools you have (price trend, weather anomaly, news headlines, supply facts).\n"
        f"2. Form three scenarios for the average cocoa price over the next {cfg.horizon_months} months. "
        f"Use calc_margin_impact to test your proposed hedge in at least the highest- and lowest-price scenarios.\n"
        f"3. Then reply with ONLY a JSON object in this format, with no other text:\n{OUTPUT_SCHEMA}\n\n"
        f"Rules:\n"
        f"- Every key driver must cite evidence ids that appeared in tool results (e.g. PRICE, WEATHER, N05, S03).\n"
        f"- Scenario probabilities must sum to 1.\n"
        f"- Be honest about uncertainty; if evidence is mixed, say so and lower your confidence.\n"
        f"- This is a teaching demo with a fictional company, not investment advice."
    )


def extract_json(text: str) -> dict:
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        a, b = text.find("{"), text.rfind("}")
        if a == -1 or b == -1:
            raise ValueError("no JSON object found")
        text = text[a:b + 1]
    return json.loads(text)


def validate_recommendation(rec: dict, tools: AgentTools):
    """Checks the agent's answer. Returns (clean_rec, problems)."""
    problems = []
    required = ["situation", "key_drivers", "scenarios", "recommended_hedge_pct", "confidence",
                "recommendation_rationale"]
    for k in required:
        if k not in rec:
            problems.append(f"missing field '{k}'")
    if problems:
        return None, problems

    try:
        h = float(rec["recommended_hedge_pct"])
    except Exception:
        return None, ["recommended_hedge_pct is not a number"]
    if not 0 <= h <= 100:
        problems.append("recommended_hedge_pct outside 0-100 (clipped)")
    rec["recommended_hedge_pct"] = round(max(0.0, min(100.0, h)), 1)

    scen = [s for s in rec.get("scenarios", []) if isinstance(s, dict)]
    clean = []
    for s in scen:
        try:
            price = float(s.get("avg_price_usd_t", 0))
            prob = float(s.get("probability", 0))
        except Exception:
            continue
        if price > 0 and prob >= 0:
            clean.append({**s, "avg_price_usd_t": price, "probability": prob})
    if len(clean) < 2:
        return None, problems + ["need at least two scenarios with a price and a probability"]
    total = sum(s["probability"] for s in clean)
    if total <= 0:
        return None, problems + ["scenario probabilities sum to zero"]
    if abs(total - 1) > 0.02:
        problems.append(f"probabilities summed to {total:.2f} (normalised)")
    for s in clean:
        s["probability"] = round(s["probability"] / total, 3)
    rec["scenarios"] = clean

    if str(rec.get("confidence", "")).lower() not in ("low", "medium", "high"):
        problems.append("confidence not low/medium/high")

    cited = set(rec.get("evidence_ids", []) or [])
    for d in rec.get("key_drivers", []) or []:
        if isinstance(d, dict):
            cited |= set(d.get("evidence_ids", []) or [])
    cited = {str(c) for c in cited}
    valid = sorted(c for c in cited if c in tools.evidence)
    invalid = sorted(c for c in cited if c not in tools.evidence)
    rec["_evidence_check"] = {
        "cited": sorted(cited), "valid": valid, "unsupported": invalid,
        "valid_ratio": round(len(valid) / len(cited), 2) if cited else 0.0,
        "drivers_without_evidence": sum(1 for d in rec.get("key_drivers", [])
                                        if isinstance(d, dict) and not d.get("evidence_ids")),
    }
    if invalid:
        problems.append(f"unsupported evidence ids: {invalid}")
    return rec, problems


def _preview(result: dict, n: int = 220) -> str:
    s = json.dumps(result, default=str)
    return s if len(s) <= n else s[:n] + "..."


def run_agent(data: DataBundle, profile: CompanyProfile, cfg: AgentConfig, router: LLMRouter,
              verbose: bool = True) -> dict:
    snap = get_snapshot(data, cfg.as_of)
    tools = AgentTools(snap, profile, cfg.horizon_months, cfg.enabled_tools)
    messages = [
        {"role": "system", "content": build_system_prompt(profile, cfg)},
        {"role": "user", "content": f"Decision date {snap.as_of.date()}. Investigate with your tools, "
                                    f"then return the hedge recommendation JSON for the CFO."},
    ]
    trace, final_text = [], None
    owner = None          # the model that owns this conversation's tool-call history
    t0 = time.time()

    def ask(with_tools: bool) -> dict:
        nonlocal owner, messages
        out = router.chat(messages, tools.schemas() if with_tools else None, cfg.temperature, owner=owner)
        if out.get("flattened"):          # a different model took over: keep the text version of history
            messages = flatten_tool_turns(messages)
        owner = out["model"]
        return out

    for step in range(cfg.max_steps):
        out = ask(with_tools=True)
        if out["tool_calls"]:
            messages.append({"role": "assistant", "content": out["content"] or "",
                             "tool_calls": out["tool_calls"]})
            for c in out["tool_calls"]:
                fname = c["function"]["name"]
                try:
                    args = json.loads(c["function"]["arguments"] or "{}") or {}
                except Exception:
                    args = {}
                result = tools.dispatch(fname, args)
                trace.append({"step": step + 1, "tool": fname, "args": args,
                              "result_preview": _preview(result), "model": out["model"]})
                if verbose:
                    print(f"[step {step + 1}] {fname}({args}) -> {_preview(result, 140)}")
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": json.dumps(result, default=str)})
            continue
        if not tools.calls:   # answered without looking at any data: push back
            messages.append({"role": "assistant", "content": out["content"] or ""})
            messages.append({"role": "user", "content": "You have not used any tools yet. Call the tools first; "
                                                        "do not rely on prior knowledge."})
            continue
        final_text = out["content"]
        break

    if final_text is None:
        messages.append({"role": "user", "content": "Stop calling tools. Return the final JSON object only."})
        final_text = ask(with_tools=False)["content"]

    rec, problems = None, []
    for attempt in range(2):
        try:
            rec, problems = validate_recommendation(extract_json(final_text), tools)
        except Exception as e:
            rec, problems = None, [f"invalid JSON: {e}"]
        if rec is not None:
            break
        messages.append({"role": "assistant", "content": final_text})
        messages.append({"role": "user", "content": f"Your answer could not be used: {problems}. "
                                                    f"Return only the corrected JSON object."})
        final_text = ask(with_tools=False)["content"]
    if rec is None:
        raise RuntimeError(f"Agent did not return a valid recommendation: {problems}")

    # Deterministic post-check: recompute margins in code rather than trusting the model's arithmetic
    h = rec["recommended_hedge_pct"]
    locked, pdate, _ = tools.current_price()
    scen_rows = []
    for s in rec["scenarios"]:
        with_h = tools.calc_margin_impact(s["avg_price_usd_t"], h)
        no_h = tools.calc_margin_impact(s["avg_price_usd_t"], 0)
        scen_rows.append({"scenario": s.get("name", ""), "probability": s["probability"],
                          "avg_price_usd_t": round(s["avg_price_usd_t"]),
                          "margin_pct_with_hedge": with_h["gross_margin_pct"],
                          "margin_pct_no_hedge": no_h["gross_margin_pct"],
                          "monthly_margin_with_hedge_usd": with_h["monthly_gross_margin_usd"]})
    expected_price = sum(s["avg_price_usd_t"] * s["probability"] for s in rec["scenarios"])

    return {
        "as_of": str(snap.as_of.date()),
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()},
        "profile": profile.__dict__.copy(),
        "model": router.last_used,
        "recommendation": rec,
        "validation_notes": problems,
        "locked_price_usd_t": round(locked), "locked_price_date": pdate,
        "expected_price_usd_t": round(expected_price),
        "scenario_margins": scen_rows,
        "evidence": list(tools.evidence.values()),
        "trace": trace,
        "tool_calls": len(tools.calls),
        "seconds": round(time.time() - t0, 1),
    }


# %% STEP 6 | CFO memo
def render_memo(res: dict) -> str:
    r = res["recommendation"]
    lines = [
        f"### Cocoa hedge memo - {res['as_of']}",
        f"**To:** CFO, {res['profile']['name']}  ",
        f"**Recommendation:** buy forward **{r['recommended_hedge_pct']:.0f}%** of the next "
        f"{res['config']['horizon_months']} months' cocoa need at about ${res['locked_price_usd_t']:,}/t "
        f"(currently {res['profile']['current_hedge_pct']:.0f}%). Confidence: **{r['confidence']}**.",
        "",
        f"**Situation.** {r['situation']}",
        "",
        "**Key drivers**",
    ]
    for d in r.get("key_drivers", []):
        if isinstance(d, dict):
            ev = ", ".join(d.get("evidence_ids", []) or ["no evidence"])
            lines.append(f"- {d.get('driver', '')} ({d.get('direction', '')}) [{ev}]")
    lines += ["", "**Scenarios (next months' average price)**",
              "| Scenario | Probability | Price $/t | Margin with hedge | Margin unhedged |",
              "|---|---|---|---|---|"]
    for s in res["scenario_margins"]:
        lines.append(f"| {s['scenario']} | {s['probability']:.0%} | {s['avg_price_usd_t']:,} | "
                     f"{s['margin_pct_with_hedge']}% | {s['margin_pct_no_hedge']}% |")
    lines += ["", f"Probability-weighted price: ${res['expected_price_usd_t']:,}/t", "",
              f"**Why.** {r['recommendation_rationale']}", "", "**Risks and limits**"]
    for x in r.get("risks_and_limits", []) or []:
        lines.append(f"- {x}")
    ec = r.get("_evidence_check", {})
    lines += ["", f"_Evidence check: {len(ec.get('valid', []))} of {len(ec.get('cited', []))} cited ids found in "
                  f"tool results. Model: {res['model']}. Fictional company - not investment advice._"]
    return "\n".join(lines)


def render_memo_text(res: dict) -> str:
    """Plain-text memo for email bodies."""
    r = res["recommendation"]
    t = [f"COCOA HEDGE MEMO - {res['as_of']}",
         f"To: CFO, {res['profile']['name']}", "",
         f"RECOMMENDATION: buy forward {r['recommended_hedge_pct']:.0f}% of the next "
         f"{res['config']['horizon_months']} months' cocoa need at about ${res['locked_price_usd_t']:,}/t "
         f"(currently {res['profile']['current_hedge_pct']:.0f}%). Confidence: {r['confidence']}.", "",
         f"SITUATION: {r['situation']}", "", "KEY DRIVERS:"]
    for d in r.get("key_drivers", []):
        if isinstance(d, dict):
            t.append(f"- {d.get('driver', '')} ({d.get('direction', '')}) [{', '.join(d.get('evidence_ids', []) or [])}]")
    t += ["", "SCENARIOS (probability / avg price / margin with hedge / margin unhedged):"]
    for s in res["scenario_margins"]:
        t.append(f"- {s['scenario']}: {s['probability']:.0%} / ${s['avg_price_usd_t']:,} / "
                 f"{s['margin_pct_with_hedge']}% / {s['margin_pct_no_hedge']}%")
    t += [f"Probability-weighted price: ${res['expected_price_usd_t']:,}/t", "",
          f"WHY: {r['recommendation_rationale']}", "", "RISKS AND LIMITS:"]
    t += [f"- {x}" for x in r.get("risks_and_limits", []) or []]
    t += ["", "SOURCES:"]
    t += [f"[{e['id']}] {e['detail']} {e['url']}".strip() for e in res["evidence"] if e["id"] != "CALC"]
    t += ["", f"Model: {res['model']}. Fictional company - teaching demo, not investment advice."]
    return "\n".join(t)


def render_memo_html(res: dict) -> str:
    """Self-contained HTML report for email and download."""
    from html import escape as e
    r = res["recommendation"]
    drivers = "".join(f"<li>{e(str(d.get('driver', '')))} <i>({e(str(d.get('direction', '')))})</i> "
                      f"[{e(', '.join(d.get('evidence_ids', []) or []))}]</li>"
                      for d in r.get("key_drivers", []) if isinstance(d, dict))
    scen = "".join(f"<tr><td>{e(str(s['scenario']))}</td><td>{s['probability']:.0%}</td>"
                   f"<td>${s['avg_price_usd_t']:,}</td><td>{s['margin_pct_with_hedge']}%</td>"
                   f"<td>{s['margin_pct_no_hedge']}%</td></tr>" for s in res["scenario_margins"])
    risks = "".join(f"<li>{e(str(x))}</li>" for x in r.get("risks_and_limits", []) or [])
    srcs = "".join(f"<li><b>{e(x['id'])}</b> {e(x['detail'])}"
                   + (f' <a href="{e(x["url"])}">link</a>' if x["url"] else "") + "</li>"
                   for x in res["evidence"] if x["id"] != "CALC")
    td = "border:1px solid #ccc;padding:6px 10px;text-align:left"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Cocoa hedge memo {res['as_of']}</title></head>
<body style="font-family:Arial,sans-serif;max-width:760px;margin:auto;color:#222;line-height:1.5">
<h2 style="margin-bottom:4px">Cocoa hedge memo &ndash; {res['as_of']}</h2>
<p style="margin-top:0;color:#555">To: CFO, {e(res['profile']['name'])}</p>
<div style="background:#f3efe8;padding:12px 16px;border-left:4px solid #6b4226">
<b>Recommendation:</b> buy forward <b>{r['recommended_hedge_pct']:.0f}%</b> of the next {res['config']['horizon_months']}
months' cocoa need at about ${res['locked_price_usd_t']:,}/t (currently {res['profile']['current_hedge_pct']:.0f}%).
Confidence: <b>{e(str(r['confidence']))}</b>.</div>
<h3>Situation</h3><p>{e(str(r['situation']))}</p>
<h3>Key drivers</h3><ul>{drivers}</ul>
<h3>Scenarios</h3>
<table style="border-collapse:collapse"><tr><th style="{td}">Scenario</th><th style="{td}">Probability</th>
<th style="{td}">Avg price $/t</th><th style="{td}">Margin with hedge</th><th style="{td}">Margin unhedged</th></tr>
{scen.replace('<td>', f'<td style="{td}">')}</table>
<p>Probability-weighted price: ${res['expected_price_usd_t']:,}/t</p>
<h3>Why</h3><p>{e(str(r['recommendation_rationale']))}</p>
<h3>Risks and limits</h3><ul>{risks}</ul>
<h3>Sources</h3><ol style="font-size:13px">{srcs}</ol>
<p style="font-size:12px;color:#777">Generated by the Cocoa Procurement Agent ({e(str(res['model']))}).
Fictional company &ndash; teaching demo, not investment advice.</p></body></html>"""


# %% STEP 7 | Evaluation: rules baseline, cost simulation, backtest
def rules_baseline(tools: AgentTools) -> dict:
    """A transparent non-AI rule to compare against: momentum + weather + price level."""
    pt = tools.get_price_trend()
    wa = tools.get_weather_anomaly() if "weather" in tools.enabled else {}
    hedge, why = 50.0, ["start at 50%"]
    c3 = pt.get("change_3m_pct", 0)
    if c3 > 10:
        hedge += 20; why.append(f"3-month price change {c3}% > +10%: +20")
    elif c3 < -10:
        hedge -= 20; why.append(f"3-month price change {c3}% < -10%: -20")
    rain = wa.get("avg_rain_anomaly_pct")
    if rain is not None and rain < -15:
        hedge += 15; why.append(f"rain {rain}% below normal: +15")
    if pt.get("percentile_vs_last_10y", 0) >= 90:
        hedge -= 15; why.append("price in top 10% of last 10 years (reversal risk): -15")
    return {"hedge_pct": max(0.0, min(100.0, hedge)), "why": why}


def actual_prices_after(data: DataBundle, as_of, horizon: int) -> pd.Series:
    start = pd.Timestamp(as_of).to_period("M") + 1
    months = [start + i for i in range(horizon)]
    mp = data.monthly_price
    return mp[mp.index.isin(months)]


def simulate_cost(actual: pd.Series, locked_price: float, hedge_pct: float, monthly_t: float) -> float:
    """Hedged share bought at today's price; the rest bought each month at that month's actual price.
    Simplification: today's futures price is used as the forward price."""
    h = hedge_pct / 100
    return float(sum(monthly_t * (h * locked_price + (1 - h) * p) for p in actual.values))


def run_backtest(data: DataBundle, profile: CompanyProfile, router: Optional[LLMRouter],
                 dates=("2024-02-15", "2024-12-01", "2026-04-01"), horizon: int = 6, runs: int = 1,
                 verbose: bool = True, **cfg_kwargs):
    rows, agent_results = [], []
    for d in dates:
        snap = get_snapshot(data, d)
        tools = AgentTools(snap, profile, horizon)
        locked, _, _ = tools.current_price()
        actual = actual_prices_after(data, d, horizon)
        if actual.empty:
            print(f"{d}: no actual prices yet for the next {horizon} months - skipped")
            continue
        strategies = {"No hedge (buy monthly)": 0.0, "Fixed 50%": 50.0, "Full hedge": 100.0,
                      "Rules baseline": rules_baseline(tools)["hedge_pct"]}
        hedges = []
        if router is not None:
            for r in range(runs):
                print(f"=== {d} | run {r + 1}/{runs} ===")
                try:
                    res = run_agent(data, profile, AgentConfig(as_of=d, horizon_months=horizon, **cfg_kwargs),
                                    router, verbose=verbose)
                except Exception as e:   # one failed run should not stop the whole backtest
                    print(f"   run failed: {str(e)[:300]}")
                    continue
                agent_results.append(res)
                hedges.append(res["recommendation"]["recommended_hedge_pct"])
                print(f"   hedge {hedges[-1]:.0f}% | {res['model']} | {res['tool_calls']} tool calls | {res['seconds']}s")
            if hedges:
                strategies["AI agent"] = float(np.mean(hedges))
        costs = {k: simulate_cost(actual, locked, h, profile.monthly_cocoa_t) for k, h in strategies.items()}
        best = min(costs["No hedge (buy monthly)"], costs["Full hedge"])   # best possible in hindsight
        for k, h in strategies.items():
            rows.append({"decision_date": d, "strategy": k, "hedge_pct": round(h, 1),
                         "price_at_decision": round(locked),
                         "avg_actual_price_next_months": round(float(actual.mean())),
                         "months_evaluated": len(actual),
                         "total_cost_usd": round(costs[k]),
                         "avg_price_paid_usd_t": round(costs[k] / (profile.monthly_cocoa_t * len(actual))),
                         "regret_vs_hindsight_usd": round(costs[k] - best),
                         "agent_runs_hedge_pct": hedges if k == "AI agent" else None})
    return pd.DataFrame(rows), agent_results


def summarise_checks(agent_results: list) -> pd.DataFrame:
    """Evidence and consistency checks per decision date."""
    rows = []
    for res in agent_results:
        ec = res["recommendation"].get("_evidence_check", {})
        rows.append({"as_of": res["as_of"], "hedge_pct": res["recommendation"]["recommended_hedge_pct"],
                     "confidence": res["recommendation"]["confidence"], "model": res["model"],
                     "tool_calls": res["tool_calls"], "evidence_valid_ratio": ec.get("valid_ratio"),
                     "unsupported_ids": len(ec.get("unsupported", [])),
                     "validation_notes": len(res["validation_notes"])})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    cons = df.groupby("as_of")["hedge_pct"].agg(["mean", "std", "min", "max"]).round(1)
    cons.columns = ["hedge_mean", "hedge_std", "hedge_min", "hedge_max"]
    return df.merge(cons, left_on="as_of", right_index=True)


# %% STEP 8 | Charts
def plot_price_history(data: DataBundle, decision_dates=(), start="2022-01-01", ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4.5))
    else:
        fig = ax.figure
    if data.daily is not None:
        s = data.daily[data.daily.index >= start]
        ax.plot(s.index, s.values, lw=1, label="ICE cocoa futures (daily)")
    m = data.monthly_price[data.monthly_price.index >= pd.Period(start, "M")]
    ax.plot(m.index.to_timestamp(), m.values, lw=2, label="Monthly average")
    for d in decision_dates:
        ax.axvline(pd.Timestamp(d), ls="--", lw=1, color="grey")
        ax.text(pd.Timestamp(d), ax.get_ylim()[1] * 0.97, f" {d}", fontsize=8, va="top")
    ax.set_ylabel("USD per tonne")
    ax.set_title("Cocoa price and decision dates")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_backtest(bt: pd.DataFrame):
    import matplotlib.pyplot as plt
    piv = bt.pivot(index="decision_date", columns="strategy", values="avg_price_paid_usd_t")
    order = [c for c in ["No hedge (buy monthly)", "Fixed 50%", "Full hedge", "Rules baseline", "AI agent"] if c in piv]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    piv[order].plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("Average price paid, USD per tonne (lower is better)")
    ax.set_xlabel("Decision date")
    ax.set_title("Backtest: what each strategy actually paid over the following months")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    fig.tight_layout()
    return fig


# %% STEP 9 | Save a live run with a timestamp (for the paper's follow-up check)
def save_run(res: dict, folder: str = "outputs") -> str:
    os.makedirs(folder, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(folder, f"agent_run_{res['as_of']}_{stamp}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2, default=str)
    return path
