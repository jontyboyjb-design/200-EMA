"""
ChartFlow — EMA Box Screener (Bitget / Bybit / OKX edition)
─────────────────────────────────────────────────────────
Run with:   streamlit run ema_box_screener_streamlit.py

Logic  : Price is INSIDE the band  [ EMA × (1 - dn%) , EMA × (1 + up%) ]
Columns: Exchange | TF | Symbol | Type | Price | EMA | % from EMA | Chart
Movers : Spot movers per exchange + US Stock movers
"""

import time
import textwrap
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import ccxt
import yfinance as yf
import pandas as pd
import streamlit as st

# Optional Telegram alert hook — only used if the module is present locally.
try:
    from telegram_alerts import send_box_alerts
except Exception:
    send_box_alerts = None

# Optional auto-refresh helper (pip install streamlit-autorefresh) — the app
# works fine without it, you'll just need to click "Scan Now" to refresh.
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False


# ══════════════════════════════════════════════════════════════════════════════
#  Exchange config — Bitget, Bybit, OKX
# ══════════════════════════════════════════════════════════════════════════════

EXCHANGE_META = {
    "bitget": {"label": "Bitget Spot", "tv_prefix": "BITGET", "ccxt_id": "bitget", "options": {}},
    "bybit":  {"label": "Bybit Spot",  "tv_prefix": "BYBIT",  "ccxt_id": "bybit",  "options": {"defaultType": "spot"}},
    "okx":    {"label": "OKX Spot",    "tv_prefix": "OKX",    "ccxt_id": "okx",    "options": {"defaultType": "spot"}},
}
EXCHANGE_ORDER = ["bitget", "bybit", "okx"]


@st.cache_resource(show_spinner=False)
def get_exchange_client(exchange_id: str):
    meta = EXCHANGE_META[exchange_id]
    klass = getattr(ccxt, meta["ccxt_id"])
    params = {"enableRateLimit": True, "timeout": 20000}
    if meta["options"]:
        params["options"] = meta["options"]
    return klass(params)


# ── Timeframe lists ────────────────────────────────────────────────────────────
CRYPTO_TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "3d", "1w"]
STOCK_TIMEFRAMES = ["1m", "5m", "15m", "1h", "1d", "1wk", "1mo"]

YF_INTERVAL_MAP = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "1d": "1d", "1wk": "1wk", "1mo": "1mo"}
YF_PERIOD_MAP = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d", "1d": "5y", "1wk": "5y", "1mo": "10y"}

AUTO_SCAN_SECONDS = 3 * 60   # 3 minutes

# ── US Stock universe — AUTO-POPULATED (no manual ticker list) ────────────────
_FALLBACK_STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK-B", "LLY", "AVGO",
    "JPM", "TSLA", "UNH", "V", "XOM", "MA", "COST", "HD", "PG", "WMT",
]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_us_stock_universe(limit=100):
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        tickers = (
            df["Symbol"].astype(str).str.strip()
            .str.replace(".", "-", regex=False)   # BRK.B -> BRK-B (Yahoo Finance format)
            .tolist()
        )
        if tickers:
            return tickers[:limit]
    except Exception:
        pass
    return _FALLBACK_STOCKS[:limit]


# ══════════════════════════════════════════════════════════════════════════════
#  Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def fetch_candles_crypto(client, symbol, timeframe, limit=300, retries=3):
    for attempt in range(retries):
        try:
            raw = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not raw or len(raw) < 50:
                return None
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = df[c].astype(float)
            return df.reset_index(drop=True)
        except Exception:
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
            else:
                return None


def fetch_candles_stock(symbol, timeframe):
    try:
        interval = YF_INTERVAL_MAP.get(timeframe)
        period = YF_PERIOD_MAP.get(timeframe)
        if not interval or not period:
            return None
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df is None or len(df) < 50:
            return None
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"date": "timestamp", "datetime": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        return df[["timestamp", "open", "high", "low", "close", "volume"]].tail(300).reset_index(drop=True)
    except Exception:
        return None


# ── Market helpers ─────────────────────────────────────────────────────────────

def get_quote_volume(ticker):
    qv = ticker.get("quoteVolume")
    if qv is not None:
        return float(qv or 0)
    bv = ticker.get("baseVolume")
    lp = ticker.get("last") or ticker.get("close")
    if bv is not None and lp is not None:
        return float(bv or 0) * float(lp or 0)
    return 0.0


# ── Stablecoin exclusion ────────────────────────────────────────────────────
STABLECOIN_BASES = {
    "USDC", "USDT", "BUSD", "FDUSD", "TUSD", "USDP", "DAI", "GUSD", "USDD", "PYUSD",
    "USDE", "FRAX", "LUSD", "SUSD", "SUSDE", "USTC", "UST", "MIM", "USDS", "PAX",
    "HUSD", "OUSD", "USDX", "EUROC", "EURT", "EURS", "CUSD", "USD1", "AUSD",
}


def is_stablecoin_pair(symbol):
    base = symbol.split("/")[0].upper()
    return base in STABLECOIN_BASES


# ── Tokenized-stock exclusion (Bitget "Stocks 2.0" RWA program only) ─────────
# Bitget lists hundreds of tokenized stocks/ETFs (e.g. RCOST/USDT) as ordinary
# USDT spot pairs, all systematically prefixed with "R". This heuristic is
# Bitget-specific — Bybit/OKX don't run this program, so it's skipped there.
PROTECTED_R_CRYPTO_TICKERS = {
    "RNDR", "RENDER", "RUNE", "RVN", "RAY", "ROSE", "RSR", "REN", "RLC", "RACA",
    "RDNT", "RPL", "REQ", "RIF", "RBN", "RSS3", "RARE", "RGT", "REZ", "RAD", "RON",
    "RFOX", "RAMP", "RACE", "REI", "RIO", "RBTC", "RDAO", "RAIL", "RIDE", "RFUEL",
    "RAIN", "RDPX", "RWA", "RBX", "RING", "RIZON", "ROOT", "ROUTE", "REVV", "RFR",
    "RIN", "RZR", "RAID", "RDC", "RBC", "RSC", "RGB", "RPLS",
}


def is_tokenized_stock_pair(symbol, exchange_id, market=None):
    if exchange_id != "bitget":
        return False

    base = symbol.split("/")[0].upper()
    if base in PROTECTED_R_CRYPTO_TICKERS:
        return False

    if market:
        info = market.get("info", {}) or {}
        for key, val in info.items():
            if "RWA" in str(key).upper():
                if str(val).upper() in ("YES", "TRUE", "1", "Y"):
                    return True

    if len(base) >= 2 and base[0] == "R" and base[1:].isalpha():
        return True

    return False


def is_market_live(market, ticker):
    if market.get("active", True) is False:
        return False

    info = market.get("info", {}) or {}
    status = str(info.get("status") or info.get("symbolStatus") or info.get("state") or "").upper()
    bad_statuses = {"BREAK", "HALT", "HALTED", "OFFLINE", "DELISTED", "SUSPEND", "SUSPENDED", "CLOSE", "CLOSED", "PAUSE", "PAUSED"}
    if status and status in bad_statuses:
        return False

    if not ticker:
        return False
    last = ticker.get("last") or ticker.get("close")
    if not last or float(last) <= 0:
        return False
    if get_quote_volume(ticker) <= 0:
        return False

    return True


def get_ticker_percentage(ticker):
    for key in ("percentage", "changePercentage"):
        v = ticker.get(key)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    info = ticker.get("info", {}) or {}
    for key in ("priceChangePercent", "change24h", "changeUtc24h"):
        v = info.get(key)
        if v is not None:
            try:
                pct = float(v)
                return pct * 100 if abs(pct) <= 1 else pct
            except Exception:
                pass
    lp = ticker.get("last") or ticker.get("close")
    op = ticker.get("open") or ticker.get("previousClose")
    try:
        if lp and op:
            return (float(lp) - float(op)) / float(op) * 100
    except Exception:
        pass
    return None


@st.cache_data(ttl=90, show_spinner=False)
def fetch_markets_and_tickers(exchange_id):
    """Cached (90s) markets + tickers snapshot for one exchange."""
    client = get_exchange_client(exchange_id)
    markets = client.load_markets()
    tickers = client.fetch_tickers()
    return markets, tickers


def get_all_pairs(exchange_id):
    try:
        markets, tickers = fetch_markets_and_tickers(exchange_id)
        return sorted([
            s for s in markets
            if s.endswith("/USDT")
            and not is_stablecoin_pair(s)
            and not is_tokenized_stock_pair(s, exchange_id, markets[s])
            and is_market_live(markets[s], tickers.get(s))
        ])
    except Exception:
        return []


def get_top_pairs(exchange_id, limit=200):
    try:
        markets, tickers = fetch_markets_and_tickers(exchange_id)
        pairs = [
            s for s in markets
            if s.endswith("/USDT")
            and not is_stablecoin_pair(s)
            and not is_tokenized_stock_pair(s, exchange_id, markets[s])
            and is_market_live(markets[s], tickers.get(s))
        ]
        pairs.sort(key=lambda s: get_quote_volume(tickers.get(s, {})), reverse=True)
        return pairs[:limit]
    except Exception:
        return get_all_pairs(exchange_id)[:limit]


def tradingview_url(result):
    if result.get("AssetType") == "stock":
        return f"https://www.tradingview.com/chart/?symbol={result['Symbol']}"
    tv_prefix = EXCHANGE_META[result["Exchange ID"]]["tv_prefix"]
    return f"https://www.tradingview.com/chart/?symbol={tv_prefix}:{result['Symbol'].replace('/', '')}"


def fmt_price(price):
    if price >= 1000:
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:.4f}"
    return f"{price:.6f}"


# ══════════════════════════════════════════════════════════════════════════════
#  Core scan logic — price inside EMA band
# ══════════════════════════════════════════════════════════════════════════════

def analyze_ema_box(job, ema_period, up_pct, dn_pct):
    """job = (symbol, timeframe, asset_type, exchange_id)"""
    symbol, timeframe, asset_type, exchange_id = job
    try:
        if asset_type == "crypto":
            client = get_exchange_client(exchange_id)
            df = fetch_candles_crypto(client, symbol, timeframe)
            exchange_label = EXCHANGE_META[exchange_id]["label"]
        else:
            df = fetch_candles_stock(symbol, timeframe)
            exchange_label = "US Stock"
            exchange_id = "us_stock"

        if df is None or len(df) < max(ema_period, 20):
            return None

        df["ema"] = calc_ema(df["close"], ema_period)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        if len(df) < 5:
            return None

        last_close = float(df["close"].iloc[-1])
        last_ema = float(df["ema"].iloc[-1])

        if last_ema <= 0:
            return None

        upper_band = last_ema * (1 + up_pct / 100)
        lower_band = last_ema * (1 - dn_pct / 100)

        if not (lower_band <= last_close <= upper_band):
            return None

        pct_from_ema = (last_close - last_ema) / last_ema * 100

        return {
            "Exchange": exchange_label,
            "Exchange ID": exchange_id,
            "Timeframe": timeframe,
            "Symbol": symbol,
            "AssetType": asset_type,
            "Price": last_close,
            "EMA": last_ema,
            "UpperBand": upper_band,
            "LowerBand": lower_band,
            "PctFromEMA": round(pct_from_ema, 3),
        }
    except Exception:
        return None


def run_ema_box_scan(jobs, ema_period, up_pct, dn_pct, progress_cb=None):
    results = []
    total = len(jobs) or 1

    def scan_one(job):
        return analyze_ema_box(job, ema_period, up_pct, dn_pct)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(scan_one, job): job for job in jobs}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            if progress_cb:
                progress_cb(done_count, total)
            result = future.result()
            if result is not None:
                results.append(result)

    results.sort(key=lambda x: abs(x.get("PctFromEMA", 0)))

    if send_box_alerts:
        try:
            alert_list = []
            for r in results:
                label = "Above" if r["PctFromEMA"] >= 0 else "Below"
                alert_list.append({
                    "Exchange": r["Exchange"],
                    "Exchange ID": r["Exchange ID"],
                    "Timeframe": r["Timeframe"],
                    "Symbol": r["Symbol"],
                    "AssetType": r["AssetType"],
                    "Direction": f"{label} EMA ({r['PctFromEMA']:+.2f}%)",
                    "Candles Since": 0,
                })
            if alert_list:
                send_box_alerts(alert_list, EXCHANGE_META)
        except Exception:
            pass

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Movers helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_exchange_mover_rows(exchange_id, limit=20):
    markets, tickers = fetch_markets_and_tickers(exchange_id)

    rows = []
    for symbol, ticker in tickers.items():
        if not symbol.endswith("/USDT"):
            continue
        if is_stablecoin_pair(symbol):
            continue
        market = markets.get(symbol)
        if not market or is_tokenized_stock_pair(symbol, exchange_id, market):
            continue
        if not is_market_live(market, ticker):
            continue
        pct = get_ticker_percentage(ticker)
        if pct is not None:
            rows.append({"symbol": symbol, "pct": pct})

    gainers = sorted(rows, key=lambda x: x["pct"], reverse=True)[:limit]
    losers = sorted(rows, key=lambda x: x["pct"])[:limit]
    for idx, r in enumerate(gainers, 1):
        r["rank"] = idx
    for idx, r in enumerate(losers, 1):
        r["rank"] = idx
    return gainers, losers


@st.cache_data(ttl=90, show_spinner=False)
def get_us_stock_mover_rows(limit=20):
    rows = []
    stock_list = get_us_stock_universe(100)
    try:
        raw = yf.download(
            " ".join(stock_list),
            period="2d", interval="1d",
            progress=False, group_by="ticker", auto_adjust=True,
        )
        for sym in stock_list:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    closes = raw[sym]["Close"].dropna()
                else:
                    closes = raw["Close"].dropna()
                if len(closes) >= 2:
                    pct = (closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2] * 100
                    rows.append({"symbol": sym, "pct": round(float(pct), 2)})
            except Exception:
                pass
    except Exception:
        pass
    gainers = sorted(rows, key=lambda x: x["pct"], reverse=True)[:limit]
    losers = sorted(rows, key=lambda x: x["pct"])[:limit]
    for idx, r in enumerate(gainers, 1):
        r["rank"] = idx
    for idx, r in enumerate(losers, 1):
        r["rank"] = idx
    return gainers, losers


# ══════════════════════════════════════════════════════════════════════════════
#  Theme — light, 3D cards, soft shadows, Roboto Mono
# ══════════════════════════════════════════════════════════════════════════════

def inject_theme():
    css = textwrap.dedent("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
      :root{
        --bg:#eef1f7;
        --bg-grad: radial-gradient(circle at 10% 0%, #f7f9fd 0%, #e7ebf4 45%, #dfe4f0 100%);
        --card:#ffffff;
        --card-border: rgba(15,23,42,0.06);
        --shadow-sm: 0 1px 2px rgba(15,23,42,0.05), 0 1px 1px rgba(15,23,42,0.03);
        --shadow-md: 0 10px 24px rgba(30,41,59,0.08), 0 2px 6px rgba(30,41,59,0.05);
        --shadow-lg: 0 18px 40px rgba(30,41,59,0.12), 0 4px 10px rgba(30,41,59,0.06);
        --text:#1e2433;
        --muted:#6b7488;
        --faint:#98a2b8;
        --teal:#0d9488;
        --teal-soft:#e6f7f5;
        --indigo:#6366f1;
        --indigo-soft:#eef0fe;
        --rose:#e11d48;
        --rose-soft:#fdedf1;
      }

      html, body, [class*="css"]  {
        font-family: 'Roboto Mono', monospace !important;
      }

      .stApp{
        background: var(--bg-grad);
        color: var(--text);
      }

      #MainMenu, footer, header {visibility:hidden;}

      section[data-testid="stSidebar"]{
        background: linear-gradient(180deg, #f8faff 0%, #eef1f8 100%);
        border-right: 1px solid var(--card-border);
      }

      .cf-card{
        background: var(--card);
        border-radius: 16px;
        border: 1px solid var(--card-border);
        box-shadow: var(--shadow-md);
        padding: 18px 20px;
        margin-bottom: 16px;
        transition: box-shadow .2s ease, transform .2s ease;
      }
      .cf-card:hover{ box-shadow: var(--shadow-lg); transform: translateY(-1px); }

      .cf-card-flat{
        background: var(--card);
        border-radius: 14px;
        border: 1px solid var(--card-border);
        box-shadow: var(--shadow-sm);
        padding: 14px 16px;
        margin-bottom: 14px;
      }

      .cf-header{
        font-size: 22px; font-weight: 800; letter-spacing: .5px;
        background: linear-gradient(90deg,var(--teal),var(--indigo));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
      }
      .cf-subheader{
        font-size: 10.5px; color: var(--faint); letter-spacing: 3px; font-weight: 700;
      }

      .cf-section-title{
        font-size: 10.5px; color: var(--indigo); font-weight: 800;
        letter-spacing: 1.5px; margin-bottom: 10px; text-transform: uppercase;
      }

      .cf-pill{
        display:inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 11px; font-weight: 700; box-shadow: var(--shadow-sm);
      }
      .cf-pill-up{ background: var(--teal-soft); color: var(--teal); }
      .cf-pill-down{ background: var(--rose-soft); color: var(--rose); }
      .cf-pill-type-crypto{ background: var(--indigo-soft); color: var(--indigo); }
      .cf-pill-type-stock{ background: #fff4e0; color: #c2760c; }

      table.cf-table{ width:100%; border-collapse: collapse; font-size: 13px; }
      table.cf-table thead th{
        text-align:left; font-size:10px; color: var(--faint); font-weight:800;
        letter-spacing: 1px; padding: 10px 12px; border-bottom: 1px solid var(--card-border);
        text-transform: uppercase;
      }
      table.cf-table tbody td{
        padding: 10px 12px; border-bottom: 1px solid #f1f3f9; color: var(--text);
      }
      table.cf-table tbody tr:hover{ background: #f8faff; }
      table.cf-table a.cf-chart-link{
        color: var(--teal); text-decoration:none; font-weight:700; font-size:12px;
        background: var(--teal-soft); padding: 4px 10px; border-radius: 6px;
        border: 1px solid rgba(13,148,136,0.15); box-shadow: var(--shadow-sm);
      }

      .stButton>button{
        font-family:'Roboto Mono', monospace; font-weight:800; letter-spacing:1px;
        border-radius: 12px; border: 1px solid rgba(13,148,136,0.2);
        background: linear-gradient(180deg,#0fb3a1,#0d9488);
        color:white; box-shadow: var(--shadow-md); padding: 10px 0;
      }
      .stButton>button:hover{ box-shadow: var(--shadow-lg); transform: translateY(-1px); }

      .cf-progress-label{ font-size:12px; color: var(--muted); margin-bottom:6px; }
    </style>
    """)
    # Streamlit's markdown parser splits on blank lines and only applies
    # unsafe_allow_html to the first block — strip blank lines so the whole
    # thing renders as one continuous HTML block.
    css = "\n".join(line for line in css.splitlines() if line.strip() != "")
    st.markdown(css, unsafe_allow_html=True)


# ── HTML render helpers ─────────────────────────────────────────────────────

def render_results_table(results):
    if not results:
        return '<div class="cf-card-flat" style="text-align:center;color:var(--muted);font-size:13px;">No tickers matched. Try widening the Upper / Lower % range.</div>'

    rows_html = []
    for r in results:
        url = tradingview_url(r)
        dist = r["PctFromEMA"]
        pill_dir = "cf-pill-up" if dist >= 0 else "cf-pill-down"
        is_stock = r.get("AssetType") == "stock"
        type_pill = "cf-pill-type-stock" if is_stock else "cf-pill-type-crypto"
        type_label = "Stock" if is_stock else "Crypto"

        rows_html.append(f"""
        <tr>
          <td>{r['Exchange']}</td>
          <td style="text-align:center;color:var(--indigo);">{r['Timeframe']}</td>
          <td style="font-weight:800;">{r['Symbol']}</td>
          <td style="text-align:center;"><span class="cf-pill {type_pill}">{type_label}</span></td>
          <td style="text-align:right;">{fmt_price(r['Price'])}</td>
          <td style="text-align:right;color:var(--indigo);">{fmt_price(r['EMA'])}</td>
          <td style="text-align:center;"><span class="cf-pill {pill_dir}">{dist:+.3f}%</span></td>
          <td style="text-align:center;"><a class="cf-chart-link" href="{url}" target="_blank">Chart</a></td>
        </tr>""")

    return f"""
    <div class="cf-card">
      <table class="cf-table">
        <thead><tr>
          <th>Exchange</th><th>TF</th><th>Symbol</th><th>Type</th>
          <th style="text-align:right;">Price</th><th style="text-align:right;">EMA</th>
          <th style="text-align:center;">% from EMA</th><th style="text-align:center;">Chart</th>
        </tr></thead>
        <tbody>{"".join(rows_html)}</tbody>
      </table>
    </div>
    """


def render_mover_table(title, rows, positive, is_stock, tv_prefix=None):
    color_class = "cf-pill-up" if positive else "cf-pill-down"
    if not rows:
        return f"""
        <div class="cf-card-flat">
          <div class="cf-section-title">{title}</div>
          <div style="font-size:12px;color:var(--faint);">No data</div>
        </div>"""

    def make_href(sym):
        if is_stock:
            return f"https://www.tradingview.com/chart/?symbol={sym}"
        return f"https://www.tradingview.com/chart/?symbol={tv_prefix}:{sym.replace('/', '')}"

    body = "".join(f"""
        <tr>
          <td style="color:var(--faint);">{r['rank']}</td>
          <td><a href="{make_href(r['symbol'])}" target="_blank" style="color:var(--text);font-weight:700;text-decoration:none;">{r['symbol']}</a></td>
          <td style="text-align:right;"><span class="cf-pill {color_class}">{r['pct']:+.2f}%</span></td>
        </tr>""" for r in rows)

    return f"""
    <div class="cf-card-flat">
      <div class="cf-section-title">{title}</div>
      <table class="cf-table">
        <thead><tr><th>Rank</th><th>Ticker</th><th style="text-align:right;">%</th></tr></thead>
        <tbody>{body}</tbody>
      </table>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
#  Streamlit app
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="ChartFlow — EMA Box Screener", page_icon="▦", layout="wide")
inject_theme()

if "results" not in st.session_state:
    st.session_state.results = []
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = None
if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="cf-header">▦ CHARTFLOW</div>', unsafe_allow_html=True)
    st.markdown('<div class="cf-subheader">EMA BOX SCREENER</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="cf-card"><div class="cf-section-title">Exchanges</div>', unsafe_allow_html=True)
    selected_exchanges = st.multiselect(
        "Crypto exchanges",
        options=EXCHANGE_ORDER,
        default=EXCHANGE_ORDER,
        format_func=lambda x: EXCHANGE_META[x]["label"],
        label_visibility="collapsed",
    )
    universe_scope = st.radio(
        "Pair scope",
        options=["top200", "all"],
        format_func=lambda x: "Top 200 by volume" if x == "top200" else "All pairs",
        horizontal=True,
        label_visibility="collapsed",
    )
    include_stocks = st.checkbox("Include Top 100 US Stocks", value=False)
    st.markdown("</div>", unsafe_allow_html=True)

    if selected_exchanges:
        st.markdown('<div class="cf-card"><div class="cf-section-title">Crypto Timeframe</div>', unsafe_allow_html=True)
        crypto_tf = st.selectbox("Crypto TF", CRYPTO_TIMEFRAMES, index=CRYPTO_TIMEFRAMES.index("1d"), label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        crypto_tf = "1d"

    if include_stocks:
        st.markdown('<div class="cf-card"><div class="cf-section-title">Stock Timeframe</div>', unsafe_allow_html=True)
        stock_tf = st.selectbox("Stock TF", STOCK_TIMEFRAMES, index=STOCK_TIMEFRAMES.index("1d"), label_visibility="collapsed")
        st.caption("1m = last 7d · 5m/15m = last 60d")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        stock_tf = "1d"

    st.markdown('<div class="cf-card"><div class="cf-section-title">EMA Settings</div>', unsafe_allow_html=True)
    ema_period = st.number_input("EMA Period", min_value=1, value=200, step=1)
    st.caption("Price must be inside EMA ± % band to match")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="cf-card"><div class="cf-section-title">EMA Box Range</div>', unsafe_allow_html=True)
    up_pct = st.number_input("Upper Band % (above EMA)", min_value=0.01, max_value=50.0, value=2.0, step=0.1)
    dn_pct = st.number_input("Lower Band % (below EMA)", min_value=0.01, max_value=50.0, value=2.0, step=0.1)
    st.markdown(
        f'<div style="margin-top:6px;font-size:11px;">'
        f'<span class="cf-pill cf-pill-up">+{up_pct:.1f}%</span> '
        f'&nbsp;←&nbsp; EMA {int(ema_period)} &nbsp;→&nbsp; '
        f'<span class="cf-pill cf-pill-down">-{dn_pct:.1f}%</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    auto_scan = st.checkbox("Auto-scan every 3 minutes", value=False, disabled=not HAS_AUTOREFRESH)
    if not HAS_AUTOREFRESH:
        st.caption("Install `streamlit-autorefresh` to enable auto-scan.")

    scan_clicked = st.button("🔍  SCAN NOW", use_container_width=True)

    st.markdown(
        '<div style="margin-top:10px;font-size:10px;color:var(--faint);line-height:1.6;">'
        'Scans for tickers where price is inside EMA ± % band.</div>',
        unsafe_allow_html=True,
    )

# ── Auto-refresh trigger ────────────────────────────────────────────────────
due_for_autoscan = False
if HAS_AUTOREFRESH and auto_scan:
    st_autorefresh(interval=15_000, key="cf_autorefresh")
    now = time.time()
    if st.session_state.last_scan_time is None or (now - st.session_state.last_scan_time) >= AUTO_SCAN_SECONDS:
        due_for_autoscan = True

# ── Main content ─────────────────────────────────────────────────────────────
st.markdown('<div class="cf-header">EMA Box Screener</div>', unsafe_allow_html=True)
st.markdown('<div class="cf-subheader">BITGET · BYBIT · OKX · US STOCKS</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

should_scan = scan_clicked or due_for_autoscan

if should_scan:
    if not selected_exchanges and not include_stocks:
        st.warning("Select at least one exchange or include US stocks before scanning.")
    else:
        jobs = []
        with st.spinner("Building ticker universe..."):
            for ex_id in selected_exchanges:
                pairs = get_all_pairs(ex_id) if universe_scope == "all" else get_top_pairs(ex_id, 200)
                for pair in pairs:
                    jobs.append((pair, crypto_tf, "crypto", ex_id))
            if include_stocks:
                for sym in get_us_stock_universe(100):
                    jobs.append((sym, stock_tf, "stock", None))

        progress_label = st.empty()
        progress_bar = st.progress(0)

        def update_progress(done, total):
            pct = int(done / total * 100)
            progress_label.markdown(f'<div class="cf-progress-label">Scanning... {done}/{total} ({pct}%)</div>', unsafe_allow_html=True)
            progress_bar.progress(pct)

        results = run_ema_box_scan(jobs, int(ema_period), float(up_pct), float(dn_pct), progress_cb=update_progress)

        st.session_state.results = results
        st.session_state.last_scan_time = time.time()
        st.session_state.scan_count += 1

        progress_label.empty()
        progress_bar.empty()

# ── Results ──────────────────────────────────────────────────────────────────
results = st.session_state.results
count = len(results)

meta_cols = st.columns([3, 1])
with meta_cols[0]:
    if st.session_state.last_scan_time:
        ts = datetime.fromtimestamp(st.session_state.last_scan_time).strftime("%H:%M:%S")
        st.markdown(f'<div style="font-size:13px;color:var(--muted);">✅ Scan complete — {count} ticker{"s" if count != 1 else ""} inside EMA box · last run {ts}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:13px;color:var(--muted);">Configure your scan in the sidebar and hit SCAN NOW.</div>', unsafe_allow_html=True)
with meta_cols[1]:
    if auto_scan and HAS_AUTOREFRESH and st.session_state.last_scan_time:
        remaining = max(0, AUTO_SCAN_SECONDS - (time.time() - st.session_state.last_scan_time))
        st.markdown(f'<div style="font-size:11px;color:var(--faint);text-align:right;">next auto-scan in {int(remaining)}s</div>', unsafe_allow_html=True)

st.markdown(render_results_table(results), unsafe_allow_html=True)

# ── Movers section ───────────────────────────────────────────────────────────
st.markdown('<div class="cf-header" style="font-size:18px;margin-top:8px;">Market Movers</div>', unsafe_allow_html=True)

mover_tabs_labels = [EXCHANGE_META[e]["label"] for e in EXCHANGE_ORDER] + ["US Stocks"]
tabs = st.tabs(mover_tabs_labels)

for i, ex_id in enumerate(EXCHANGE_ORDER):
    with tabs[i]:
        try:
            gainers, losers = get_exchange_mover_rows(ex_id)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(render_mover_table("TOP GAINERS", gainers, True, False, EXCHANGE_META[ex_id]["tv_prefix"]), unsafe_allow_html=True)
            with c2:
                st.markdown(render_mover_table("TOP LOSERS", losers, False, False, EXCHANGE_META[ex_id]["tv_prefix"]), unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div style="font-size:12px;color:var(--rose);">{EXCHANGE_META[ex_id]["label"]} movers error: {e}</div>', unsafe_allow_html=True)

with tabs[-1]:
    try:
        us_gainers, us_losers = get_us_stock_mover_rows()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(render_mover_table("TOP GAINERS", us_gainers, True, True), unsafe_allow_html=True)
        with c2:
            st.markdown(render_mover_table("TOP LOSERS", us_losers, False, True), unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div style="font-size:12px;color:var(--rose);">US Stock movers error: {e}</div>', unsafe_allow_html=True)
