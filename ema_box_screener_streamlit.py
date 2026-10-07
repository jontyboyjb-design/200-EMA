"""
ChartFlow — EMA Box Screener (Bitget edition)
─────────────────────────────────────────────
Run with:   streamlit run ema_box_screener_streamlit.py

Logic   : Price is INSIDE the band  [ EMA × (1 - dn%) , EMA × (1 + up%) ]
Markets : Bitget Spot | Bitget Futures (USDT-M perpetuals) | Bitget Tokenized Stocks
          The three markets are mutually exclusive — a tokenized stock (e.g. rTSLA/USDT)
          only ever appears under "Bitget Tokenized Stocks", never under "Bitget Spot".
Columns : Market | TF | Symbol | Price | Chart
Movers  : Top gainers / losers per market
"""

import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import ccxt
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
#  Market config — Bitget Spot, Bitget Futures, Bitget Tokenized Stocks
# ══════════════════════════════════════════════════════════════════════════════
# "client" decides which ccxt client (spot or swap) supplies candles/tickers.
# "kind"   decides how symbols are filtered.
# "tv_suffix" is appended to the TradingView symbol (perpetuals use ".P").

EXCHANGE_META = {
    "bitget_spot": {
        "label": "Bitget Spot", "tv_prefix": "BITGET", "tv_suffix": "",
        "client": "spot", "kind": "spot",
    },
    "bitget_futures": {
        "label": "Bitget Futures", "tv_prefix": "BITGET", "tv_suffix": ".P",
        "client": "swap", "kind": "futures",
    },
    "bitget_stocks": {
        "label": "Bitget Tokenized Stocks", "tv_prefix": "BITGET", "tv_suffix": "",
        "client": "spot", "kind": "stocks",
    },
}
EXCHANGE_ORDER = ["bitget_spot", "bitget_futures", "bitget_stocks"]

CLIENT_OPTIONS = {
    "spot": {"defaultType": "spot"},
    "swap": {"defaultType": "swap", "defaultSubType": "linear"},
}


@st.cache_resource(show_spinner=False)
def get_client(client_type: str):
    """One ccxt Bitget client per product type ('spot' or 'swap')."""
    return ccxt.bitget({
        "enableRateLimit": True,
        "timeout": 20000,
        "options": CLIENT_OPTIONS[client_type],
    })


def get_market_client(market_id: str):
    return get_client(EXCHANGE_META[market_id]["client"])


# ── Timeframes ─────────────────────────────────────────────────────────────────
CRYPTO_TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "3d", "1w"]

AUTO_SCAN_SECONDS = 3 * 60   # 3 minutes


# ══════════════════════════════════════════════════════════════════════════════
#  Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def fetch_candles(client, symbol, timeframe, limit=300, retries=3):
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


# ── Tokenized-stock detection (Bitget "Stocks 2.0" rTokens) ──────────────────
# Bitget lists tokenized stocks / ETFs as ordinary USDT spot pairs using an
# "r + ticker" convention (rTSLA/USDT, rNVDA/USDT, rSPY/USDT, ...).
# Because a handful of real crypto coins also start with "R" (RUNE, RAY, ROSE...),
# those are whitelisted below so they stay in Bitget Spot.
PROTECTED_R_CRYPTO_TICKERS = {
    "RNDR", "RENDER", "RUNE", "RVN", "RAY", "ROSE", "RSR", "REN", "RLC", "RACA",
    "RDNT", "RPL", "REQ", "RIF", "RBN", "RSS3", "RARE", "RGT", "REZ", "RAD", "RON",
    "RFOX", "RAMP", "RACE", "REI", "RIO", "RBTC", "RDAO", "RAIL", "RIDE", "RFUEL",
    "RAIN", "RDPX", "RWA", "RBX", "RING", "RIZON", "ROOT", "ROUTE", "REVV", "RFR",
    "RIN", "RZR", "RAID", "RDC", "RBC", "RSC", "RGB", "RPLS", "REEF", "RARI", "RLY",
    "REP", "RONIN", "RPK", "RETH", "RSETH", "RAVE", "RECALL", "RED", "RESOLV", "REX",
    "RBNT", "RCADE", "RIFSOL", "ROAM", "ROCK", "RYO",
}


def is_tokenized_stock_pair(symbol, market=None):
    """True for Bitget spot tokenized stocks / ETFs (rTSLA, rNVDA, rSPY, ...)."""
    base = symbol.split("/")[0]
    base_up = base.upper()

    # Explicit RWA flag from the exchange metadata, if present.
    if market:
        info = market.get("info", {}) or {}
        for key, val in info.items():
            if "RWA" in str(key).upper() and str(val).upper() in ("YES", "TRUE", "1", "Y"):
                return True

    # Strong signal: lowercase "r" prefix + uppercase ticker (e.g. "rTSLA").
    if len(base) >= 2 and base[0] == "r" and base[1:].replace("-", "").replace(".", "").isalnum() \
            and base[1:].upper() == base[1:]:
        return True

    if base_up in PROTECTED_R_CRYPTO_TICKERS:
        return False

    # Fallback heuristic: R + letters (also allow "." / "-" for tickers like BRK-B).
    if len(base_up) >= 2 and base_up[0] == "R":
        rest = base_up[1:]
        if rest.replace("-", "").replace(".", "").isalpha():
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
def fetch_markets_and_tickers(client_type):
    """Cached (90s) markets + tickers snapshot for the spot or swap client."""
    client = get_client(client_type)
    markets = client.load_markets()
    tickers = client.fetch_tickers()
    return markets, tickers


def symbol_belongs_to_market(market_id, symbol, market):
    """Single source of truth — decides which of the 3 markets a symbol lives in."""
    kind = EXCHANGE_META[market_id]["kind"]
    if not market:
        return False

    if kind == "futures":
        return bool(
            market.get("swap")
            and market.get("linear")
            and market.get("quote") == "USDT"
            and not is_stablecoin_pair(symbol)
        )

    # spot + tokenized stocks both come from USDT spot pairs
    if not market.get("spot") or not symbol.endswith("/USDT"):
        return False

    is_stock = is_tokenized_stock_pair(symbol, market)
    if kind == "stocks":
        return is_stock
    # kind == "spot": real crypto only
    return (not is_stock) and (not is_stablecoin_pair(symbol))


def list_market_symbols(market_id):
    """All live symbols for a market with their tickers: list of (symbol, ticker)."""
    client_type = EXCHANGE_META[market_id]["client"]
    markets, tickers = fetch_markets_and_tickers(client_type)
    out = []
    for symbol, ticker in tickers.items():
        market = markets.get(symbol)
        if symbol_belongs_to_market(market_id, symbol, market) and is_market_live(market, ticker):
            out.append((symbol, ticker))
    return out


def get_all_pairs(market_id):
    try:
        return sorted(s for s, _ in list_market_symbols(market_id))
    except Exception:
        return []


def get_top_pairs(market_id, limit=200):
    try:
        rows = list_market_symbols(market_id)
        rows.sort(key=lambda r: get_quote_volume(r[1]), reverse=True)
        return [s for s, _ in rows[:limit]]
    except Exception:
        return get_all_pairs(market_id)[:limit]


def tradingview_symbol(market_id, symbol):
    meta = EXCHANGE_META[market_id]
    base_quote = symbol.split(":")[0].replace("/", "")   # BTC/USDT:USDT -> BTCUSDT
    return f"{meta['tv_prefix']}:{base_quote}{meta['tv_suffix']}"


def tradingview_url(result):
    return f"https://www.tradingview.com/chart/?symbol={tradingview_symbol(result['Exchange ID'], result['Symbol'])}"


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
    """job = (symbol, timeframe, market_id)"""
    symbol, timeframe, market_id = job
    try:
        client = get_market_client(market_id)
        df = fetch_candles(client, symbol, timeframe)
        market_label = EXCHANGE_META[market_id]["label"]

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
            "Exchange": market_label,
            "Exchange ID": market_id,
            "Timeframe": timeframe,
            "Symbol": symbol,
            "AssetType": "crypto",
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

def get_market_mover_rows(market_id, limit=20):
    rows = []
    for symbol, ticker in list_market_symbols(market_id):
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


# ══════════════════════════════════════════════════════════════════════════════
#  Render helpers — plain pandas DataFrames, default Streamlit look
# ══════════════════════════════════════════════════════════════════════════════

def results_to_dataframe(results):
    if not results:
        return pd.DataFrame(columns=["Market", "TF", "Symbol", "Price", "Chart"])

    rows = []
    for r in results:
        rows.append({
            "Market": r["Exchange"],
            "TF": r["Timeframe"],
            "Symbol": r["Symbol"],
            "Price": fmt_price(r["Price"]),
            "Chart": tradingview_url(r),
        })
    return pd.DataFrame(rows)


def movers_to_dataframe(rows, market_id):
    if not rows:
        return pd.DataFrame(columns=["Rank", "Ticker", "%", "Chart"])

    return pd.DataFrame([
        {
            "Rank": r["rank"],
            "Ticker": r["symbol"],
            "%": f"{r['pct']:+.2f}%",
            "Chart": f"https://www.tradingview.com/chart/?symbol={tradingview_symbol(market_id, r['symbol'])}",
        }
        for r in rows
    ])


def full_height(df, row_px=35, header_px=38, buffer_px=3):
    """Height that fits every row so st.dataframe doesn't add an internal scrollbar."""
    return header_px + row_px * max(len(df), 1) + buffer_px


# ══════════════════════════════════════════════════════════════════════════════
#  Streamlit app
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="ChartFlow — EMA Box Screener", page_icon="▦", layout="wide")

if "results" not in st.session_state:
    st.session_state.results = []
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = None
if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("▦ ChartFlow")
    st.caption("EMA Box Screener")

    st.subheader("Markets")
    selected_markets = st.multiselect(
        "Bitget markets",
        options=EXCHANGE_ORDER,
        default=EXCHANGE_ORDER,
        format_func=lambda x: EXCHANGE_META[x]["label"],
    )
    universe_scope = st.radio(
        "Pair scope",
        options=["top200", "all"],
        format_func=lambda x: "Top 200 by volume" if x == "top200" else "All pairs",
        horizontal=True,
    )

    timeframe = st.selectbox("Timeframe", CRYPTO_TIMEFRAMES, index=CRYPTO_TIMEFRAMES.index("1d"))

    st.subheader("EMA Settings")
    ema_period = st.number_input("EMA Period", min_value=1, value=200, step=1)
    st.caption("Price must be inside EMA ± % band to match")

    st.subheader("EMA Box Range")
    use_upper = st.checkbox("Upper Band (above EMA)", value=True)
    up_pct = st.number_input(
        "Upper Band %", min_value=0.01, max_value=50.0, value=2.0, step=0.1,
        disabled=not use_upper, label_visibility="collapsed",
    )
    use_lower = st.checkbox("Lower Band (below EMA)", value=True)
    dn_pct = st.number_input(
        "Lower Band %", min_value=0.01, max_value=50.0, value=2.0, step=0.1,
        disabled=not use_lower, label_visibility="collapsed",
    )

    # A disabled band is set to 0%, so price can't sit on that side of the EMA.
    eff_up_pct = float(up_pct) if use_upper else 0.0
    eff_dn_pct = float(dn_pct) if use_lower else 0.0
    up_txt = f"+{eff_up_pct:.1f}%" if use_upper else "off"
    dn_txt = f"-{eff_dn_pct:.1f}%" if use_lower else "off"
    st.caption(f"{up_txt}  ←  EMA {int(ema_period)}  →  {dn_txt}")

    auto_scan = st.checkbox("Auto-scan every 3 minutes", value=False, disabled=not HAS_AUTOREFRESH)
    if not HAS_AUTOREFRESH:
        st.caption("Install `streamlit-autorefresh` to enable auto-scan.")

    scan_clicked = st.button("🔍 Scan Now", use_container_width=True)

    st.caption("Scans for tickers where price is inside EMA ± % band.")

# ── Auto-refresh trigger ────────────────────────────────────────────────────
due_for_autoscan = False
if HAS_AUTOREFRESH and auto_scan:
    st_autorefresh(interval=15_000, key="cf_autorefresh")
    now = time.time()
    if st.session_state.last_scan_time is None or (now - st.session_state.last_scan_time) >= AUTO_SCAN_SECONDS:
        due_for_autoscan = True

# ── Main content ─────────────────────────────────────────────────────────────
st.title("EMA Box Screener")
st.caption("Bitget Spot · Bitget Futures · Bitget Tokenized Stocks")

should_scan = scan_clicked or due_for_autoscan

if should_scan:
    if not selected_markets:
        st.warning("Select at least one market before scanning.")
    elif not use_upper and not use_lower:
        st.warning("Tick at least one band (upper or lower) before scanning.")
    else:
        jobs = []
        with st.spinner("Building ticker universe..."):
            for mkt_id in selected_markets:
                pairs = get_all_pairs(mkt_id) if universe_scope == "all" else get_top_pairs(mkt_id, 200)
                for pair in pairs:
                    jobs.append((pair, timeframe, mkt_id))

        progress_label = st.empty()
        progress_bar = st.progress(0)

        def update_progress(done, total):
            pct = int(done / total * 100)
            progress_label.text(f"Scanning... {done}/{total} ({pct}%)")
            progress_bar.progress(pct)

        results = run_ema_box_scan(jobs, int(ema_period), eff_up_pct, eff_dn_pct, progress_cb=update_progress)

        st.session_state.results = results
        st.session_state.last_scan_time = time.time()
        st.session_state.scan_count += 1

        progress_label.empty()
        progress_bar.empty()

# ── Results ──────────────────────────────────────────────────────────────────
results = st.session_state.results
count = len(results)

col1, col2 = st.columns([3, 1])
with col1:
    if st.session_state.last_scan_time:
        ts = datetime.fromtimestamp(st.session_state.last_scan_time).strftime("%H:%M:%S")
        st.write(f"✅ Scan complete — {count} ticker{'s' if count != 1 else ''} inside EMA box · last run {ts}")
    else:
        st.write("Configure your scan in the sidebar and hit Scan Now.")
with col2:
    if auto_scan and HAS_AUTOREFRESH and st.session_state.last_scan_time:
        remaining = max(0, AUTO_SCAN_SECONDS - (time.time() - st.session_state.last_scan_time))
        st.caption(f"next auto-scan in {int(remaining)}s")

# Results are shown in separate tables per market so spot and tokenized
# stocks can never be mixed together.
if not results:
    empty_df = results_to_dataframe([])
    st.dataframe(empty_df, use_container_width=True, hide_index=True, height=full_height(empty_df))
else:
    for mkt_id in EXCHANGE_ORDER:
        mkt_results = [r for r in results if r["Exchange ID"] == mkt_id]
        if not mkt_results:
            continue
        st.subheader(f"{EXCHANGE_META[mkt_id]['label']} ({len(mkt_results)})")
        mkt_df = results_to_dataframe(mkt_results)
        st.dataframe(
            mkt_df,
            use_container_width=True,
            hide_index=True,
            height=full_height(mkt_df),
            column_config={"Chart": st.column_config.LinkColumn("Chart", display_text="Open")},
        )

# ── Movers section ───────────────────────────────────────────────────────────
st.header("Market Movers")

tabs = st.tabs([EXCHANGE_META[m]["label"] for m in EXCHANGE_ORDER])

for i, mkt_id in enumerate(EXCHANGE_ORDER):
    with tabs[i]:
        try:
            gainers, losers = get_market_mover_rows(mkt_id)
            gainers_df = movers_to_dataframe(gainers, mkt_id)
            losers_df = movers_to_dataframe(losers, mkt_id)
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Top Gainers")
                st.dataframe(
                    gainers_df,
                    use_container_width=True, hide_index=True,
                    height=full_height(gainers_df),
                    column_config={"Chart": st.column_config.LinkColumn("Chart", display_text="Open")},
                )
            with c2:
                st.subheader("Top Losers")
                st.dataframe(
                    losers_df,
                    use_container_width=True, hide_index=True,
                    height=full_height(losers_df),
                    column_config={"Chart": st.column_config.LinkColumn("Chart", display_text="Open")},
                )
        except Exception as e:
            st.error(f"{EXCHANGE_META[mkt_id]['label']} movers error: {e}")
