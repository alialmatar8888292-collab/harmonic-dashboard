"""
Harmonic Patterns Completion Dashboard - PRO Version
====================================================
Features:
- Stronger multi-pattern detection (Gartley, Bat, Alt-Bat, Butterfly, Crab, Deep Crab, Cypher, Shark, ABCD)
- Completion % toward PRZ
- Simple Volume Profile (POC approximation)
- Telegram alerts (user provides bot token + chat_id)
- Configurable tickers & filters
- Auto refresh
- CSV export

Run:
    pip install streamlit yfinance pandas numpy plotly scipy requests
    streamlit run harmonic_dashboard_pro.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, argrelextrema
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
import json

# ====================== DEFAULT CONFIG ======================
DEFAULT_TICKERS = {
    "ETFs": ["SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "XLK", "QQQM"],
    "Mag7_exGOOGL": ["AAPL", "MSFT", "AMZN", "NVDA", "META", "TSLA"]
}

TIMEFRAMES = {
    "15m": ("15m", "5d"),
    "1H": ("1h", "1mo"),
    "4H": ("4h", "3mo"),
    "Daily": ("1d", "1y")
}

# Scott Carney style ratios with tolerance
PATTERN_DEFS = {
    "Gartley": {
        "AB_XA": (0.618, 0.06),
        "BC_AB": (0.382, 0.886),
        "CD_BC": (1.13, 1.618),
        "AD_XA": (0.786, 0.05),
        "type": "XABCD"
    },
    "Bat": {
        "AB_XA": (0.382, 0.50),
        "BC_AB": (0.382, 0.886),
        "CD_BC": (1.618, 2.618),
        "AD_XA": (0.886, 0.04),
        "type": "XABCD"
    },
    "AltBat": {
        "AB_XA": (0.382, 0.05),
        "BC_AB": (0.382, 0.886),
        "CD_BC": (2.0, 3.618),
        "AD_XA": (1.13, 0.05),
        "type": "XABCD"
    },
    "Butterfly": {
        "AB_XA": (0.786, 0.05),
        "BC_AB": (0.382, 0.886),
        "CD_BC": (1.618, 2.24),
        "AD_XA": (1.27, 0.05),
        "type": "XABCD"
    },
    "Crab": {
        "AB_XA": (0.382, 0.618),
        "BC_AB": (0.382, 0.886),
        "CD_BC": (2.618, 3.618),
        "AD_XA": (1.618, 0.04),
        "type": "XABCD"
    },
    "DeepCrab": {
        "AB_XA": (0.886, 0.05),
        "BC_AB": (0.382, 0.886),
        "CD_BC": (2.0, 3.618),
        "AD_XA": (1.618, 0.04),
        "type": "XABCD"
    },
    "Cypher": {
        "AB_XA": (0.382, 0.618),
        "BC_AB": (1.13, 1.414),
        "CD_BC": (1.272, 2.0),
        "AD_XA": (0.786, 0.05),
        "type": "XABCD"
    },
    "Shark": {
        "AB_XA": (0.382, 0.618),
        "BC_AB": (1.13, 1.618),
        "CD_BC": (1.618, 2.24),
        "AD_XA": (0.886, 1.13),
        "type": "XABCD"
    },
    "ABCD": {
        "BC_AB": (0.382, 0.886),
        "CD_BC": (1.13, 2.618),
        "type": "ABCD"
    }
}

# ====================== TELEGRAM ======================
def send_telegram(token: str, chat_id: str, message: str) -> bool:
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


# ====================== DATA & DETECTION ======================
@st.cache_data(ttl=300, show_spinner=False)
def get_ohlcv(ticker: str, interval: str, period: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True, threads=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception:
        return pd.DataFrame()


def detect_swings(df: pd.DataFrame, order: int = 5) -> list:
    """Return list of dicts: {idx, price, type, time}"""
    if len(df) < order * 3:
        return []

    high = df["High"].values
    low = df["Low"].values
    times = df.index

    peak_idx = argrelextrema(high, np.greater_equal, order=order)[0]
    trough_idx = argrelextrema(low, np.less_equal, order=order)[0]

    swings = []
    for i in peak_idx:
        swings.append({"idx": int(i), "price": float(high[i]), "type": "H", "time": times[i]})
    for i in trough_idx:
        swings.append({"idx": int(i), "price": float(low[i]), "type": "L", "time": times[i]})

    swings = sorted(swings, key=lambda x: x["idx"])

    cleaned = []
    for s in swings:
        if not cleaned:
            cleaned.append(s)
            continue
        if s["type"] == cleaned[-1]["type"]:
            if (s["type"] == "H" and s["price"] > cleaned[-1]["price"]) or \
               (s["type"] == "L" and s["price"] < cleaned[-1]["price"]):
                cleaned[-1] = s
        else:
            cleaned.append(s)
    return cleaned


def ratio_ok(value: float, target) -> bool:
    if isinstance(target[0], (int, float)) and isinstance(target[1], (int, float)):
        if target[0] < 1.0 and target[1] < 1.5:
            return target[0] * 0.92 <= value <= target[1] * 1.08
        else:
            return abs(value - target[0]) <= target[1]
    return False


def detect_harmonic_patterns(swings: list, current_price: float, lookback: int = 15) -> list:
    if len(swings) < 4:
        return []

    pts = swings[-lookback:]
    results = []

    for i in range(len(pts) - 3):
        X, A, B, C = pts[i], pts[i+1], pts[i+2], pts[i+3]

        if not (X["type"] != A["type"] and A["type"] != B["type"] and B["type"] != C["type"]):
            continue

        XA = abs(A["price"] - X["price"])
        AB = abs(B["price"] - A["price"])
        BC = abs(C["price"] - B["price"])
        if XA < 1e-8 or AB < 1e-8:
            continue

        ab_xa = AB / XA
        bc_ab = BC / AB

        for name, rules in PATTERN_DEFS.items():
            if rules["type"] != "XABCD":
                continue

            if not ratio_ok(ab_xa, rules["AB_XA"]):
                continue
            if not ratio_ok(bc_ab, rules["BC_AB"]):
                continue

            ad_target = rules["AD_XA"][0]
            if X["type"] == "L":
                ideal_d = X["price"] + XA * ad_target
                direction = "Bullish"
            else:
                ideal_d = X["price"] - XA * ad_target
                direction = "Bearish"

            move_from_c = abs(ideal_d - C["price"])
            if move_from_c < 1e-8:
                continue
            dist_now = abs(current_price - ideal_d)
            progress = 1.0 - (dist_now / (abs(C["price"] - ideal_d) + 1e-8))
            completion = max(0.0, min(100.0, progress * 100))

            quality = 55
            if abs(ab_xa - 0.618) < 0.04:
                quality += 12
            if 0.382 <= bc_ab <= 0.886:
                quality += 10
            if abs(current_price - ideal_d) / XA < 0.03:
                quality += 15
            quality = min(98, quality)

            results.append({
                "pattern": name,
                "direction": direction,
                "X": round(X["price"], 2),
                "A": round(A["price"], 2),
                "B": round(B["price"], 2),
                "C": round(C["price"], 2),
                "ideal_D": round(ideal_d, 2),
                "current": round(current_price, 2),
                "completion": round(completion, 1),
                "quality": quality,
                "ab_xa": round(ab_xa, 3),
                "bc_ab": round(bc_ab, 3),
                "time_C": str(C["time"])[:16],
                "prz_dist_pct": round(abs(current_price - ideal_d) / current_price * 100, 2)
            })

    for i in range(len(pts) - 2):
        A, B, C = pts[i], pts[i+1], pts[i+2]
        if A["type"] == B["type"] or B["type"] == C["type"]:
            continue
        AB = abs(B["price"] - A["price"])
        BC = abs(C["price"] - B["price"])
        if AB < 1e-8:
            continue
        bc_ab = BC / AB
        if not ratio_ok(bc_ab, PATTERN_DEFS["ABCD"]["BC_AB"]):
            continue

        if A["type"] == "L":
            ideal_d = C["price"] + AB
            direction = "Bullish"
        else:
            ideal_d = C["price"] - AB
            direction = "Bearish"

        dist = abs(current_price - ideal_d)
        completion = max(0, min(100, 100 - (dist / AB * 80)))
        quality = 60 + (10 if 0.5 < bc_ab < 0.8 else 0)

        results.append({
            "pattern": "ABCD",
            "direction": direction,
            "X": None,
            "A": round(A["price"], 2),
            "B": round(B["price"], 2),
            "C": round(C["price"], 2),
            "ideal_D": round(ideal_d, 2),
            "current": round(current_price, 2),
            "completion": round(completion, 1),
            "quality": min(90, quality),
            "ab_xa": None,
            "bc_ab": round(bc_ab, 3),
            "time_C": str(C["time"])[:16],
            "prz_dist_pct": round(abs(current_price - ideal_d) / current_price * 100, 2)
        })

    results = sorted(results, key=lambda x: (-x["completion"], -x["quality"]))
    unique = []
    seen = set()
    for r in results:
        key = (r["pattern"], r["direction"], round(r["ideal_D"], 1))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique[:6]


def simple_volume_profile(df: pd.DataFrame, bins: int = 24) -> dict:
    if df.empty or "Volume" not in df.columns:
        return {"poc": None, "vah": None, "val": None}
    try:
        prices = (df["High"] + df["Low"] + df["Close"]) / 3
        vol = df["Volume"].values
        hist, edges = np.histogram(prices, bins=bins, weights=vol)
        poc_idx = np.argmax(hist)
        poc = (edges[poc_idx] + edges[poc_idx + 1]) / 2
        total = hist.sum()
        target = total * 0.7
        sorted_idx = np.argsort(hist)[::-1]
        cum = 0
        selected = []
        for i in sorted_idx:
            cum += hist[i]
            selected.append(i)
            if cum >= target:
                break
        low_bin = min(selected)
        high_bin = max(selected)
        val = edges[low_bin]
        vah = edges[high_bin + 1]
        return {"poc": round(float(poc), 2), "vah": round(float(vah), 2), "val": round(float(val), 2)}
    except Exception:
        return {"poc": None, "vah": None, "val": None}


def make_chart(df: pd.DataFrame, ticker: str, patterns: list, vp: dict):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.75, 0.25], subplot_titles=(ticker, "Volume"))

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC"
    ), row=1, col=1)

    colors = ["#26a69a" if c >= o else "#ef5350" for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors, name="Vol"), row=2, col=1)

    for p in patterns[:3]:
        color = "#00e676" if p["direction"] == "Bullish" else "#ff1744"
        fig.add_hline(y=p["ideal_D"], line_dash="dot", line_color=color,
                      annotation_text=f"{p['pattern']} PRZ {p['ideal_D']}", row=1, col=1)

    if vp.get("poc"):
        fig.add_hline(y=vp["poc"], line_color="#ffab00", line_width=1,
                      annotation_text=f"POC {vp['poc']}", row=1, col=1)

    fig.update_layout(
        height=520, xaxis_rangeslider_visible=False,
        template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


# ====================== UI ======================
st.set_page_config(page_title="Harmonic PRO Dashboard", page_icon="🎯", layout="wide")

st.title("🎯 Harmonic Patterns Completion Dashboard — PRO")
st.caption("ETFs + Mag7 (ex-GOOGL) | اكتمال النماذج + Volume Profile + Telegram Alerts")

with st.sidebar:
    st.header("⚙️ الإعدادات")

    st.subheader("الرموز")
    custom_tickers = st.text_area(
        "أضف أو عدّل الرموز (مفصولة بفاصلة)",
        value=",".join(DEFAULT_TICKERS["ETFs"] + DEFAULT_TICKERS["Mag7_exGOOGL"]),
        height=80
    )
    tickers = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]

    tf_label = st.selectbox("الإطار الزمني", list(TIMEFRAMES.keys()), index=3)
    interval, period = TIMEFRAMES[tf_label]

    min_comp = st.slider("أقل اكتمال %", 0, 100, 45)
    min_qual = st.slider("أقل جودة", 50, 98, 65)
    max_prz_dist = st.slider("أقصى بعد عن PRZ %", 0.5, 8.0, 4.0)

    st.markdown("---")
    st.subheader("🔔 تيليجرام")
    tg_token = st.text_input("Bot Token", type="password", help="من @BotFather")
    tg_chat = st.text_input("Chat ID", help="من @userinfobot أو مجموعة")
    alert_on_scan = st.checkbox("أرسل تنبيه عند وجود إشارات قوية (≥80% اكتمال)", value=False)

    st.markdown("---")
    auto_refresh = st.checkbox("تحديث تلقائي كل 5 دقائق", value=False)
    if st.button("🔄 مسح الآن", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

if auto_refresh:
    time.sleep(1)
    st.markdown("""
    <meta http-equiv="refresh" content="300">
    """, unsafe_allow_html=True)

status = st.empty()
progress = st.progress(0)
all_signals = []

for i, ticker in enumerate(tickers):
    status.info(f"جاري فحص **{ticker}** ... ({i+1}/{len(tickers)})")
    df = get_ohlcv(ticker, interval, period)
    if df.empty or len(df) < 40:
        progress.progress((i + 1) / len(tickers))
        continue

    current = float(df["Close"].iloc[-1])
    order = 4 if interval in ["15m", "1h"] else 3
    swings = detect_swings(df, order=order)
    patterns = detect_harmonic_patterns(swings, current)
    vp = simple_volume_profile(df.tail(120))

    for p in patterns:
        if (p["completion"] >= min_comp and
            p["quality"] >= min_qual and
            p["prz_dist_pct"] <= max_prz_dist):
            p["ticker"] = ticker
            p["vp_poc"] = vp.get("poc")
            p["vp_vah"] = vp.get("vah")
            p["vp_val"] = vp.get("val")
            all_signals.append(p)

    progress.progress((i + 1) / len(tickers))

status.empty()
progress.empty()

all_signals = sorted(all_signals, key=lambda x: (-x["completion"], -x["quality"]))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("عدد الإشارات", len(all_signals))
c2.metric("أعلى اكتمال", f"{max((s['completion'] for s in all_signals), default=0):.0f}%")
c3.metric("أقوى جودة", f"{max((s['quality'] for s in all_signals), default=0)}")
c4.metric("الإطار", tf_label)
c5.metric("آخر تحديث", datetime.now().strftime("%H:%M:%S"))

if alert_on_scan and tg_token and tg_chat:
    strong = [s for s in all_signals if s["completion"] >= 80]
    if strong:
        msg = f"<b>🎯 Harmonic Alerts ({tf_label})</b>\n\n"
        for s in strong[:8]:
            emoji = "🟢" if s["direction"] == "Bullish" else "🔴"
            msg += (f"{emoji} <b>{s['ticker']}</b> {s['pattern']} {s['direction']}\n"
                    f"اكتمال: {s['completion']}% | جودة: {s['quality']}\n"
                    f"PRZ: {s['ideal_D']} | الحالي: {s['current']}\n\n")
        ok = send_telegram(tg_token, tg_chat, msg)
        if ok:
            st.success("تم إرسال التنبيهات إلى تيليجرام")
        else:
            st.error("فشل إرسال تيليجرام — تحقق من Token و Chat ID")

st.markdown("---")

if not all_signals:
    st.warning("لا توجد إشارات تستوفي الشروط حالياً. خفّض الفلاتر أو غيّر الإطار الزمني.")
else:
    df_show = pd.DataFrame(all_signals)
    cols = ["ticker", "pattern", "direction", "completion", "quality", "ideal_D",
            "current", "prz_dist_pct", "vp_poc", "ab_xa", "bc_ab", "time_C"]
    df_show = df_show[[c for c in cols if c in df_show.columns]]
    df_show.columns = ["الرمز", "النموذج", "الاتجاه", "اكتمال%", "الجودة", "PRZ (D)",
                       "السعر", "بعد%", "POC", "AB/XA", "BC/AB", "وقت C"]

    st.subheader("📋 جدول الإشارات")
    st.dataframe(
        df_show,
        use_container_width=True,
        height=380
    )

    csv = df_show.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ تحميل CSV", csv, "harmonic_signals.csv", "text/csv")

    st.subheader("📊 تفاصيل الرمز")
    unique_tickers = sorted(set(s["ticker"] for s in all_signals))
    sel = st.selectbox("اختر الرمز", unique_tickers)

    if sel:
        df = get_ohlcv(sel, interval, period)
        sigs = [s for s in all_signals if s["ticker"] == sel]
        vp = simple_volume_profile(df.tail(150))

        col_chart, col_info = st.columns([2.2, 1])
        with col_chart:
            st.plotly_chart(make_chart(df, sel, sigs, vp), use_container_width=True)
        with col_info:
            st.markdown("#### النماذج")
            for s in sigs:
                color = "🟢" if s["direction"] == "Bullish" else "🔴"
                st.markdown(f"""
                **{color} {s['pattern']}** — {s['direction']}  
                اكتمال: **{s['completion']}%** | جودة: **{s['quality']}**  
                PRZ ≈ `{s['ideal_D']}`  
                الحالي: `{s['current']}` (بعد {s['prz_dist_pct']}%)  
                POC: `{s.get('vp_poc', '-')}`  
                """)
            if vp.get("poc"):
                st.info(f"Volume Profile\nPOC: {vp['poc']}\nVAH: {vp['vah']}\nVAL: {vp['val']}")

st.markdown("---")
with st.expander("ℹ️ ملاحظات فنية"):
    st.markdown("""
    - الكاشف يعتمد على **Swing points** + نسب فيبوناتشي حسب قواعد Scott Carney (مع هامش تسامح).
    - نسبة الاكتمال تقيس اقتراب السعر الحالي من منطقة الـ PRZ المتوقعة بعد تشكل نقطة C.
    - Volume Profile هنا تقريبي (POC من توزع الحجم على آخر الشموع).
    - للدقة المؤسسية يُفضّل دمج مكتبة `HarmonicPatterns` أو `pyharmonics` مع بيانات عالية الجودة (Polygon / Databento).
    - تيليجرام: أنشئ بوت من @BotFather وضع الـ Token و Chat ID.
    - **ليس نصيحة استثمارية**.
    """)
