"""
Harmonic Patterns Completion Dashboard - PRO
Clear pattern drawing on chart (X-A-B-C-D lines + markers)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time

DEFAULT_TICKERS = ["SPY", "QQQ", "SPCX", "AAPL", "MSFT", "AMZN", "NVDA", "META", "TSLA"]

TIMEFRAMES = {
    "5m": ("5m", "5d"),
    "15m": ("15m", "10d"),
    "1H": ("1h", "1mo"),
    "4H": ("4h", "3mo"),
    "Daily": ("1d", "1y")
}

PATTERN_DEFS = {
    "Gartley": {"AB_XA": (0.618, 0.06), "BC_AB": (0.382, 0.886), "AD_XA": (0.786, 0.05), "type": "XABCD"},
    "Bat": {"AB_XA": (0.382, 0.50), "BC_AB": (0.382, 0.886), "AD_XA": (0.886, 0.04), "type": "XABCD"},
    "AltBat": {"AB_XA": (0.382, 0.05), "BC_AB": (0.382, 0.886), "AD_XA": (1.13, 0.05), "type": "XABCD"},
    "Butterfly": {"AB_XA": (0.786, 0.05), "BC_AB": (0.382, 0.886), "AD_XA": (1.27, 0.05), "type": "XABCD"},
    "Crab": {"AB_XA": (0.382, 0.618), "BC_AB": (0.382, 0.886), "AD_XA": (1.618, 0.04), "type": "XABCD"},
    "DeepCrab": {"AB_XA": (0.886, 0.05), "BC_AB": (0.382, 0.886), "AD_XA": (1.618, 0.04), "type": "XABCD"},
    "Cypher": {"AB_XA": (0.382, 0.618), "BC_AB": (1.13, 1.414), "AD_XA": (0.786, 0.05), "type": "XABCD"},
    "Shark": {"AB_XA": (0.382, 0.618), "BC_AB": (1.13, 1.618), "AD_XA": (0.886, 1.13), "type": "XABCD"},
    "ABCD": {"BC_AB": (0.382, 0.886), "type": "ABCD"}
}

def send_telegram(token, chat_id, message):
    if not token or not chat_id:
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
        return r.status_code == 200
    except:
        return False

@st.cache_data(ttl=180, show_spinner=False)
def get_ohlcv(ticker, interval, period):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True, threads=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except:
        return pd.DataFrame()

def detect_swings(df, order=5):
    if len(df) < order * 3:
        return []
    high, low, times = df["High"].values, df["Low"].values, df.index
    peaks = argrelextrema(high, np.greater_equal, order=order)[0]
    troughs = argrelextrema(low, np.less_equal, order=order)[0]
    swings = []
    for i in peaks:
        swings.append({"idx": int(i), "price": float(high[i]), "type": "H", "time": times[i]})
    for i in troughs:
        swings.append({"idx": int(i), "price": float(low[i]), "type": "L", "time": times[i]})
    swings = sorted(swings, key=lambda x: x["idx"])
    cleaned = []
    for s in swings:
        if not cleaned:
            cleaned.append(s)
            continue
        if s["type"] == cleaned[-1]["type"]:
            if (s["type"] == "H" and s["price"] > cleaned[-1]["price"]) or (s["type"] == "L" and s["price"] < cleaned[-1]["price"]):
                cleaned[-1] = s
        else:
            cleaned.append(s)
    return cleaned

def ratio_ok(value, target):
    if target[0] < 1.0 and target[1] < 1.5:
        return target[0] * 0.90 <= value <= target[1] * 1.10
    return abs(value - target[0]) <= target[1]

def detect_harmonic_patterns(swings, current_price, lookback=18):
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
        ab_xa, bc_ab = AB / XA, BC / AB

        for name, rules in PATTERN_DEFS.items():
            if rules["type"] != "XABCD":
                continue
            if not ratio_ok(ab_xa, rules["AB_XA"]) or not ratio_ok(bc_ab, rules["BC_AB"]):
                continue
            ad_target = rules["AD_XA"][0]
            if X["type"] == "L":
                ideal_d = X["price"] + XA * ad_target
                direction = "Bullish"
            else:
                ideal_d = X["price"] - XA * ad_target
                direction = "Bearish"

            dist_now = abs(current_price - ideal_d)
            progress = 1.0 - (dist_now / (abs(C["price"] - ideal_d) + 1e-8))
            completion = max(0.0, min(100.0, progress * 100))

            quality = 55
            if abs(ab_xa - 0.618) < 0.05: quality += 12
            if 0.382 <= bc_ab <= 0.886: quality += 10
            if abs(current_price - ideal_d) / XA < 0.04: quality += 15
            quality = min(98, quality)

            results.append({
                "pattern": name, "direction": direction,
                "X_price": round(X["price"], 2), "A_price": round(A["price"], 2),
                "B_price": round(B["price"], 2), "C_price": round(C["price"], 2),
                "ideal_D": round(ideal_d, 2), "current": round(current_price, 2),
                "X_time": X["time"], "A_time": A["time"], "B_time": B["time"], "C_time": C["time"],
                "completion": round(completion, 1), "quality": quality,
                "ab_xa": round(ab_xa, 3), "bc_ab": round(bc_ab, 3),
                "prz_dist_pct": round(abs(current_price - ideal_d) / current_price * 100, 2)
            })

    for i in range(len(pts) - 2):
        A, B, C = pts[i], pts[i+1], pts[i+2]
        if A["type"] == B["type"] or B["type"] == C["type"]:
            continue
        AB = abs(B["price"] - A["price"])
        if AB < 1e-8: continue
        bc_ab = abs(C["price"] - B["price"]) / AB
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
            "pattern": "ABCD", "direction": direction,
            "X_price": None, "A_price": round(A["price"], 2),
            "B_price": round(B["price"], 2), "C_price": round(C["price"], 2),
            "ideal_D": round(ideal_d, 2), "current": round(current_price, 2),
            "X_time": None, "A_time": A["time"], "B_time": B["time"], "C_time": C["time"],
            "completion": round(completion, 1), "quality": min(90, quality),
            "ab_xa": None, "bc_ab": round(bc_ab, 3),
            "prz_dist_pct": round(abs(current_price - ideal_d) / current_price * 100, 2)
        })

    results = sorted(results, key=lambda x: (-x["completion"], -x["quality"]))
    unique, seen = [], set()
    for r in results:
        key = (r["pattern"], r["direction"], round(r["ideal_D"], 1))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique[:6]

def simple_volume_profile(df, bins=24):
    if df.empty or "Volume" not in df.columns:
        return {"poc": None, "vah": None, "val": None}
    try:
        prices = (df["High"] + df["Low"] + df["Close"]) / 3
        hist, edges = np.histogram(prices, bins=bins, weights=df["Volume"].values)
        poc_idx = np.argmax(hist)
        poc = (edges[poc_idx] + edges[poc_idx+1]) / 2
        total = hist.sum()
        sorted_idx = np.argsort(hist)[::-1]
        cum, selected = 0, []
        for i in sorted_idx:
            cum += hist[i]
            selected.append(i)
            if cum >= total * 0.7: break
        return {"poc": round(float(poc), 2), "vah": round(float(edges[max(selected)+1]), 2), "val": round(float(edges[min(selected)]), 2)}
    except:
        return {"poc": None, "vah": None, "val": None}

def make_chart(df, ticker, patterns, vp):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.78, 0.22], subplot_titles=(f"{ticker}", "Volume"))

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color="#00E676", increasing_fillcolor="#00E676",
        decreasing_line_color="#FF1744", decreasing_fillcolor="#FF1744",
        line=dict(width=1.6), whiskerwidth=0.85, name="Price"
    ), row=1, col=1)

    vol_colors = ["#00C853" if c >= o else "#D50000" for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, opacity=0.65, name="Vol"), row=2, col=1)

    for p_idx, p in enumerate(patterns[:2]):
        color = "#00E676" if p["direction"] == "Bullish" else "#FF5252"
        points = []
        if p.get("X_time") is not None and p.get("X_price") is not None:
            points.append(("X", p["X_time"], p["X_price"]))
        if p.get("A_time") is not None:
            points.append(("A", p["A_time"], p["A_price"]))
        if p.get("B_time") is not None:
            points.append(("B", p["B_time"], p["B_price"]))
        if p.get("C_time") is not None:
            points.append(("C", p["C_time"], p["C_price"]))

        if len(points) >= 3:
            xs = [pt[1] for pt in points]
            ys = [pt[2] for pt in points]
            labels = [pt[0] for pt in points]

            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers+text",
                line=dict(color=color, width=2.8),
                marker=dict(size=11, color=color, line=dict(width=1.5, color="white")),
                text=labels, textposition="top center",
                textfont=dict(size=13, color=color, family="Arial Black"),
                name=f"{p['pattern']}", showlegend=False
            ), row=1, col=1)

            last_t = points[-1][1]
            fig.add_trace(go.Scatter(
                x=[last_t, df.index[-1]],
                y=[points[-1][2], p["ideal_D"]],
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                showlegend=False
            ), row=1, col=1)

        fig.add_hline(y=p["ideal_D"], line_dash="dash", line_color=color, line_width=2.2,
                      annotation_text=f"PRZ {p['pattern']} {p['ideal_D']}", annotation_font_size=12,
                      annotation_font_color=color, row=1, col=1)

    if vp.get("poc"):
        fig.add_hline(y=vp["poc"], line_color="#FFD600", line_width=2,
                      annotation_text=f"POC {vp['poc']}", annotation_font_color="#FFD600", row=1, col=1)

    fig.update_layout(
        height=650, template="plotly_dark",
        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        margin=dict(l=10, r=10, t=40, b=10), showlegend=False, font=dict(size=12)
    )
    fig.update_xaxes(rangeslider_visible=False, gridcolor="#1F2937")
    fig.update_yaxes(gridcolor="#1F2937")
    return fig

st.set_page_config(page_title="Harmonic PRO", page_icon="🎯", layout="wide")
st.title("🎯 Harmonic Patterns Completion Dashboard — PRO")
st.caption("SPY + QQQ + SPCX + Mag7 | رسم النموذج على الشارت + تنبيهات تيليجرام")

with st.sidebar:
    st.header("⚙️ الإعدادات")
    custom = st.text_area("الرموز", value=",".join(DEFAULT_TICKERS), height=90)
    tickers = [t.strip().upper() for t in custom.split(",") if t.strip()]
    tf_label = st.selectbox("الإطار الزمني", list(TIMEFRAMES.keys()), index=2)
    interval, period = TIMEFRAMES[tf_label]
    min_comp = st.slider("أقل اكتمال %", 50, 100, 70)
    min_qual = st.slider("أقل جودة", 50, 98, 65)
    max_dist = st.slider("أقصى بعد عن PRZ %", 0.3, 6.0, 3.0)
    st.markdown("---")
    st.subheader("🔔 تيليجرام")
    tg_token = st.text_input("Bot Token", type="password")
    tg_chat = st.text_input("Chat ID")
    alert_th = st.slider("تنبيه إذا اكتمال ≥", 70, 100, 80)
    alert_on = st.checkbox("تفعيل التنبيهات", value=True)
    if st.button("🔄 مسح الآن", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

status, progress = st.empty(), st.progress(0)
all_signals = []

for i, ticker in enumerate(tickers):
    status.info(f"جاري فحص **{ticker}** ({i+1}/{len(tickers)})")
    df = get_ohlcv(ticker, interval, period)
    if df.empty or len(df) < 40:
        progress.progress((i+1)/len(tickers))
        continue
    current = float(df["Close"].iloc[-1])
    order = 5 if interval in ["5m","15m"] else 4 if interval=="1h" else 3
    swings = detect_swings(df, order=order)
    patterns = detect_harmonic_patterns(swings, current)
    vp = simple_volume_profile(df.tail(150))
    for p in patterns:
        if p["completion"] >= min_comp and p["quality"] >= min_qual and p["prz_dist_pct"] <= max_dist:
            p["ticker"] = ticker
            p["vp_poc"] = vp.get("poc")
            all_signals.append(p)
    progress.progress((i+1)/len(tickers))

status.empty(); progress.empty()
all_signals = sorted(all_signals, key=lambda x: (-x["completion"], -x["quality"]))

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("عدد الإشارات", len(all_signals))
c2.metric("أعلى اكتمال", f"{max((s['completion'] for s in all_signals), default=0):.0f}%")
c3.metric("أقوى جودة", f"{max((s['quality'] for s in all_signals), default=0)}")
c4.metric("الإطار", tf_label)
c5.metric("آخر تحديث", datetime.now().strftime("%H:%M:%S"))

if alert_on and tg_token and tg_chat:
    strong = [s for s in all_signals if s["completion"] >= alert_th]
    if strong:
        msg = f"<b>🎯 Harmonic Alerts ({tf_label})</b>\n{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        for s in strong[:8]:
            emoji = "🟢" if s["direction"]=="Bullish" else "🔴"
            msg += f"{emoji} <b>{s['ticker']}</b> {s['pattern']} {s['direction']}\nاكتمال: <b>{s['completion']}%</b> | PRZ: {s['ideal_D']}\n\n"
        if send_telegram(tg_token, tg_chat, msg):
            st.success(f"✅ تم إرسال {len(strong)} تنبيه")
        else:
            st.error("❌ فشل إرسال تيليجرام")

st.markdown("---")
if not all_signals:
    st.warning("لا توجد إشارات حالياً. خفّض الفلاتر.")
else:
    rows = []
    for s in all_signals:
        rows.append({
            "الرمز": s["ticker"], "النموذج": s["pattern"], "الاتجاه": s["direction"],
            "اكتمال%": s["completion"], "الجودة": s["quality"], "PRZ": s["ideal_D"],
            "السعر": s["current"], "بعد%": s["prz_dist_pct"], "POC": s.get("vp_poc")
        })
    df_show = pd.DataFrame(rows)
    st.subheader("📋 جدول الإشارات")
    st.dataframe(df_show, use_container_width=True, height=360)
    st.download_button("⬇️ CSV", df_show.to_csv(index=False).encode("utf-8-sig"), "signals.csv", "text/csv")

    st.subheader("📊 تفاصيل + رسم النموذج")
    sel = st.selectbox("اختر الرمز", sorted(set(s["ticker"] for s in all_signals)))
    if sel:
        df = get_ohlcv(sel, interval, period)
        sigs = [s for s in all_signals if s["ticker"] == sel]
        vp = simple_volume_profile(df.tail(180))
        col1, col2 = st.columns([2.4, 1])
        with col1:
            st.plotly_chart(make_chart(df, sel, sigs, vp), use_container_width=True)
        with col2:
            for s in sigs:
                color = "🟢" if s["direction"]=="Bullish" else "🔴"
                st.markdown(f"""
**{color} {s['pattern']}** — {s['direction']}  
اكتمال: **{s['completion']}%** | جودة: **{s['quality']}**  
PRZ: `{s['ideal_D']}`  
السعر: `{s['current']}`  
بعد: {s['prz_dist_pct']}%
""")
