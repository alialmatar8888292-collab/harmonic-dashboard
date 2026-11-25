"""
Harmonic Patterns Dashboard - Multi-Timeframe Auto Scan
يفحص جميع الفواصل تلقائياً ويرسل تنبيهات من أي إطار
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

DEFAULT_TICKERS = ["SPY", "QQQ", "SPCX", "AAPL", "MSFT", "AMZN", "NVDA", "META", "TSLA"]

TIMEFRAMES = {
    "5m":  ("5m",  "5d"),
    "15m": ("15m", "10d"),
    "1H":  ("1h",  "1mo"),
    "4H":  ("4h",  "3mo"),
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
    if not token or not chat_id: return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=12)
        return r.status_code == 200
    except: return False

@st.cache_data(ttl=120, show_spinner=False)
def get_ohlcv(ticker, interval, period):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True, threads=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df[["Open","High","Low","Close","Volume"]].dropna()
    except: return pd.DataFrame()

def detect_swings(df, order=5):
    if len(df) < order*3: return []
    high, low, times = df["High"].values, df["Low"].values, df.index
    peaks = argrelextrema(high, np.greater_equal, order=order)[0]
    troughs = argrelextrema(low, np.less_equal, order=order)[0]
    swings = [{"idx":int(i),"price":float(high[i]),"type":"H","time":times[i]} for i in peaks]
    swings += [{"idx":int(i),"price":float(low[i]),"type":"L","time":times[i]} for i in troughs]
    swings = sorted(swings, key=lambda x: x["idx"])
    cleaned = []
    for s in swings:
        if not cleaned: cleaned.append(s); continue
        if s["type"] == cleaned[-1]["type"]:
            if (s["type"]=="H" and s["price"]>cleaned[-1]["price"]) or (s["type"]=="L" and s["price"]<cleaned[-1]["price"]):
                cleaned[-1] = s
        else: cleaned.append(s)
    return cleaned

def ratio_ok(value, target):
    if target[0] < 1.0 and target[1] < 1.5:
        return target[0]*0.90 <= value <= target[1]*1.10
    return abs(value - target[0]) <= target[1]

def detect_harmonic_patterns(swings, current_price, lookback=18):
    if len(swings) < 4: return []
    pts = swings[-lookback:]
    results = []
    for i in range(len(pts)-3):
        X,A,B,C = pts[i],pts[i+1],pts[i+2],pts[i+3]
        if not (X["type"]!=A["type"] and A["type"]!=B["type"] and B["type"]!=C["type"]): continue
        XA = abs(A["price"]-X["price"])
        AB = abs(B["price"]-A["price"])
        if XA < 1e-8 or AB < 1e-8: continue
        ab_xa = AB/XA
        bc_ab = abs(C["price"]-B["price"])/AB
        for name, rules in PATTERN_DEFS.items():
            if rules["type"] != "XABCD": continue
            if not ratio_ok(ab_xa, rules["AB_XA"]) or not ratio_ok(bc_ab, rules["BC_AB"]): continue
            ad = rules["AD_XA"][0]
            ideal_d = X["price"] + XA*ad if X["type"]=="L" else X["price"] - XA*ad
            direction = "Bullish" if X["type"]=="L" else "Bearish"
            dist = abs(current_price - ideal_d)
            progress = 1.0 - (dist / (abs(C["price"]-ideal_d)+1e-8))
            completion = max(0.0, min(100.0, progress*100))
            quality = 55
            if abs(ab_xa-0.618)<0.05: quality += 12
            if 0.382<=bc_ab<=0.886: quality += 10
            if abs(current_price-ideal_d)/XA < 0.04: quality += 15
            results.append({
                "pattern":name, "direction":direction,
                "X_price":round(X["price"],2), "A_price":round(A["price"],2),
                "B_price":round(B["price"],2), "C_price":round(C["price"],2),
                "ideal_D":round(ideal_d,2), "current":round(current_price,2),
                "X_time":X["time"], "A_time":A["time"], "B_time":B["time"], "C_time":C["time"],
                "completion":round(completion,1), "quality":min(98,quality),
                "prz_dist_pct":round(abs(current_price-ideal_d)/current_price*100,2)
            })
    for i in range(len(pts)-2):
        A,B,C = pts[i],pts[i+1],pts[i+2]
        if A["type"]==B["type"] or B["type"]==C["type"]: continue
        AB = abs(B["price"]-A["price"])
        if AB < 1e-8: continue
        bc_ab = abs(C["price"]-B["price"])/AB
        if not ratio_ok(bc_ab, PATTERN_DEFS["ABCD"]["BC_AB"]): continue
        ideal_d = C["price"]+AB if A["type"]=="L" else C["price"]-AB
        direction = "Bullish" if A["type"]=="L" else "Bearish"
        dist = abs(current_price-ideal_d)
        completion = max(0, min(100, 100-(dist/AB*80)))
        results.append({
            "pattern":"ABCD", "direction":direction,
            "X_price":None, "A_price":round(A["price"],2),
            "B_price":round(B["price"],2), "C_price":round(C["price"],2),
            "ideal_D":round(ideal_d,2), "current":round(current_price,2),
            "X_time":None, "A_time":A["time"], "B_time":B["time"], "C_time":C["time"],
            "completion":round(completion,1), "quality":min(90,60+(10 if 0.5<bc_ab<0.8 else 0)),
            "prz_dist_pct":round(abs(current_price-ideal_d)/current_price*100,2)
        })
    results = sorted(results, key=lambda x: (-x["completion"], -x["quality"]))
    unique, seen = [], set()
    for r in results:
        key = (r["pattern"], r["direction"], round(r["ideal_D"],1))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique[:4]

def simple_volume_profile(df, bins=20):
    if df.empty or "Volume" not in df.columns: return {"poc":None}
    try:
        prices = (df["High"]+df["Low"]+df["Close"])/3
        hist, edges = np.histogram(prices, bins=bins, weights=df["Volume"].values)
        poc = (edges[np.argmax(hist)] + edges[np.argmax(hist)+1])/2
        return {"poc": round(float(poc),2)}
    except: return {"poc":None}

def make_chart(df, ticker, patterns, vp, tf_label=""):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.8, 0.2], subplot_titles=(f"{ticker} | {tf_label}", "Volume"))

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color="#00E676", increasing_fillcolor="#00E676",
        decreasing_line_color="#FF1744", decreasing_fillcolor="#FF1744",
        line=dict(width=1.5), whiskerwidth=0.8, name="Price"
    ), row=1, col=1)

    vol_c = ["#00C853" if c>=o else "#D50000" for o,c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_c, opacity=0.6, name="Vol"), row=2, col=1)

    if patterns:
        p = patterns[0]
        color = "#00E676" if p["direction"]=="Bullish" else "#FF5252"
        pts = []
        if p.get("X_time") is not None and p.get("X_price") is not None:
            pts.append(("X", p["X_time"], p["X_price"]))
        if p.get("A_time") is not None: pts.append(("A", p["A_time"], p["A_price"]))
        if p.get("B_time") is not None: pts.append(("B", p["B_time"], p["B_price"]))
        if p.get("C_time") is not None: pts.append(("C", p["C_time"], p["C_price"]))

        if len(pts) >= 3:
            xs, ys, labels = [t[1] for t in pts], [t[2] for t in pts], [t[0] for t in pts]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers+text",
                line=dict(color=color, width=3),
                marker=dict(size=13, color=color, symbol="circle", line=dict(width=2, color="white")),
                text=labels, textposition="top center",
                textfont=dict(size=13, color="white", family="Arial Black"),
                showlegend=False
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[pts[-1][1], df.index[-1]], y=[pts[-1][2], p["ideal_D"]],
                mode="lines", line=dict(color=color, width=2.5, dash="dash"), showlegend=False
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[df.index[-1]], y=[p["ideal_D"]], mode="markers+text",
                marker=dict(size=15, color=color, symbol="diamond", line=dict(width=2, color="white")),
                text=["D"], textposition="middle right",
                textfont=dict(size=13, color=color, family="Arial Black"), showlegend=False
            ), row=1, col=1)

        fig.add_hline(y=p["ideal_D"], line_dash="dash", line_color=color, line_width=2,
                      annotation_text=f"{p['pattern']} PRZ {p['ideal_D']}", annotation_font=dict(size=12, color=color), row=1, col=1)

    if vp.get("poc"):
        fig.add_hline(y=vp["poc"], line_color="#FFD600", line_width=2,
                      annotation_text=f"POC {vp['poc']}", annotation_font=dict(color="#FFD600"), row=1, col=1)

    fig.update_layout(height=620, template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                      margin=dict(l=8,r=8,t=40,b=8), showlegend=False, font=dict(size=12))
    fig.update_xaxes(rangeslider_visible=False, gridcolor="#1F2937")
    fig.update_yaxes(gridcolor="#1F2937")
    return fig

st.set_page_config(page_title="Harmonic Multi-TF", page_icon="🎯", layout="wide")
st.title("🎯 Harmonic Multi-Timeframe Scanner")
st.caption("يفحص جميع الفواصل تلقائياً (5m → Daily) + تنبيهات تيليجرام")

with st.sidebar:
    st.header("⚙️ الإعدادات")
    custom = st.text_area("الرموز", value=",".join(DEFAULT_TICKERS), height=85)
    tickers = [t.strip().upper() for t in custom.split(",") if t.strip()]

    st.markdown("### الفواصل الزمنية")
    selected_tfs = st.multiselect(
        "اختر الفواصل (اترك الكل مفعّل)",
        list(TIMEFRAMES.keys()),
        default=list(TIMEFRAMES.keys())
    )
    if not selected_tfs:
        selected_tfs = list(TIMEFRAMES.keys())

    min_comp = st.slider("أقل اكتمال %", 50, 100, 70)
    min_qual = st.slider("أقل جودة", 50, 98, 65)
    max_dist = st.slider("أقصى بعد عن PRZ %", 0.3, 6.0, 3.0)

    st.markdown("---")
    st.subheader("🔔 تيليجرام")
    tg_token = st.text_input("Bot Token", type="password")
    tg_chat = st.text_input("Chat ID")
    alert_th = st.slider("تنبيه إذا اكتمال ≥", 70, 100, 80)
    alert_on = st.checkbox("تفعيل التنبيهات", True)

    if st.button("🔄 مسح جميع الفواصل الآن", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

status = st.empty()
progress = st.progress(0)
all_signals = []
total_steps = len(tickers) * len(selected_tfs)
step = 0

for ticker in tickers:
    for tf_label in selected_tfs:
        step += 1
        status.info(f"فحص **{ticker}** على **{tf_label}** ... ({step}/{total_steps})")
        interval, period = TIMEFRAMES[tf_label]
        df = get_ohlcv(ticker, interval, period)
        if df.empty or len(df) < 40:
            progress.progress(step / total_steps)
            continue
        current = float(df["Close"].iloc[-1])
        order = 5 if tf_label in ["5m","15m"] else 4 if tf_label=="1H" else 3
        swings = detect_swings(df, order)
        patterns = detect_harmonic_patterns(swings, current)
        vp = simple_volume_profile(df.tail(120))
        for p in patterns:
            if p["completion"] >= min_comp and p["quality"] >= min_qual and p["prz_dist_pct"] <= max_dist:
                p["ticker"] = ticker
                p["timeframe"] = tf_label
                p["vp_poc"] = vp.get("poc")
                all_signals.append(p)
        progress.progress(step / total_steps)

status.empty()
progress.empty()

all_signals = sorted(all_signals, key=lambda x: (-x["completion"], -x["quality"]))

c1,c2,c3,c4 = st.columns(4)
c1.metric("إجمالي الإشارات", len(all_signals))
c2.metric("أعلى اكتمال", f"{max((s['completion'] for s in all_signals), default=0):.0f}%")
c3.metric("الفواصل المفحوصة", len(selected_tfs))
c4.metric("آخر تحديث", datetime.now().strftime("%H:%M:%S"))

if alert_on and tg_token and tg_chat:
    strong = [s for s in all_signals if s["completion"] >= alert_th]
    if strong:
        msg = f"<b>🎯 Multi-TF Harmonic Alerts</b>\n{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        for s in strong[:12]:
            e = "🟢" if s["direction"]=="Bullish" else "🔴"
            msg += f"{e} <b>{s['ticker']}</b> [{s['timeframe']}] {s['pattern']} {s['direction']}\nاكتمال: <b>{s['completion']}%</b> | PRZ: {s['ideal_D']}\n\n"
        if send_telegram(tg_token, tg_chat, msg):
            st.success(f"✅ تم إرسال {len(strong)} تنبيه من عدة فواصل")
        else:
            st.error("❌ فشل إرسال تيليجرام")

st.markdown("---")

if not all_signals:
    st.warning("لا توجد إشارات حالياً على الفواصل المختارة. خفّض الفلاتر.")
else:
    rows = []
    for s in all_signals:
        rows.append({
            "الرمز": s["ticker"],
            "الفاصل": s["timeframe"],
            "النموذج": s["pattern"],
            "الاتجاه": s["direction"],
            "اكتمال%": s["completion"],
            "الجودة": s["quality"],
            "PRZ": s["ideal_D"],
            "السعر": s["current"],
            "بعد%": s["prz_dist_pct"]
        })
    st.subheader("📋 جميع الإشارات من كل الفواصل")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=400)

    st.subheader("📊 تفاصيل + رسم النموذج")
    unique_tickers = sorted(set(s["ticker"] for s in all_signals))
    sel = st.selectbox("اختر الرمز", unique_tickers)
    if sel:
        sigs = [s for s in all_signals if s["ticker"] == sel]
        best = sigs[0]
        tf_label = best["timeframe"]
        interval, period = TIMEFRAMES[tf_label]
        df = get_ohlcv(sel, interval, period)
        vp = simple_volume_profile(df.tail(150))
        st.plotly_chart(make_chart(df, sel, [best], vp, tf_label), use_container_width=True)
        st.markdown(f"**أفضل إشارة على {sel}:**")
        for s in sigs[:3]:
            e = "🟢" if s["direction"]=="Bullish" else "🔴"
            st.markdown(f"{e} **{s['timeframe']}** | {s['pattern']} {s['direction']} | اكتمال **{s['completion']}%** | PRZ `{s['ideal_D']}`")
