"""
Harmonic Scanner - Scott Carney Official Patterns
Patterns: Gartley, Bat, Alt Bat, Butterfly, Crab, Deep Crab, Cypher, Shark, 5-0
No standalone ABCD
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime
import plotly.graph_objects as go
import requests

DEFAULT_TICKERS = ["SPY", "QQQ", "SPCX", "AAPL", "MSFT", "AMZN", "NVDA", "META", "TSLA"]
TIMEFRAMES = {
    "5m": ("5m", "5d"), "15m": ("15m", "10d"),
    "1H": ("1h", "1mo"), "4H": ("4h", "3mo"), "Daily": ("1d", "1y")
}

PATTERN_DEFS = {
    "Gartley": {
        "AB_XA": (0.618, 0.04),
        "BC_AB": (0.382, 0.886),
        "AD_XA": (0.786, 0.04),
        "desc": "Scott Carney Gartley"
    },
    "Bat": {
        "AB_XA": (0.382, 0.50),
        "BC_AB": (0.382, 0.886),
        "AD_XA": (0.886, 0.03),
        "desc": "Scott Carney Bat"
    },
    "AltBat": {
        "AB_XA": (0.382, 0.04),
        "BC_AB": (0.382, 0.886),
        "AD_XA": (1.13, 0.04),
        "desc": "Scott Carney Alternate Bat"
    },
    "Butterfly": {
        "AB_XA": (0.786, 0.04),
        "BC_AB": (0.382, 0.886),
        "AD_XA": (1.27, 0.05),
        "desc": "Scott Carney Butterfly"
    },
    "Crab": {
        "AB_XA": (0.382, 0.618),
        "BC_AB": (0.382, 0.886),
        "AD_XA": (1.618, 0.04),
        "desc": "Scott Carney Crab"
    },
    "DeepCrab": {
        "AB_XA": (0.886, 0.04),
        "BC_AB": (0.382, 0.886),
        "AD_XA": (1.618, 0.04),
        "desc": "Scott Carney Deep Crab"
    },
    "Cypher": {
        "AB_XA": (0.382, 0.618),
        "BC_AB": (1.13, 1.414),
        "AD_XA": (0.786, 0.04),
        "desc": "Scott Carney Cypher"
    },
    "Shark": {
        "AB_XA": (0.382, 0.618),
        "BC_AB": (1.13, 1.618),
        "AD_XA": (0.886, 1.13),
        "desc": "Scott Carney Shark"
    },
    "5-0": {
        "AB_XA": (1.13, 1.618),
        "BC_AB": (1.618, 2.24),
        "AD_XA": (0.50, 0.05),
        "desc": "Scott Carney 5-0"
    },
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
    a, b = target
    if b > a and b < 3.5 and a < 2.0:
        return a * 0.93 <= value <= b * 1.07
    return abs(value - a) <= b

def detect_harmonic_patterns(swings, current_price, lookback=22):
    if len(swings) < 4: return []
    pts = swings[-lookback:]
    results = []

    for i in range(len(pts) - 3):
        X, A, B, C = pts[i], pts[i+1], pts[i+2], pts[i+3]
        if not (X["type"] != A["type"] and A["type"] != B["type"] and B["type"] != C["type"]):
            continue

        XA = abs(A["price"] - X["price"])
        AB = abs(B["price"] - A["price"])
        BC = abs(C["price"] - B["price"])
        if XA < 1e-8 or AB < 1e-8: continue

        ab_xa = AB / XA
        bc_ab = BC / AB

        for name, rules in PATTERN_DEFS.items():
            if not ratio_ok(ab_xa, rules["AB_XA"]): continue
            if not ratio_ok(bc_ab, rules["BC_AB"]): continue

            ad_target = rules["AD_XA"][0]
            if X["type"] == "L":
                ideal_d = X["price"] + XA * ad_target
                direction = "Bullish"
            else:
                ideal_d = X["price"] - XA * ad_target
                direction = "Bearish"

            dist = abs(current_price - ideal_d)
            full = abs(C["price"] - ideal_d) + 1e-8
            completion = max(0.0, min(100.0, (1.0 - dist / full) * 100))

            quality = 62
            if abs(ab_xa - 0.618) < 0.035: quality += 10
            if 0.382 <= bc_ab <= 0.886: quality += 8
            if abs(current_price - ideal_d) / XA < 0.03: quality += 12
            quality = min(98, quality)

            results.append({
                "pattern": name,
                "direction": direction,
                "X_price": round(X["price"], 2),
                "A_price": round(A["price"], 2),
                "B_price": round(B["price"], 2),
                "C_price": round(C["price"], 2),
                "ideal_D": round(ideal_d, 2),
                "current": round(current_price, 2),
                "X_time": X["time"], "A_time": A["time"],
                "B_time": B["time"], "C_time": C["time"],
                "completion": round(completion, 1),
                "quality": quality,
                "ab_xa": round(ab_xa, 3),
                "bc_ab": round(bc_ab, 3),
                "prz_dist_pct": round(abs(current_price - ideal_d) / current_price * 100, 2)
            })

    results = sorted(results, key=lambda x: (-x["completion"], -x["quality"]))
    unique, seen = [], set()
    for r in results:
        key = (r["pattern"], r["direction"], round(r["ideal_D"], 1))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique[:5]

def simple_volume_profile(df, bins=20):
    if df.empty or "Volume" not in df.columns: return {"poc": None}
    try:
        prices = (df["High"] + df["Low"] + df["Close"]) / 3
        hist, edges = np.histogram(prices, bins=bins, weights=df["Volume"].values)
        poc = (edges[np.argmax(hist)] + edges[np.argmax(hist)+1]) / 2
        return {"poc": round(float(poc), 2)}
    except: return {"poc": None}

def make_chart(df, ticker, patterns, vp, tf_label=""):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color="#26a69a", increasing_fillcolor="#26a69a",
        decreasing_line_color="#ef5350", decreasing_fillcolor="#ef5350",
        line=dict(width=1), whiskerwidth=0.6
    ))
    if not patterns:
        fig.update_layout(height=500, template="plotly_dark", paper_bgcolor="#0B0E14",
                          plot_bgcolor="#0B0E14", xaxis_rangeslider_visible=False)
        return fig

    p = patterns[0]
    color = "#00E676" if p["direction"] == "Bullish" else "#FF5252"
    pts = []
    if p.get("X_time") is not None: pts.append(("X", p["X_time"], p["X_price"]))
    if p.get("A_time") is not None: pts.append(("A", p["A_time"], p["A_price"]))
    if p.get("B_time") is not None: pts.append(("B", p["B_time"], p["B_price"]))
    if p.get("C_time") is not None: pts.append(("C", p["C_time"], p["C_price"]))

    if len(pts) >= 3:
        fig.add_trace(go.Scatter(
            x=[t[1] for t in pts], y=[t[2] for t in pts], mode="lines",
            line=dict(color=color, width=3.5), showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=[pts[-1][1], df.index[-1]], y=[pts[-1][2], p["ideal_D"]],
            mode="lines", line=dict(color=color, width=2.5, dash="dash"), showlegend=False
        ))
        for i, (lab, t, pr) in enumerate(pts + [("D", df.index[-1], p["ideal_D"])]):
            fig.add_trace(go.Scatter(
                x=[t], y=[pr], mode="markers+text",
                marker=dict(size=17 if lab=="D" else 13, color=color,
                            symbol="diamond" if lab=="D" else "circle",
                            line=dict(width=2.5, color="white")),
                text=[lab], textposition="top center" if i%2==0 else "bottom center",
                textfont=dict(size=14, color="white", family="Arial Black"), showlegend=False
            ))

    fig.add_hline(y=p["ideal_D"], line_color=color, line_width=2.5,
                  annotation_text=f"{p['pattern']} PRZ {p['ideal_D']}",
                  annotation_font=dict(size=12, color="white"), annotation_bgcolor=color)
    if vp.get("poc"):
        fig.add_hline(y=vp["poc"], line_color="#FFD600", line_width=1.5,
                      annotation_text=f"POC {vp['poc']}", annotation_font=dict(color="#FFD600"))

    all_t = [t[1] for t in pts] + [df.index[-1]]
    all_p = [t[2] for t in pts] + [p["ideal_D"]]
    if all_t:
        try:
            pad = (max(all_t) - min(all_t)) * 0.12
            fig.update_xaxes(range=[min(all_t)-pad, max(all_t)+pad])
            fig.update_yaxes(range=[min(all_p)*0.997, max(all_p)*1.003])
        except: pass

    fig.update_layout(
        title=dict(text=f"<b>{ticker}</b> • {tf_label} • {p['pattern']} {p['direction']} • {p['completion']}%",
                   font=dict(size=14, color="white"), x=0.5),
        height=520, template="plotly_dark", paper_bgcolor="#0B0E14", plot_bgcolor="#0B0E14",
        margin=dict(l=6, r=6, t=50, b=6), xaxis_rangeslider_visible=False, showlegend=False,
        xaxis=dict(gridcolor="#1a1f2e"), yaxis=dict(gridcolor="#1a1f2e", side="right")
    )
    return fig

st.set_page_config(page_title="Scott Carney Harmonics", page_icon="🎯", layout="wide")
st.title("🎯 Scott Carney Harmonic Scanner")
st.caption("النماذج الرسمية: Gartley • Bat • AltBat • Butterfly • Crab • DeepCrab • Cypher • Shark • 5-0")

with st.sidebar:
    st.header("⚙️ الإعدادات")
    custom = st.text_area("الرموز", value=",".join(DEFAULT_TICKERS), height=80)
    tickers = [t.strip().upper() for t in custom.split(",") if t.strip()]
    selected_tfs = st.multiselect("الفواصل", list(TIMEFRAMES.keys()), default=list(TIMEFRAMES.keys()))
    if not selected_tfs: selected_tfs = list(TIMEFRAMES.keys())
    min_comp = st.slider("أقل اكتمال %", 50, 100, 70)
    min_qual = st.slider("أقل جودة", 50, 98, 65)
    max_dist = st.slider("أقصى بعد %", 0.3, 6.0, 3.0)
    st.markdown("---")
    st.subheader("🔔 تيليجرام")
    tg_token = st.text_input("Bot Token", type="password")
    tg_chat = st.text_input("Chat ID")
    alert_th = st.slider("تنبيه ≥", 70, 100, 80)
    alert_on = st.checkbox("تفعيل التنبيهات", True)
    if st.button("🔄 مسح جميع الفواصل", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

status = st.empty()
progress = st.progress(0)
all_signals = []
total = max(1, len(tickers) * len(selected_tfs))
step = 0

for ticker in tickers:
    for tf_label in selected_tfs:
        step += 1
        status.info(f"**{ticker}** • **{tf_label}** ({step}/{total})")
        interval, period = TIMEFRAMES[tf_label]
        df = get_ohlcv(ticker, interval, period)
        if df.empty or len(df) < 40:
            progress.progress(step/total)
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
        progress.progress(step/total)

status.empty(); progress.empty()
all_signals = sorted(all_signals, key=lambda x: (-x["completion"], -x["quality"]))

c1,c2,c3,c4 = st.columns(4)
c1.metric("الإشارات", len(all_signals))
c2.metric("أعلى اكتمال", f"{max((s['completion'] for s in all_signals), default=0):.0f}%")
c3.metric("الفواصل", len(selected_tfs))
c4.metric("التحديث", datetime.now().strftime("%H:%M"))

if alert_on and tg_token and tg_chat:
    strong = [s for s in all_signals if s["completion"] >= alert_th]
    if strong:
        msg = f"<b>🎯 Scott Carney Alerts</b>\n{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        for s in strong[:10]:
            e = "🟢" if s["direction"]=="Bullish" else "🔴"
            msg += f"{e} <b>{s['ticker']}</b> [{s['timeframe']}] {s['pattern']} {s['direction']}\nاكتمال: <b>{s['completion']}%</b> | PRZ: {s['ideal_D']}\n\n"
        if send_telegram(tg_token, tg_chat, msg):
            st.success(f"✅ تم إرسال {len(strong)} تنبيه")
        else:
            st.error("❌ فشل إرسال تيليجرام")

st.markdown("---")
if not all_signals:
    st.warning("لا توجد إشارات من نماذج سكوت كارني حالياً. خفّض الفلاتر قليلاً.")
else:
    rows = [{"الرمز":s["ticker"],"الفاصل":s["timeframe"],"النموذج":s["pattern"],
             "الاتجاه":s["direction"],"اكتمال%":s["completion"],"الجودة":s["quality"],
             "PRZ":s["ideal_D"],"السعر":s["current"],"بعد%":s["prz_dist_pct"]} for s in all_signals]
    st.subheader("📋 إشارات سكوت كارني")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=360)

    st.subheader("📊 الشارت + رسم النموذج")
    sel = st.selectbox("اختر الرمز", sorted(set(s["ticker"] for s in all_signals)))
    if sel:
        sigs = [s for s in all_signals if s["ticker"]==sel]
        best = sigs[0]
        tf_label = best["timeframe"]
        interval, period = TIMEFRAMES[tf_label]
        df = get_ohlcv(sel, interval, period)
        vp = simple_volume_profile(df.tail(150))

        ymap = {"5m":"5m","15m":"15m","1H":"1h","4H":"1h","Daily":"1d"}
        yahoo_url = f"https://finance.yahoo.com/chart/{sel}?interval={ymap.get(tf_label,'1d')}"
        st.markdown(f'<a href="{yahoo_url}" target="_blank" style="display:inline-block;padding:11px 18px;background:#6A1B9A;color:white;text-decoration:none;border-radius:9px;font-weight:bold;margin:6px 0;">📈 فتح شارت Yahoo Finance — {sel}</a>', unsafe_allow_html=True)

        st.plotly_chart(make_chart(df, sel, [best], vp, tf_label), use_container_width=True)

        st.markdown("#### نقاط النموذج (Scott Carney)")
        pr = []
        if best.get("X_price"): pr.append({"النقطة":"X", "السعر":best["X_price"]})
        pr += [{"النقطة":"A","السعر":best["A_price"]}, {"النقطة":"B","السعر":best["B_price"]},
               {"النقطة":"C","السعر":best["C_price"]}, {"النقطة":"D (PRZ)","السعر":best["ideal_D"]}]
        st.table(pd.DataFrame(pr))

        for s in sigs[:3]:
            e = "🟢" if s["direction"]=="Bullish" else "🔴"
            st.markdown(f"{e} **{s['timeframe']}** | {s['pattern']} {s['direction']} | اكتمال **{s['completion']}%** | PRZ `{s['ideal_D']}`")
