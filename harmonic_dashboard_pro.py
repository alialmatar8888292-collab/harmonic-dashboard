"""
Harmonic Scanner - Scott Carney (نسخة مصححة)
الأنماط: Gartley, Bat, AltBat, Butterfly, Crab, DeepCrab, Cypher, 5-0
(Shark يحتاج نقطة O إضافية، وسيُضاف لاحقاً بمنطق منفصل)
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from scipy.signal import argrelextrema
from datetime import datetime
import plotly.graph_objects as go

DEFAULT_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "NVDA", "META", "TSLA", "GOOGL"]

# الفاصل: (interval, period, resample)
TIMEFRAMES = {
    "5m": ("5m", "5d", None),
    "15m": ("15m", "10d", None),
    "1H": ("1h", "1mo", None),
    "4H": ("1h", "3mo", "4h"),
    "Daily": ("1d", "1y", None),
}

PATTERNS = {
    "Gartley":   {"AB_XA": (0.618, 0.618), "BC_AB": (0.382, 0.886), "D": ("A", 0.786)},
    "Bat":       {"AB_XA": (0.382, 0.50),  "BC_AB": (0.382, 0.886), "D": ("A", 0.886)},
    "AltBat":    {"AB_XA": (0.382, 0.382), "BC_AB": (0.382, 0.886), "D": ("A", 1.13)},
    "Butterfly": {"AB_XA": (0.786, 0.786), "BC_AB": (0.382, 0.886), "D": ("A", 1.27)},
    "Crab":      {"AB_XA": (0.382, 0.618), "BC_AB": (0.382, 0.886), "D": ("A", 1.618)},
    "DeepCrab":  {"AB_XA": (0.886, 0.886), "BC_AB": (0.382, 0.886), "D": ("A", 1.618)},
    "Cypher":    {"AB_XA": (0.382, 0.618), "BC_XA": (1.13, 1.414),  "D": ("XC", 0.786), "C_beyond_A": True},
    "5-0":       {"AB_XA": (1.13, 1.618),  "BC_AB": (1.618, 2.24),  "D": ("BC", 0.50)},
}

SLACK = 0.03


def send_telegram(token, chat_id, message):
    if not token or not chat_id:
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=12)
        return r.status_code == 200
    except Exception:
        return False


@st.cache_data(ttl=120, show_spinner=False)
def get_ohlcv(ticker, tf_label):
    interval, period, rs = TIMEFRAMES[tf_label]
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False,
                         auto_adjust=True, threads=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if rs:
            df = df.resample(rs).agg({"Open": "first", "High": "max", "Low": "min",
                                      "Close": "last", "Volume": "sum"}).dropna()
        return df
    except Exception:
        return pd.DataFrame()


def detect_swings(df, order=5):
    if len(df) < order * 3:
        return []
    high, low, times = df["High"].values, df["Low"].values, df.index
    peaks = argrelextrema(high, np.greater_equal, order=order)[0]
    troughs = argrelextrema(low, np.less_equal, order=order)[0]
    swings = [{"idx": int(i), "price": float(high[i]), "type": "H", "time": times[i]} for i in peaks]
    swings += [{"idx": int(i), "price": float(low[i]), "type": "L", "time": times[i]} for i in troughs]
    swings.sort(key=lambda s: s["idx"])
    cleaned = []
    for s in swings:
        if not cleaned:
            cleaned.append(s)
            continue
        last = cleaned[-1]
        if s["type"] == last["type"]:
            better = (s["type"] == "H" and s["price"] > last["price"]) or \
                     (s["type"] == "L" and s["price"] < last["price"])
            if better:
                cleaned[-1] = s
        else:
            cleaned.append(s)
    return cleaned


def in_range(v, lo, hi):
    return lo * (1 - SLACK) <= v <= hi * (1 + SLACK)


def detect_patterns(swings, price, lookback=12):
    if len(swings) < 4:
        return []
    pts = swings[-lookback:]
    results = []
    for i in range(len(pts) - 3):
        if i + 3 < len(pts) - 2:
            continue
        X, A, B, C = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]
        sgn = 1 if A["price"] > X["price"] else -1
        XA = abs(A["price"] - X["price"])
        AB = abs(B["price"] - A["price"])
        BC = abs(C["price"] - B["price"])
        if XA < 1e-8 or AB < 1e-8:
            continue
        vals = {"AB_XA": AB / XA, "BC_AB": BC / AB, "BC_XA": BC / XA}

        for name, rules in PATTERNS.items():
            checks, ok = [], True
            for key in ("AB_XA", "BC_AB", "BC_XA"):
                if key in rules:
                    lo, hi = rules[key]
                    v = vals[key]
                    checks.append((key.replace("_", "/"), v, lo, hi))
                    if not in_range(v, lo, hi):
                        ok = False
                        break
            if not ok:
                continue
            if rules.get("C_beyond_A") and sgn * (C["price"] - A["price"]) <= 0:
                continue

            kind, r = rules["D"]
            if kind == "A":
                D = A["price"] - sgn * r * XA
            elif kind == "XC":
                D = C["price"] - sgn * r * abs(C["price"] - X["price"])
            else:
                D = C["price"] - sgn * r * BC
            if sgn * (C["price"] - D) <= 0:
                continue

            band = 0.05 * XA
            progress = (C["price"] - price) / (C["price"] - D) * 100
            if progress < 0:
                continue
            if sgn * (price - D) < -band:
                continue
            progress = min(progress, 100.0)

            devs = []
            for _, v, lo, hi in checks:
                center, half = (lo + hi) / 2, max((hi - lo) / 2, 0.04)
                devs.append(min(abs(v - center) / half, 1.5))
            quality = int(max(40, round(100 - 35 * float(np.mean(devs)))))

            results.append({
                "pattern": name,
                "direction": "Bullish" if sgn == 1 else "Bearish",
                "X": X, "A": A, "B": B, "C": C,
                "ideal_D": round(D, 2), "band": round(band, 2),
                "current": round(price, 2),
                "completion": round(progress, 1),
                "quality": quality,
                "checks": checks,
                "d_ratio": f"{kind}:{r}",
                "prz_dist_pct": round(abs(price - D) / price * 100, 2),
            })

    results.sort(key=lambda r: (-r["completion"], -r["quality"]))
    unique, seen = [], set()
    for r in results:
        key = (r["pattern"], r["direction"], round(r["ideal_D"], 1))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique[:5]


@st.cache_data(ttl=120, show_spinner=False)
def scan(tickers, tfs):
    out = []
    for t in tickers:
        for tf in tfs:
            df = get_ohlcv(t, tf)
            if df.empty or len(df) < 40:
                continue
            price = float(df["Close"].iloc[-1])
            order = 5 if tf in ("5m", "15m") else 4 if tf == "1H" else 3
            for p in detect_patterns(detect_swings(df, order), price):
                p["ticker"], p["timeframe"] = t, tf
                out.append(p)
    return out


def make_chart(df, p, tf_label):
    n = len(df)
    xs = list(range(n))
    fmt = "%Y-%m-%d" if tf_label == "Daily" else "%m-%d %H:%M"
    labels = [d.strftime(fmt) for d in df.index]
    color = "#00E676" if p["direction"] == "Bullish" else "#FF5252"
    X, A, B, C = p["X"], p["A"], p["B"], p["C"]
    d_x = n - 1 + max(6, n // 25)
    D = p["ideal_D"]
    band = p["band"]

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=xs, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], text=labels,
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="rgba(38,166,154,0.55)", decreasing_fillcolor="rgba(239,83,80,0.55)",
        line=dict(width=1), showlegend=False))

    legs = [("XA", X, A), ("AB", A, B), ("BC", B, C)]
    for _, s, e in legs:
        fig.add_trace(go.Scatter(x=[s["idx"], e["idx"]], y=[s["price"], e["price"]], mode="lines",
                                 line=dict(color=color, width=1.6), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[C["idx"], d_x], y=[C["price"], D], mode="lines",
                             line=dict(color=color, width=1.6, dash="dash"),
                             hoverinfo="skip", showlegend=False))

    ab_xa = next((c[1] for c in p["checks"] if c[0] == "AB/XA"), None)
    for txt, s, e in [(f"{ab_xa:.3f}" if ab_xa else "", A, B)]:
        if txt:
            fig.add_annotation(x=(s["idx"] + e["idx"]) / 2, y=(s["price"] + e["price"]) / 2,
                               text=txt, showarrow=False, font=dict(size=11, color="#cfd8dc"),
                               bgcolor="rgba(11,14,20,0.8)")
    bc_ab = next((c[1] for c in p["checks"] if c[0] == "BC/AB"), None)
    if bc_ab:
        fig.add_annotation(x=(B["idx"] + C["idx"]) / 2, y=(B["price"] + C["price"]) / 2,
                           text=f"{bc_ab:.3f}", showarrow=False, font=dict(size=11, color="#cfd8dc"),
                           bgcolor="rgba(11,14,20,0.8)")

    d_type = "L" if p["direction"] == "Bullish" else "H"
    for lab, x, y, t in [("X", X["idx"], X["price"], X["type"]), ("A", A["idx"], A["price"], A["type"]),
                         ("B", B["idx"], B["price"], B["type"]), ("C", C["idx"], C["price"], C["type"]),
                         ("D", d_x, D, d_type)]:
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text", text=[lab],
            textposition="top center" if t == "H" else "bottom center",
            marker=dict(size=9, color=color, line=dict(width=1.5, color="white"),
                        symbol="diamond" if lab == "D" else "circle"),
            textfont=dict(size=13, color="white"), hoverinfo="skip", showlegend=False))

    fig.add_shape(type="rect", x0=C["idx"], x1=d_x + 2, y0=D - band, y1=D + band,
                  fillcolor=color, opacity=0.18, line=dict(width=0))
    fig.add_annotation(x=d_x + 2, y=D + band if p["direction"] == "Bearish" else D - band,
                       text=f"PRZ {D}", showarrow=False, xanchor="right",
                       yanchor="bottom" if p["direction"] == "Bearish" else "top",
                       font=dict(size=11, color="white"))

    x0 = max(0, X["idx"] - 3)
    seg = df.iloc[x0:]
    ymin = min(float(seg["Low"].min()), D - band)
    ymax = max(float(seg["High"].max()), D + band)
    pad = (ymax - ymin) * 0.06
    step = max(1, (d_x - x0) // 6)
    tickvals = [i for i in range(x0, n, step)]
    fig.update_xaxes(range=[x0, d_x + 4], tickvals=tickvals, ticktext=[labels[i] for i in tickvals],
                     gridcolor="#1a1f2e", rangeslider_visible=False)
    fig.update_yaxes(range=[ymin - pad, ymax + pad], gridcolor="#1a1f2e", side="right")
    fig.update_layout(height=480, template="plotly_dark", paper_bgcolor="#0B0E14",
                      plot_bgcolor="#0B0E14", margin=dict(l=6, r=6, t=10, b=6), showlegend=False)
    return fig


def main():
    st.set_page_config(page_title="Scott Carney Harmonics", page_icon="🎯", layout="wide")
    st.title("🎯 ماسح نماذج سكوت كارني")
    st.caption("Gartley • Bat • AltBat • Butterfly • Crab • DeepCrab • Cypher • 5-0")

    with st.sidebar:
        st.header("⚙️ الإعدادات")
        custom = st.text_area("الرموز", value=",".join(DEFAULT_TICKERS), height=80)
        tickers = tuple(t.strip().upper() for t in custom.split(",") if t.strip())
        tfs = st.multiselect("الفواصل", list(TIMEFRAMES.keys()), default=list(TIMEFRAMES.keys()))
        tfs = tuple(tfs) if tfs else tuple(TIMEFRAMES.keys())
        min_comp = st.slider("أقل اكتمال %", 0, 100, 60)
        min_qual = st.slider("أقل جودة", 40, 100, 65)
        max_dist = st.slider("أقصى بعد عن D %", 0.3, 6.0, 3.0)
        st.markdown("---")
        st.subheader("🔔 تيليجرام")
        tg_token = st.text_input("Bot Token", type="password")
        tg_chat = st.text_input("Chat ID")
        alert_th = st.slider("تنبيه ≥", 70, 100, 80)
        alert_on = st.checkbox("تفعيل التنبيهات", True)
        if st.button("🔄 تحديث", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("جاري المسح..."):
        found = scan(tickers, tfs)

    signals = [s for s in found if s["completion"] >= min_comp and s["quality"] >= min_qual
               and s["prz_dist_pct"] <= max_dist]
    signals.sort(key=lambda s: (-s["completion"], -s["quality"]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("الإشارات", len(signals))
    c2.metric("أعلى اكتمال", f"{max((s['completion'] for s in signals), default=0):.0f}%")
    c3.metric("الفواصل", len(tfs))
    c4.metric("التحديث", datetime.now().strftime("%H:%M"))

    if alert_on and tg_token and tg_chat:
        sent = st.session_state.setdefault("sent", set())
        new = []
        for s in signals:
            if s["completion"] < alert_th:
                continue
            key = (s["ticker"], s["timeframe"], s["pattern"], s["direction"], round(s["ideal_D"], 1))
            if key not in sent:
                new.append((key, s))
        if new:
            msg = f"<b>🎯 تنبيهات كارني</b>\n{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            for _, s in new[:10]:
                e = "🟢" if s["direction"] == "Bullish" else "🔴"
                msg += (f"{e} <b>{s['ticker']}</b> [{s['timeframe']}] {s['pattern']} {s['direction']}\n"
                        f"اكتمال: <b>{s['completion']}%</b> | D: {s['ideal_D']}\n\n")
            if send_telegram(tg_token, tg_chat, msg):
                sent.update(k for k, _ in new)
                st.success(f"✅ تم إرسال {len(new)} تنبيه")
            else:
                st.error("❌ فشل إرسال تيليجرام")

    st.markdown("---")
    if not signals:
        st.warning("لا توجد إشارات حالياً. خفّض الفلاتر قليلاً.")
        return

    rows = [{"الرمز": s["ticker"], "الفاصل": s["timeframe"], "النموذج": s["pattern"],
             "الاتجاه": s["direction"], "اكتمال%": s["completion"], "الجودة": s["quality"],
             "D": s["ideal_D"], "السعر": s["current"], "بعد%": s["prz_dist_pct"]} for s in signals]
    st.subheader("📋 الإشارات")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=300)

    st.subheader("📊 الشارت")
    labels = [f"{s['ticker']} • {s['timeframe']} • {s['pattern']} {s['direction']} • {s['completion']}%"
              for s in signals]
    choice = st.selectbox("اختر الإشارة", range(len(signals)), format_func=lambda i: labels[i])
    s = signals[choice]
    df = get_ohlcv(s["ticker"], s["timeframe"])

    e = "🟢" if s["direction"] == "Bullish" else "🔴"
    st.markdown(f"### {e} {s['ticker']} • {s['timeframe']} • {s['pattern']} {s['direction']}")
    st.caption(f"اكتمال {s['completion']}% • جودة {s['quality']} • D عند {s['ideal_D']} • السعر {s['current']}")
    st.plotly_chart(make_chart(df, s, s["timeframe"]), use_container_width=True)

    st.markdown("#### النسب الفعلية مقابل المطلوبة")
    ratio_rows = [{"النسبة": n, "الفعلية": round(v, 3),
                   "المطلوب": f"{lo}" if lo == hi else f"{lo} – {hi}",
                   "مطابقة": "✅" if in_range(v, lo, hi) else "❌"} for n, v, lo, hi in s["checks"]]
    st.table(pd.DataFrame(ratio_rows))

    st.markdown("#### النقاط")
    st.table(pd.DataFrame([
        {"النقطة": "X", "السعر": round(s["X"]["price"], 2)},
        {"النقطة": "A", "السعر": round(s["A"]["price"], 2)},
        {"النقطة": "B", "السعر": round(s["B"]["price"], 2)},
        {"النقطة": "C", "السعر": round(s["C"]["price"], 2)},
        {"النقطة": "D (متوقعة)", "السعر": s["ideal_D"]},
    ]))


if __name__ == "__main__":
    main()
