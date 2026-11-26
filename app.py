import html
import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
CLASSES = ["No_DR", "Mild_Moderate", "Severe_Proliferative"]
LABEL = {
    "No_DR": "No DR",
    "Mild_Moderate": "Mild / moderate",
    "Severe_Proliferative": "Severe / proliferative",
}
TONE = {"No_DR": "#4F7A3A", "Mild_Moderate": "#B0731A", "Severe_Proliferative": "#A3382E"}
PRIMARY = "#1F5C63"
NEUTRAL = "#C9C5B8"

st.set_page_config(page_title="Retina DR", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root {
  --bg: #F7F6F2; --side: #EFEDE6; --surface: #FFFFFF;
  --ink: #1C2126; --muted: #4F5761; --faint: #6E757F;
  --rule: #DAD6CB; --primary: #1F5C63; --primary-hover: #174A50;
}
.stApp, .stApp *:not([data-testid="stIconMaterial"]):not([class*="material"]) {
  font-family: 'IBM Plex Sans', -apple-system, 'Segoe UI', sans-serif !important;
}
.stApp .mono { font-family: 'IBM Plex Mono', ui-monospace, monospace !important; font-variant-numeric: tabular-nums; }
.stApp { background: var(--bg); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container, [data-testid="stMainBlockContainer"] { max-width: 960px; padding: 2.4rem 2rem 3.5rem; }
[data-testid="stSidebar"] { background: var(--side); border-right: 1px solid var(--rule); }
.stApp h1 { font-size: 1.5rem; font-weight: 600; letter-spacing: -0.01em; }
.stApp h2, .stApp h3 { font-size: 1rem; font-weight: 600; }
.stApp p, .stApp li, .stApp label, .stApp span { color: var(--ink); }
[data-testid="stCaptionContainer"] { color: var(--faint) !important; }
.stApp button[data-testid="stBaseButton-secondary"], .stApp button[kind="secondary"],
.stApp button[data-testid="stBaseButton-primary"], .stApp button[kind="primary"] {
  background: var(--primary); border: 1px solid var(--primary); border-radius: 4px;
  box-shadow: none; font-weight: 500; color: #fff;
}
.stApp button[data-testid="stBaseButton-primary"] p,
.stApp button[kind="primary"] p { color: #fff; }
.stApp button:hover { background: var(--primary-hover); border-color: var(--primary-hover); }
[data-testid="stFileUploaderDropzone"] { background: var(--surface); border: 1px solid #BDB8AA; border-radius: 4px; }
[data-baseweb="select"] > div { background: var(--surface); border-radius: 4px; }
[data-testid="stMetric"] { background: transparent; }
.bars { margin: 0.4rem 0 0.8rem; }
.bar-row { display: grid; grid-template-columns: 11rem 1fr 4.4rem; align-items: center; gap: 0.85rem; padding: 0.45rem 0; }
.bar-label { font-size: 0.92rem; }
.bar-track { height: 10px; background: #E9E6DC; border-radius: 2px; }
.bar-fill { height: 100%; border-radius: 2px; }
.bar-val { text-align: right; font-size: 0.875rem; }
.mk { display: inline-block; width: 8px; height: 8px; margin-right: 0.5rem; vertical-align: 0.05em; }
</style>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Menu", ["Dashboard", "Predict", "Visualizations", "Upload & Retrain"])


def esc(x):
    return html.escape(str(x))


def bars(data, vmax, fmt="{:.0f}", highlight=None):
    rows = []
    for k in CLASSES:
        if k not in data:
            continue
        v = data[k]
        pct = 0 if vmax <= 0 else max(0.0, min(100.0, 100.0 * v / vmax))
        colour = PRIMARY if (highlight is None or k == highlight) else NEUTRAL
        rows.append(
            f'<div class="bar-row"><div class="bar-label">'
            f'<span class="mk" style="background:{TONE[k]}"></span>{esc(LABEL[k])}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{colour}"></div></div>'
            f'<div class="bar-val mono">{fmt.format(v)}</div></div>'
        )
    st.markdown('<div class="bars">' + "".join(rows) + "</div>", unsafe_allow_html=True)


if page == "Dashboard":
    st.title("Dashboard")
    try:
        up = requests.get(f"{API}/uptime", timeout=10).json()["uptime_seconds"]
        s = requests.get(f"{API}/status", timeout=10).json()
        c1, c2, c3 = st.columns(3)
        c1.metric("API up-time (seconds)", int(up))
        c2.metric("Retraining", s["status"])
        c3.metric("Pending uploads", s["pending_uploads"])
        if s.get("last_metrics"):
            st.subheader("Last retrain")
            m = s["last_metrics"]
            a, b, c = st.columns(3)
            a.metric("Accuracy", f"{m['accuracy']:.1%}")
            b.metric("Macro-F1", f"{m['macro_f1']:.3f}")
            c.metric("New images", m["trained_on_new_images"])
        if s.get("last_error"):
            st.error(s["last_error"])
    except Exception as e:
        st.error(f"API unreachable: {e}")

elif page == "Predict":
    st.title("Predict")
    f = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if f:
        st.image(f, width=280)
        if st.button("Predict"):
            r = requests.post(f"{API}/predict", files={"file": (f.name, f.getvalue(), f.type)}, timeout=120)
            if r.ok:
                j = r.json()
                st.success(f"{LABEL.get(j['prediction'], j['prediction'])} ({j['confidence']:.1%})")
                bars(j["probabilities"], 1.0, fmt="{:.1%}", highlight=j["prediction"])
            else:
                st.error(r.text)

elif page == "Visualizations":
    st.title("Visualizations")
    try:
        d = requests.get(f"{API}/insights", timeout=300).json()
        st.subheader("Images per class")
        bars(d["counts"], max(d["counts"].values()), fmt="{:,.0f}")
        st.caption("Severe is the smallest class, so accuracy alone is misleading. Macro-F1 is the main metric.")

        st.subheader("Average brightness")
        bars(d["brightness"], 255, fmt="{:.0f}")
        st.caption("Brightness is mostly camera lighting, not disease.")

        st.subheader("Average red channel")
        bars(d["red_mean"], 255, fmt="{:.0f}")
        st.caption("A single colour average is too coarse. The model has to learn local patterns.")
    except Exception as e:
        st.error(e)

else:
    st.title("Upload & Retrain")
    label = st.selectbox("Class of these images", CLASSES, format_func=lambda k: LABEL[k])
    files = st.file_uploader("Upload multiple images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if files and st.button("Upload"):
        r = requests.post(
            f"{API}/upload",
            data={"label": label},
            files=[("files", (x.name, x.getvalue(), x.type)) for x in files],
            timeout=600,
        )
        if r.ok:
            j = r.json()
            st.success(f"Saved {j['saved']}. Pending {j['pending_total']} / {j['auto_retrain_threshold']}.")
            if j.get("auto_retrain_started"):
                st.info("Auto-retrain started.")
        else:
            st.error(r.text)
    if st.button("Trigger retraining"):
        r = requests.post(f"{API}/retrain", timeout=30)
        if r.ok:
            st.success("Retraining started.")
        else:
            st.error(r.text)
    if st.button("Refresh status"):
        s = requests.get(f"{API}/status", timeout=10).json()
        st.write(f"Status: **{s['status']}** · Pending: **{s['pending_uploads']}**")
        if s.get("last_metrics"):
            m = s["last_metrics"]
            st.write(f"Last retrain — accuracy {m['accuracy']:.1%}, macro-F1 {m['macro_f1']:.3f}")
