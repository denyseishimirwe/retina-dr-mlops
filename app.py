import html
import os

import pandas as pd
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

# Reference results from the notebook (550 unseen test images, deployed model)
REF_ROWS = [
    ("No DR", 0.970, 0.970, 0.970, 271),
    ("Mild / moderate", 0.786, 0.893, 0.836, 206),
    ("Severe / proliferative", 0.689, 0.425, 0.526, 73),
]
REF_ACC, REF_F1 = 0.869, 0.777
REF_CM = [[184, 8, 14], [8, 263, 0], [42, 0, 31]]
CM_NAMES = ["Mild / moderate", "No DR", "Severe / proliferative"]

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
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
.block-container, [data-testid="stMainBlockContainer"] { max-width: 960px; padding: 2.4rem 2rem 3.5rem; }
[data-testid="stSidebar"] { background: var(--side); border-right: 1px solid var(--rule); }
.stApp h1 { font-size: 1.5rem; font-weight: 600; letter-spacing: -0.01em; }
.stApp h2, .stApp h3 { font-size: 1rem; font-weight: 600; }
.stApp p, .stApp li, .stApp label, .stApp span { color: var(--ink); }
[data-testid="stCaptionContainer"] { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] small { display: none !important; }
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
.bars { margin: 0.4rem 0 0.8rem; }
.bar-row { display: grid; grid-template-columns: 11rem 1fr 4.4rem; align-items: center; gap: 0.85rem; padding: 0.45rem 0; }
.bar-label { font-size: 0.92rem; }
.bar-track { height: 10px; background: #E9E6DC; border-radius: 2px; }
.bar-fill { height: 100%; border-radius: 2px; }
.bar-val { text-align: right; font-size: 0.875rem; }
.mk { display: inline-block; width: 8px; height: 8px; margin-right: 0.5rem; vertical-align: 0.05em; }
.band { display: grid; grid-template-columns: repeat(3, 1fr); border-top: 1px solid var(--ink); border-bottom: 1px solid var(--rule); margin: 0.5rem 0 1.5rem; }
.cell { padding: 1rem 1.1rem 1.1rem 0; }
.cell + .cell { border-left: 1px solid var(--rule); padding-left: 1.1rem; }
.cell-label { font-size: 0.8rem; color: var(--muted); margin-bottom: 0.3rem; }
.cell-value { font-size: 1.35rem; font-weight: 500; }
.note { color: var(--muted); font-size: 0.92rem; margin: 0.1rem 0 1.4rem; max-width: 46rem; }

/* Custom result boxes */
.alert { padding: 0.85rem 1.1rem; margin: 0.5rem 0 1rem; border-radius: 4px;
  background: #FFFFFF; border: 1px solid var(--rule); border-left: 4px solid var(--primary);
  color: var(--ink); font-size: 0.95rem; word-break: break-word; }
.alert.error { background: #FBEFED; border-left-color: #A3382E; }
.alert.success { background: #F0F5EC; border-left-color: #4F7A3A; }
</style>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Predict", "Visualizations", "Evaluation", "Retrain"],
    label_visibility="collapsed",
)


def esc(x):
    return html.escape(str(x))


def alert(kind, text):
    st.markdown(f'<div class="alert {kind}">{esc(text)}</div>', unsafe_allow_html=True)


def err(text):
    alert("error", text)


def ok(text):
    alert("success", text)


def note(text):
    st.markdown(f'<div class="note">{esc(text)}</div>', unsafe_allow_html=True)


def band(cells):
    """cells: list of (label, value, mono) shown as a 3-column band."""
    parts = []
    for label, value, mono in cells:
        cls = "cell-value mono" if mono else "cell-value"
        parts.append(
            f'<div class="cell"><div class="cell-label">{esc(label)}</div>'
            f'<div class="{cls}">{esc(value)}</div></div>'
        )
    st.markdown('<div class="band">' + "".join(parts) + "</div>", unsafe_allow_html=True)


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


def status_band(s):
    m = s.get("last_metrics") or {}
    band([
        ("Status", s["status"], False),
        ("Pending uploads", s["pending_uploads"], True),
        ("Last macro-F1", f'{m["macro_f1"]:.3f}' if "macro_f1" in m else "-", True),
    ])
    if m:
        band([
            ("Accuracy", f'{m.get("accuracy", 0):.1%}', True),
            ("New images used", m.get("trained_on_new_images", 0), True),
            ("Finished", m.get("timestamp", "-"), True),
        ])
    if s.get("last_error"):
        err(s["last_error"])


# ----------------------------------------------------------------- Dashboard
if page == "Dashboard":
    st.title("Dashboard")
    try:
        up = requests.get(f"{API}/uptime", timeout=10).json()["uptime_seconds"]
        s = requests.get(f"{API}/status", timeout=10).json()
        band([
            ("Up-time", f"{int(up)}s", True),
            ("Retrain", s["status"], False),
            ("Pending", s["pending_uploads"], True),
        ])
        m = s.get("last_metrics")
        if m:
            band([
                ("Accuracy", f'{m["accuracy"]:.1%}', True),
                ("Macro-F1", f'{m["macro_f1"]:.3f}', True),
                ("New images", m["trained_on_new_images"], True),
            ])
        if s.get("last_error"):
            err(s["last_error"])
    except Exception as e:
        err(str(e))

# ------------------------------------------------------------------- Predict
elif page == "Predict":
    st.title("Predict")
    f = st.file_uploader("Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    if f:
        st.image(f, width=280)
        if st.button("Predict"):
            r = requests.post(f"{API}/predict", files={"file": (f.name, f.getvalue(), f.type)}, timeout=120)
            if r.ok:
                j = r.json()
                ok(f"{LABEL.get(j['prediction'], j['prediction'])}  {j['confidence']:.1%}")
                bars(j["probabilities"], 1.0, fmt="{:.1%}", highlight=j["prediction"])
            else:
                err(r.text)

# ------------------------------------------------------------ Visualizations
elif page == "Visualizations":
    st.title("Visualizations")
    try:
        with st.spinner("Loading..."):
            d = requests.get(f"{API}/insights", timeout=300).json()
        st.subheader("1. Images per class")
        bars(d["counts"], max(d["counts"].values()), fmt="{:,.0f}")
        note("No DR has far more images than the other classes, so accuracy alone would be "
             "misleading. That is why class weights and macro-F1 are used.")
        st.subheader("2. Average brightness")
        bars(d["brightness"], 255, fmt="{:.0f}")
        note("Brightness is almost the same across classes. It reflects the camera and "
             "lighting, not the disease, so it cannot separate them.")
        st.subheader("3. Average red channel")
        bars(d["red_mean"], 255, fmt="{:.0f}")
        note("Red averages overlap heavily between classes, so simple colour statistics tell "
             "us little. A deep model is needed to learn patterns such as red dots and "
             "yellow patches.")
    except Exception as e:
        err(str(e))

# ---------------------------------------------------------------- Evaluation
elif page == "Evaluation":
    st.title("Evaluation")

    st.subheader("Reference model (notebook, 550 unseen test images)")
    st.table(
        pd.DataFrame(REF_ROWS, columns=["Class", "Precision", "Recall", "F1", "Images"]).set_index("Class")
    )
    band([
        ("Accuracy", f"{REF_ACC:.3f}", True),
        ("Macro-F1", f"{REF_F1:.3f}", True),
        ("Test images", 550, True),
    ])

    st.subheader("Confusion matrix (rows are the true class)")
    st.table(
        pd.DataFrame(
            REF_CM,
            index=[f"True: {n}" for n in CM_NAMES],
            columns=[f"Predicted: {n}" for n in CM_NAMES],
        )
    )
    note("No Severe image was predicted as No DR. The missed Severe cases were labelled "
         "Mild / moderate. The test set has only 73 Severe images, so this result is noisy.")

    st.subheader("Model in production")
    try:
        s = requests.get(f"{API}/status", timeout=10).json()
        m = s.get("last_metrics")
        if m:
            cmp = pd.DataFrame(
                {"Reference": [REF_ACC, REF_F1],
                 "After last retrain": [m["accuracy"], m["macro_f1"]]},
                index=["Accuracy", "Macro-F1"],
            )
            cmp["Change"] = cmp["After last retrain"] - cmp["Reference"]
            st.table(cmp.round(3))
            note(f"Retrained on {m['trained_on_new_images']} new images and tested on "
                 f"{m['n_test']} images.")
        else:
            note("No retraining yet. The deployed model is the reference model.")
    except Exception as e:
        err(str(e))

# ------------------------------------------------------------------- Retrain
else:
    st.title("Retrain")
    label = st.selectbox("Class", CLASSES, format_func=lambda k: LABEL[k])
    files = st.file_uploader(
        "Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if files and st.button("Upload"):
        r = requests.post(
            f"{API}/upload",
            data={"label": label},
            files=[("files", (x.name, x.getvalue(), x.type)) for x in files],
            timeout=600,
        )
        if r.ok:
            j = r.json()
            ok(f"{j['saved']} saved · {j['pending_total']} pending")
        else:
            err(r.text)
    if st.button("Retrain"):
        r = requests.post(f"{API}/retrain", timeout=30)
        if r.ok:
            ok("Retraining started")
        else:
            err(r.text)

    st.subheader("Live status")

    @st.fragment(run_every=3)
    def live_status():
        try:
            s = requests.get(f"{API}/status", timeout=10).json()
        except Exception as e:
            err(str(e))
            return
        status_band(s)

    live_status()