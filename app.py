"""Retina DR - Streamlit front end.

All behaviour is unchanged from the original UI: it talks to the same API
endpoints (/uptime, /status, /predict, /insights, /upload, /retrain). Only the
presentation layer was redesigned.
"""
import base64
import html
import os
import time
from datetime import datetime

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
CLASSES = ["Mild_Moderate", "No_DR", "Severe_Proliferative"]
SEVERITY_ORDER = ["No_DR", "Mild_Moderate", "Severe_Proliferative"]
LABEL = {
    "No_DR": "No DR",
    "Mild_Moderate": "Mild to moderate",
    "Severe_Proliferative": "Severe / proliferative",
}
# Semantic colours are used only for the small class markers.
TONE = {"No_DR": "#4F7A3A", "Mild_Moderate": "#B0731A", "Severe_Proliferative": "#A3382E"}
PRIMARY = "#1F5C63"
NEUTRAL = "#C9C5B8"
IMAGE_TYPES = ["png", "jpg", "jpeg"]

# Reference evaluation of the originally deployed model (notebook, 550 test images).
REFERENCE = [
    ("No_DR", 0.970, 0.970, 0.970, 271),
    ("Mild_Moderate", 0.786, 0.893, 0.836, 206),
    ("Severe_Proliferative", 0.689, 0.425, 0.525, 73),
]
REFERENCE_OVERALL = (0.815, 0.763, 0.777, 550, 0.869)

try:
    st.set_page_config(page_title="Retina DR", page_icon=":material/visibility:", layout="wide")
except Exception:  # older Streamlit without Material page icons
    st.set_page_config(page_title="Retina DR", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root {
  --bg: #F7F6F2; --side: #EFEDE6; --surface: #FFFFFF;
  --ink: #1C2126; --muted: #4F5761; --faint: #6E757F;
  --rule: #DAD6CB; --rule-strong: #BDB8AA;
  --primary: #1F5C63; --primary-hover: #174A50;
  --ok: #4F7A3A; --warn: #B0731A; --bad: #A3382E;
}
.stApp, .stApp *:not([data-testid="stIconMaterial"]):not([class*="material"]) {
  font-family: 'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, sans-serif !important;
}
.stApp .mono:not([data-testid="stIconMaterial"]):not([class*="material"]) {
  font-family: 'IBM Plex Mono', ui-monospace, 'SF Mono', Menlo, monospace !important;
  font-variant-numeric: tabular-nums;
}
.stApp { background: var(--bg); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container, [data-testid="stMainBlockContainer"] { max-width: 1040px; padding: 3rem 2.5rem 4rem; }
.stApp p, .stApp li { font-size: 0.9375rem; line-height: 1.6; color: var(--ink); }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--side); border-right: 1px solid var(--rule); }
[data-testid="stSidebarNavLink"] { border-radius: 3px; }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: #E2DFD4; }
[data-testid="stSidebarNavLink"][aria-current="page"] span { font-weight: 600; }
.side-note { font-size: 0.78rem; line-height: 1.5; color: var(--muted); border-top: 1px solid var(--rule); padding-top: 0.9rem; margin-top: 1.5rem; }
.side-note b { font-weight: 600; color: var(--ink); }

/* Page header and sections */
.stApp .ph { margin-bottom: 2rem; }
.stApp .ph h1 { font-size: 1.5rem; font-weight: 600; letter-spacing: -0.01em; line-height: 1.25; margin: 0 0 0.35rem; padding: 0; color: var(--ink); }
.stApp .ph p { margin: 0; color: var(--muted); max-width: 62ch; }
.sec { display: flex; align-items: baseline; gap: 0.75rem; border-top: 1px solid var(--rule); padding-top: 1rem; margin: 2.5rem 0 1.1rem; }
.sec.first { margin-top: 0; }
.stApp .sec h2 { font-size: 1rem; font-weight: 600; margin: 0; padding: 0; color: var(--ink); letter-spacing: 0; }
.sec .n { font-size: 0.8125rem; color: var(--faint); }
.lede { color: var(--muted); max-width: 62ch; margin: 0 0 1rem; }

/* Status band: one ruled strip, no cards */
.band { display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid var(--ink); border-bottom: 1px solid var(--rule); margin-bottom: 0.5rem; }
.band.two { grid-template-columns: repeat(2, 1fr); }
.cell { padding: 1rem 1.25rem 1.1rem 0; }
.cell + .cell { border-left: 1px solid var(--rule); padding-left: 1.25rem; }
.cell-label { font-size: 0.8125rem; color: var(--muted); margin-bottom: 0.35rem; }
.cell-value { font-size: 1.375rem; font-weight: 500; line-height: 1.2; color: var(--ink); }
.cell-value .of { font-size: 0.95rem; color: var(--faint); font-weight: 400; }
.cell-sub { font-size: 0.8125rem; color: var(--faint); margin-top: 0.3rem; }
.meter { height: 4px; background: var(--rule); margin-top: 0.6rem; }
.meter i { display: block; height: 100%; background: var(--primary); }
.mk { display: inline-block; width: 8px; height: 8px; margin-right: 0.55rem; vertical-align: 0.05em; }

/* Key / value rows */
.kv { margin: 0; }
.kv > div { display: grid; grid-template-columns: 11rem 1fr; padding: 0.6rem 0; border-bottom: 1px solid var(--rule); }
.kv dt { color: var(--muted); font-size: 0.9rem; }
.kv dd { margin: 0; font-size: 0.9rem; }

/* Table */
.tbl { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.tbl th { text-align: left; font-weight: 500; font-size: 0.8125rem; color: var(--muted); padding: 0 0 0.5rem; border-bottom: 1px solid var(--rule-strong); }
.tbl td { padding: 0.6rem 0; border-bottom: 1px solid var(--rule); }
.tbl .r { text-align: right; }
.tbl tr.total td { border-bottom: none; color: var(--muted); }
.tbl .flag { color: var(--bad); font-weight: 500; }
.foot { font-size: 0.8125rem; color: var(--faint); margin-top: 0.75rem; max-width: 70ch; line-height: 1.5; }

/* Bars */
.bars { margin: 0.25rem 0; }
.bar-row { display: grid; grid-template-columns: 11.5rem 1fr 4.5rem; align-items: center; gap: 0.9rem; padding: 0.45rem 0; }
.bar-label { font-size: 0.9rem; }
.bar-track { height: 8px; background: #E9E6DC; }
.bar-fill { height: 100%; }
.bar-val { text-align: right; font-size: 0.875rem; }
.axis { font-size: 0.78rem; color: var(--faint); margin-top: 0.35rem; }

/* Notes (replace default alert boxes) */
.note { border-left: 3px solid var(--rule-strong); background: transparent; color: var(--ink); padding: 0.65rem 0.95rem; font-size: 0.9rem; line-height: 1.5; margin: 0.75rem 0; }.note.ok { border-left-color: var(--ok); }
.note.warn { border-left-color: var(--warn); }
.note.bad { border-left-color: var(--bad); }
.note code { font-size: 0.85em; background: transparent; padding: 0; }

/* Prediction result */
.result { border-top: 2px solid var(--ink); padding-top: 0.9rem; }
.result-label { font-size: 0.8125rem; color: var(--muted); }
.result-class { font-size: 1.5rem; font-weight: 600; letter-spacing: -0.01em; margin: 0.25rem 0 0.15rem; }
.result-conf { color: var(--muted); font-size: 0.95rem; margin-bottom: 1.25rem; }
.result-conf .mono { color: var(--ink); font-weight: 500; }
.empty { border-top: 1px solid var(--rule); padding-top: 0.9rem; color: var(--muted); max-width: 46ch; }
.empty b { color: var(--ink); font-weight: 600; }
.reading { border-left: 1px solid var(--rule-strong); padding-left: 1.1rem; }
.reading .cell-label { margin-bottom: 0.4rem; }
.reading p { margin: 0 0 0.6rem; font-size: 0.9rem; }

/* Widgets */
.stApp button[data-testid="stBaseButton-primary"], .stApp button[kind="primary"] {
  background: var(--primary); border: 1px solid var(--primary); border-radius: 4px; box-shadow: none; font-weight: 500;
}
.stApp button[data-testid="stBaseButton-primary"]:hover, .stApp button[kind="primary"]:hover { background: var(--primary-hover); border-color: var(--primary-hover); }
.stApp button[data-testid="stBaseButton-primary"] p, .stApp button[kind="primary"] p,
.stApp button[data-testid="stBaseButton-primary"] span, .stApp button[kind="primary"] span { color: #FFFFFF; }
.stApp button[data-testid="stBaseButton-secondary"], .stApp button[kind="secondary"] {
  background: var(--surface); border: 1px solid var(--rule-strong); border-radius: 4px; box-shadow: none; font-weight: 500; color: var(--ink);
}
.stApp button[data-testid="stBaseButton-secondary"]:hover, .stApp button[kind="secondary"]:hover { border-color: var(--ink); color: var(--ink); }
.stApp button:disabled { opacity: 0.45; }
[data-testid="stFileUploaderDropzone"] { background: var(--surface); border: 1px solid var(--rule-strong); border-radius: 4px; }
[data-baseweb="select"] > div { background: var(--surface); border: 1px solid var(--rule-strong); border-radius: 4px; }
[data-testid="stWidgetLabel"] p { font-size: 0.8125rem; font-weight: 500; color: var(--muted); }
[data-testid="stImage"] img { border: 1px solid var(--rule); border-radius: 2px; }
[data-testid="stCaptionContainer"] { color: var(--faint); }

@media (max-width: 720px) {
  .block-container, [data-testid="stMainBlockContainer"] { padding: 2rem 1rem 3rem; }
  .band, .band.two { grid-template-columns: repeat(2, 1fr); }
  .cell:nth-child(odd) { border-left: none; padding-left: 0; }
  .kv > div { grid-template-columns: 8.5rem 1fr; }
  .bar-row { grid-template-columns: 8.5rem 1fr 3.8rem; gap: 0.6rem; }
  .reading { border-left: none; padding-left: 0; border-top: 1px solid var(--rule); padding-top: 0.9rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Wordmark: a ring with an offset dot, an abstract fundus photograph with its optic disc.
_LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="150" height="30" viewBox="0 0 150 30">'
    '<circle cx="15" cy="15" r="9.5" fill="none" stroke="#1F5C63" stroke-width="2"/>'
    '<circle cx="18" cy="12.5" r="3.4" fill="#1F5C63"/>'
    '<text x="34" y="20.5" font-family="IBM Plex Sans, Helvetica, Arial, sans-serif" '
    'font-size="16" font-weight="600" fill="#1C2126">Retina DR</text></svg>'
)
_LOGO_URI = "data:image/svg+xml;base64," + base64.b64encode(_LOGO_SVG.encode()).decode()


# ----------------------------------------------------------------- helpers
def esc(x):
    return html.escape(str(x))


def api_get(path, timeout=10):
    r = requests.get(f"{API}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def error_text(r):
    try:
        d = r.json()
        return str(d.get("detail", d))
    except Exception:
        return r.text[:300]


def md(s):
    st.markdown(s, unsafe_allow_html=True)


def header(title, lede):
    md(f'<div class="ph"><h1>{esc(title)}</h1><p>{esc(lede)}</p></div>')


def section(title, n=None, first=False):
    num = f'<span class="n mono">{n}</span>' if n else ""
    cls = "sec first" if first else "sec"
    md(f'<div class="{cls}">{num}<h2>{esc(title)}</h2></div>')


def note(kind, text):
    """`text` may contain simple inline HTML; escape user data before passing it."""
    return f'<div class="note {kind}">{text}</div>'


def marker(colour):
    return f'<span class="mk" style="background:{colour}"></span>'


def cell(label, value, sub="", extra=""):
    sub_html = f'<div class="cell-sub">{sub}</div>' if sub else ""
    return (f'<div class="cell"><div class="cell-label">{label}</div>'
            f'<div class="cell-value">{value}</div>{sub_html}{extra}</div>')


def kv(rows):
    body = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in rows)
    return f'<dl class="kv">{body}</dl>'


def bar_row(label, value, vmax, colour, text, tone=None):
    pct = 0 if vmax <= 0 else max(0.0, min(100.0, 100.0 * value / vmax))
    mk = marker(tone) if tone else ""
    return (f'<div class="bar-row"><div class="bar-label">{mk}{esc(label)}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{colour}"></div></div>'
            f'<div class="bar-val mono">{text}</div></div>')


def fmt_uptime(sec):
    sec = int(sec)
    d, rem = divmod(sec, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def fmt_timestamp(ts):
    try:
        return datetime.strptime(str(ts), "%Y%m%d_%H%M%S").strftime("%d %b %Y, %H:%M UTC")
    except Exception:
        return esc(ts)


def fmt_size(n):
    return f"{n / 1024:.0f} KB" if n < 1024 * 1024 else f"{n / (1024 * 1024):.1f} MB"


def ordered(d):
    return [(k, d[k]) for k in SEVERITY_ORDER if k in d]


# ------------------------------------------------------------------ pages
def overview():
    header("Overview", "Service health, retraining state, and how the deployed model performs.")
    status, ms, up, err = None, None, None, None
    try:
        t0 = time.perf_counter()
        up = api_get("/uptime", 10)["uptime_seconds"]
        ms = (time.perf_counter() - t0) * 1000
        status = api_get("/status", 10)
    except Exception as e:
        err = e

    if status is not None:
        running = status["status"] == "retraining"
        pending, thr = status["pending_uploads"], max(1, status["auto_retrain_threshold"])
        last = status.get("last_metrics")
        band = "".join([
            cell("Service", marker(TONE["No_DR"]) + "Online", f'<span class="mono">{ms:.0f} ms</span> response'),
            cell("Up-time", f'<span class="mono">{fmt_uptime(up)}</span>', "since the API last started"),
            cell("Retraining",
                 marker(TONE["Mild_Moderate"] if running else NEUTRAL) + ("Running" if running else "Idle"),
                 f"last run {fmt_timestamp(last['timestamp'])}" if last else "none since the API started"),
            cell("Pending uploads", f'<span class="mono">{pending}</span> <span class="of mono">/ {thr}</span>',
                 "auto-retrain at the limit",
                 f'<div class="meter"><i style="width:{min(100, 100 * pending / thr):.0f}%"></i></div>'),
        ])
        md(f'<div class="band">{band}</div>')
        if status.get("last_error"):
            md(note("bad", f"The last retraining failed: <code>{esc(status['last_error'])}</code>"))
    else:
        band = "".join([
            cell("Service", marker(TONE["Severe_Proliferative"]) + "Unreachable"),
            cell("Up-time", "-"), cell("Retraining", "-"), cell("Pending uploads", "-"),
        ])
        md(f'<div class="band">{band}</div>')
        md(note("bad", f"Could not reach the API at <code>{esc(API)}</code>. {esc(err)}"))
    st.button("Refresh", icon=":material/refresh:", key="ov_refresh")

    section("Last retraining")
    last = (status or {}).get("last_metrics")
    if last:
        md(kv([
            ("Accuracy", f'<span class="mono">{last["accuracy"]:.1%}</span>'),
            ("Macro-F1", f'<span class="mono">{last["macro_f1"]:.3f}</span>'),
            ("Test images", f'<span class="mono">{last["n_test"]}</span>'),
            ("New images used", f'<span class="mono">{last["trained_on_new_images"]}</span>'),
            ("Completed", fmt_timestamp(last["timestamp"])),
        ]))
        md('<div class="foot">Measured on the same held-out test images as the reference below, '
           'so the two are directly comparable.</div>')
    else:
        md('<p class="lede">No retraining has run since the API last started. Results appear here after the first one.</p>')

    section("Reference evaluation")
    rows = ""
    for cls, p, r, f1, n in REFERENCE:
        weak = cls == "Severe_Proliferative"
        rows += (f'<tr><td>{marker(TONE[cls])}{esc(LABEL[cls])}</td>'
                 f'<td class="r mono">{p:.3f}</td>'
                 f'<td class="r mono{" flag" if weak else ""}">{r:.3f}</td>'
                 f'<td class="r mono">{f1:.3f}</td><td class="r mono">{n}</td></tr>')
    mp, mr, mf, mn, acc = REFERENCE_OVERALL
    rows += (f'<tr class="total"><td>Macro average</td><td class="r mono">{mp:.3f}</td>'
             f'<td class="r mono">{mr:.3f}</td><td class="r mono">{mf:.3f}</td><td class="r mono">{mn}</td></tr>')
    md('<table class="tbl"><thead><tr><th>Class</th><th class="r">Precision</th><th class="r">Recall</th>'
       f'<th class="r">F1</th><th class="r">Images</th></tr></thead><tbody>{rows}</tbody></table>')
    md(f'<div class="foot">Original model on 550 unseen test images, overall accuracy {acc:.1%}. '
       'Severe / proliferative is the weak class: 31 of 73 cases were caught, and the 42 missed ones '
       'were labelled mild to moderate, never as no DR. Figures are fixed from the notebook; '
       'live figures above update after each retrain.</div>')


def predict():
    header("Predict", "Classify a single retina photograph into one of three severity groups.")
    left, right = st.columns([5, 6], gap="large")
    with left:
        f = st.file_uploader("Fundus photograph", type=IMAGE_TYPES)
        if f:
            st.image(f, width=280)
            st.caption(f"{f.name}  ·  {fmt_size(f.size)}")
        run = st.button("Run prediction", type="primary", icon=":material/play_arrow:", disabled=f is None)
        if f and run:
            key = (f.name, f.size)
            with st.spinner("Analysing image"):
                try:
                    r = requests.post(f"{API}/predict", files={"file": (f.name, f.getvalue(), f.type)}, timeout=120)
                    st.session_state["pred"] = ({"key": key, "data": r.json()} if r.ok
                                                else {"key": key, "error": error_text(r)})
                except Exception as e:
                    st.session_state["pred"] = {"key": key, "error": str(e)}

    with right:
        p = st.session_state.get("pred")
        if f and p and p["key"] == (f.name, f.size):
            if "error" in p:
                md(note("bad", f"The prediction failed: {esc(p['error'])}"))
            else:
                md(result_html(p["data"]))
        else:
            md('<div class="empty"><b>No result yet.</b> Upload a retina photograph and run the prediction. '
               'Images are resized to 224 x 224 before analysis. Photos from a different camera or with '
               'different framing than the training images can confuse the model.</div>')


def result_html(j):
    top, conf = j["prediction"], j["confidence"]
    rows = sorted(j["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
    bars = "".join(bar_row(LABEL.get(k, k), v, 1.0, PRIMARY if k == top else NEUTRAL, f"{v:.1%}", TONE.get(k))
                   for k, v in rows)
    low = ""
    if conf < 0.6:
        low = note("warn", "Confidence is below 60%. Treat this result with extra caution.")
    return (f'<div class="result"><div class="result-label">Predicted class</div>'
            f'<div class="result-class">{marker(TONE.get(top, PRIMARY))}{esc(LABEL.get(top, top))}</div>'
            f'<div class="result-conf"><span class="mono">{conf:.1%}</span> confidence</div>'
            f'<div class="cell-label">Probability by class</div><div class="bars">{bars}</div>{low}'
            f'<div class="foot">Model output for screening support. It is not a diagnosis.</div></div>')


@st.cache_data(ttl=120, show_spinner=False)
def fetch_insights(api):
    r = requests.get(f"{api}/insights", timeout=300)
    r.raise_for_status()
    return r.json()


def reading(title, paragraphs):
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    return f'<div class="reading"><div class="cell-label">{title}</div>{body}</div>'


def insights():
    header("Data insights", "What the training images look like, and what that means for the model.")
    try:
        with st.spinner("Reading statistics from the training images. The first load can take a minute."):
            d = fetch_insights(API)
    except Exception as e:
        md(note("bad", f"Could not load the data insights. {esc(e)}"))
        return

    # 01 class balance
    counts = ordered(d["counts"])
    total = sum(v for _, v in counts)
    big = max(counts, key=lambda kv: kv[1])
    small = min(counts, key=lambda kv: kv[1])
    ratio = big[1] / small[1] if small[1] else float("inf")
    section("Class balance", "01", first=True)
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        md('<div class="bars">' + "".join(
            bar_row(LABEL[k], v, big[1], PRIMARY, f"{v:,}", TONE[k]) for k, v in counts) + "</div>"
           f'<div class="axis">{total:,} training images in total</div>')
    with c2:
        md(reading("What it shows", [
            f"{esc(LABEL[big[0]])} has {big[1]:,} images and {esc(LABEL[small[0]])} has {small[1]:,}"
            + (f", roughly {ratio:.1f} to 1." if ratio != float("inf") else "."),
            "A model that always guessed the majority class would still score well on accuracy while missing "
            "the cases that matter most. That is why training uses class weights and the main metric is "
            "macro-F1, not accuracy alone.",
        ]))

    # 02 brightness and 03 red channel
    for n, title, key, what, closing in [
        ("02", "Brightness", "brightness",
         "Average pixel brightness of each class",
         "Brightness mostly reflects the camera and the lighting of the photo, not the disease, so on its "
         "own it separates the classes poorly."),
        ("03", "Red channel", "red_mean",
         "Average red channel of each class",
         "Colour averages are a weak signal. The model has to learn finer patterns such as small red "
         "lesions and yellow deposits."),
    ]:
        vals = ordered(d[key])
        hi = max(vals, key=lambda kv: kv[1])
        lo = min(vals, key=lambda kv: kv[1])
        gap = hi[1] - lo[1]
        rel = gap / 255.0
        section(title, n)
        a, b = st.columns([3, 2], gap="large")
        with a:
            md('<div class="bars">' + "".join(
                bar_row(LABEL[k], v, 255, PRIMARY, f"{v:.0f}", TONE[k]) for k, v in vals) + "</div>"
               f'<div class="axis">{esc(what)}, on a 0 to 255 scale</div>')
        with b:
            size = "small" if rel < 0.15 else "visible"
            md(reading("What it shows", [
                f"{esc(LABEL[hi[0]])} is highest at {hi[1]:.0f} and {esc(LABEL[lo[0]])} is lowest at {lo[1]:.0f}, "
                f"a {size} gap of {gap:.0f} out of 255.",
                "Individual images overlap heavily between classes (see the notebook's box plots). " + closing,
            ]))
    md('<div class="foot">Counts cover the current training folder and grow as retraining moves new images in. '
       'Brightness and colour averages come from a sample of up to 60 images per class.</div>')


def retrain_page():
    header("Upload & retrain", "Add labelled images, then fine-tune the deployed model on them.")
    section("Add labelled images", "01", first=True)
    label = st.selectbox("Class of these images", SEVERITY_ORDER, format_func=lambda k: LABEL[k])
    files = st.file_uploader("Images", type=IMAGE_TYPES, accept_multiple_files=True)
    if files:
        st.caption(f"{len(files)} file{'s' if len(files) != 1 else ''} selected  ·  "
                   f"{fmt_size(sum(x.size for x in files))}")
    if st.button("Upload images", icon=":material/upload:", disabled=not files, key="upload_btn"):
        with st.spinner("Uploading"):
            try:
                r = requests.post(f"{API}/upload", data={"label": label},
                                  files=[("files", (x.name, x.getvalue(), x.type)) for x in files], timeout=600)
                if r.ok:
                    j = r.json()
                    msg = (f"Saved <b>{j['saved']}</b> image{'s' if j['saved'] != 1 else ''} as "
                           f"{esc(LABEL[label])}. {j['pending_total']} of {j['auto_retrain_threshold']} "
                           "pending before automatic retraining.")
                    if j.get("auto_retrain_started"):
                        msg += " The limit was reached, so retraining has started."
                        st.session_state["_was_running"] = True
                    md(note("ok", msg))
                else:
                    md(note("bad", f"Upload failed: {esc(error_text(r))}"))
            except Exception as e:
                md(note("bad", f"Upload failed: {esc(e)}"))

    section("Retrain", "02")
    md('<p class="lede">Retraining loads the current model, fine-tunes it on the uploaded images and '
       're-evaluates it on the test set. The previous model is backed up first, and the used images '
       'move into the training data.</p>')

    interval = None
    if st.session_state.get("_was_running"):
        interval = 4
    else:
        try:
            if api_get("/status", 10)["status"] == "retraining":
                interval = 4
        except Exception:
            pass
    st.fragment(retrain_panel, run_every=interval)()


def retrain_panel():
    try:
        s = api_get("/status", 10)
    except Exception as e:
        md(note("bad", f"Could not read the retraining status. {esc(e)}"))
        return
    running = s["status"] == "retraining"
    pending, thr = s["pending_uploads"], max(1, s["auto_retrain_threshold"])
    band = "".join([
        cell("Pending uploads", f'<span class="mono">{pending}</span> <span class="of mono">/ {thr}</span>',
             "retraining starts automatically at the limit",
             f'<div class="meter"><i style="width:{min(100, 100 * pending / thr):.0f}%"></i></div>'),
        cell("State", marker(TONE["Mild_Moderate"] if running else NEUTRAL) + ("Retraining" if running else "Idle"),
             "checking every few seconds" if running else "ready"),
    ])
    md(f'<div class="band two">{band}</div>')
    if s.get("last_error"):
        md(note("bad", f"The last retraining failed: <code>{esc(s['last_error'])}</code>"))

    if st.button("Start retraining", type="primary", icon=":material/model_training:",
                 disabled=(pending == 0 or running), key="retrain_btn"):
        try:
            r = requests.post(f"{API}/retrain", timeout=30)
            if r.ok:
                st.session_state["_was_running"] = True
                st.rerun()
            else:
                md(note("bad", esc(error_text(r))))
        except Exception as e:
            md(note("bad", f"Could not start retraining. {esc(e)}"))
    if pending == 0 and not running:
        md('<div class="foot">Upload at least one image before retraining.</div>')

    last = s.get("last_metrics")
    if last:
        md('<div style="height:1.5rem"></div>')
        md(kv([
            ("Last result", fmt_timestamp(last["timestamp"])),
            ("Accuracy", f'<span class="mono">{last["accuracy"]:.1%}</span>'),
            ("Macro-F1", f'<span class="mono">{last["macro_f1"]:.3f}</span>'),
            ("New images used", f'<span class="mono">{last["trained_on_new_images"]}</span>'),
        ]))

    if st.session_state.get("_was_running") and not running:
        st.session_state["_was_running"] = False
        st.rerun()


# ------------------------------------------------------------- navigation
try:
    st.logo(_LOGO_URI, size="large")
except Exception:
    pass

pages = [
    st.Page(overview, title="Overview", icon=":material/dashboard:", default=True),
    st.Page(predict, title="Predict", icon=":material/visibility:", url_path="predict"),
    st.Page(insights, title="Data insights", icon=":material/bar_chart:", url_path="insights"),
    st.Page(retrain_page, title="Upload & retrain", icon=":material/model_training:", url_path="retrain"),
]
nav = st.navigation(pages)
with st.sidebar:
    md('<div class="side-note"><b>MobileNetV2, 3 classes</b><br>Trained on APTOS 2019 retina images.<br><br>'
       'A learning project, not a medical device.</div>')
nav.run()
