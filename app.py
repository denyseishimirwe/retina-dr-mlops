import html
import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
SEVERITY_ORDER = ["No_DR", "Mild_Moderate", "Severe_Proliferative"]
LABEL = {
    "No_DR": "No DR",
    "Mild_Moderate": "Mild to moderate",
    "Severe_Proliferative": "Severe / proliferative",
}
TONE = {"No_DR": "#4F7A3A", "Mild_Moderate": "#B0731A", "Severe_Proliferative": "#A3382E"}
PRIMARY = "#1F5C63"
NEUTRAL = "#C9C5B8"
IMAGE_TYPES = ["png", "jpg", "jpeg"]

try:
    st.set_page_config(page_title="Retina DR", page_icon=":material/visibility:", layout="wide")
except Exception:
    st.set_page_config(page_title="Retina DR", layout="wide")

st.markdown("""
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
  font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
  font-variant-numeric: tabular-nums;
}
.stApp { background: var(--bg); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container, [data-testid="stMainBlockContainer"] { max-width: 960px; padding: 2.5rem 2rem 3.5rem; }
[data-testid="stSidebar"] { background: var(--side); border-right: 1px solid var(--rule); }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: #E2DFD4; }
.stApp .ph h1 { font-size: 1.5rem; font-weight: 600; margin: 0 0 1.25rem; color: var(--ink); }
.sec { border-top: 1px solid var(--rule); padding-top: 1rem; margin: 2rem 0 1rem; }
.stApp .sec h2 { font-size: 1rem; font-weight: 600; margin: 0; }
.band { display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid var(--ink); border-bottom: 1px solid var(--rule); }
.band.two { grid-template-columns: repeat(2, 1fr); }
.cell { padding: 1rem 1.1rem 1.1rem 0; }
.cell + .cell { border-left: 1px solid var(--rule); padding-left: 1.1rem; }
.cell-label { font-size: 0.8125rem; color: var(--muted); margin-bottom: 0.3rem; }
.cell-value { font-size: 1.3rem; font-weight: 500; color: var(--ink); }
.cell-value .of { font-size: 0.95rem; color: var(--faint); font-weight: 400; }
.meter { height: 4px; background: var(--rule); margin-top: 0.55rem; }
.meter i { display: block; height: 100%; background: var(--primary); }
.mk { display: inline-block; width: 8px; height: 8px; margin-right: 0.5rem; vertical-align: 0.05em; }
.kv { margin: 0; }
.kv > div { display: grid; grid-template-columns: 10rem 1fr; padding: 0.55rem 0; border-bottom: 1px solid var(--rule); }
.kv dt { color: var(--muted); font-size: 0.9rem; }
.kv dd { margin: 0; font-size: 0.9rem; }
.bars { margin: 0.25rem 0; }
.bar-row { display: grid; grid-template-columns: 11rem 1fr 4.2rem; align-items: center; gap: 0.8rem; padding: 0.4rem 0; }
.bar-track { height: 8px; background: #E9E6DC; }
.bar-fill { height: 100%; }
.bar-val { text-align: right; font-size: 0.875rem; }
.note { border-left: 3px solid var(--rule-strong); background: var(--side); padding: 0.6rem 0.9rem; font-size: 0.9rem; margin: 0.75rem 0; }
.note.ok { border-left-color: var(--ok); }
.note.bad { border-left-color: var(--bad); }
.result { border-top: 2px solid var(--ink); padding-top: 0.85rem; }
.result-class { font-size: 1.45rem; font-weight: 600; margin: 0.2rem 0 0.15rem; }
.result-conf { color: var(--muted); margin-bottom: 1rem; }
.reading { color: var(--muted); font-size: 0.9rem; margin: 0.4rem 0 0; }
.stApp button[data-testid="stBaseButton-primary"], .stApp button[kind="primary"] {
  background: var(--primary); border: 1px solid var(--primary); border-radius: 4px; box-shadow: none;
}
.stApp button[data-testid="stBaseButton-primary"] p, .stApp button[kind="primary"] p,
.stApp button[data-testid="stBaseButton-primary"] span, .stApp button[kind="primary"] span { color: #FFFFFF; }
[data-testid="stFileUploaderDropzone"] { background: var(--surface); border: 1px solid var(--rule-strong); border-radius: 4px; }
@media (max-width: 720px) {
  .band, .band.two { grid-template-columns: 1fr 1fr; }
}
</style>
""", unsafe_allow_html=True)


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


def title(text):
    md(f'<div class="ph"><h1>{esc(text)}</h1></div>')


def section(text):
    md(f'<div class="sec"><h2>{esc(text)}</h2></div>')


def note(kind, text):
    return f'<div class="note {kind}">{text}</div>'


def marker(colour):
    return f'<span class="mk" style="background:{colour}"></span>'


def cell(label, value, extra=""):
    return (f'<div class="cell"><div class="cell-label">{label}</div>'
            f'<div class="cell-value">{value}</div>{extra}</div>')


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
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def ordered(d):
    return [(k, d[k]) for k in SEVERITY_ORDER if k in d]


def overview():
    title("Overview")
    status, up, err = None, None, None
    try:
        up = api_get("/uptime")["uptime_seconds"]
        status = api_get("/status")
    except Exception as e:
        err = e

    if status is not None:
        running = status["status"] == "retraining"
        pending, thr = status["pending_uploads"], max(1, status["auto_retrain_threshold"])
        band = "".join([
            cell("Service", marker(TONE["No_DR"]) + "Online"),
            cell("Up-time", f'<span class="mono">{fmt_uptime(up)}</span>'),
            cell("Retraining", marker(TONE["Mild_Moderate"] if running else NEUTRAL) + ("Running" if running else "Idle")),
            cell("Pending", f'<span class="mono">{pending}</span> <span class="of mono">/ {thr}</span>',
                 f'<div class="meter"><i style="width:{min(100, 100 * pending / thr):.0f}%"></i></div>'),
        ])
        md(f'<div class="band">{band}</div>')
        if status.get("last_error"):
            md(note("bad", esc(status["last_error"])))
    else:
        md(note("bad", f"API unreachable. {esc(err)}"))
    st.button("Refresh", key="ov_refresh")

    last = (status or {}).get("last_metrics")
    if last:
        section("Last retrain")
        md(kv([
            ("Accuracy", f'<span class="mono">{last["accuracy"]:.1%}</span>'),
            ("Macro-F1", f'<span class="mono">{last["macro_f1"]:.3f}</span>'),
            ("New images", f'<span class="mono">{last["trained_on_new_images"]}</span>'),
        ]))


def predict():
    title("Predict")
    left, right = st.columns([5, 6], gap="large")
    with left:
        f = st.file_uploader("Image", type=IMAGE_TYPES)
        if f:
            st.image(f, width=260)
        run = st.button("Predict", type="primary", disabled=f is None)
        if f and run:
            key = (f.name, f.size)
            with st.spinner("Predicting"):
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
                md(note("bad", esc(p["error"])))
            else:
                j = p["data"]
                top, conf = j["prediction"], j["confidence"]
                bars = "".join(
                    bar_row(LABEL.get(k, k), v, 1.0, PRIMARY if k == top else NEUTRAL, f"{v:.1%}", TONE.get(k))
                    for k, v in sorted(j["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
                )
                md(f'<div class="result"><div class="cell-label">Predicted class</div>'
                   f'<div class="result-class">{marker(TONE.get(top, PRIMARY))}{esc(LABEL.get(top, top))}</div>'
                   f'<div class="result-conf"><span class="mono">{conf:.1%}</span></div>'
                   f'<div class="bars">{bars}</div></div>')


@st.cache_data(ttl=120, show_spinner=False)
def fetch_insights(api):
    r = requests.get(f"{api}/insights", timeout=300)
    r.raise_for_status()
    return r.json()


def insights():
    title("Data insights")
    try:
        with st.spinner("Loading"):
            d = fetch_insights(API)
    except Exception as e:
        md(note("bad", esc(e)))
        return

    counts = ordered(d["counts"])
    big = max(counts, key=lambda kv: kv[1])
    small = min(counts, key=lambda kv: kv[1])
    section("Class balance")
    md('<div class="bars">' + "".join(
        bar_row(LABEL[k], v, big[1], PRIMARY, f"{v:,}", TONE[k]) for k, v in counts) + "</div>")
    md(f'<p class="reading">{esc(LABEL[small[0]])} is the smallest class, so accuracy alone is misleading. '
       "Training uses class weights; the main metric is macro-F1.</p>")

    for title_text, key, meaning in [
        ("Brightness", "brightness", "Brightness is mostly lighting, not disease."),
        ("Red channel", "red_mean", "A colour average is too coarse; the model needs local lesion patterns."),
    ]:
        vals = ordered(d[key])
        section(title_text)
        md('<div class="bars">' + "".join(
            bar_row(LABEL[k], v, 255, PRIMARY, f"{v:.0f}", TONE[k]) for k, v in vals) + "</div>")
        md(f'<p class="reading">{meaning}</p>')


def retrain_page():
    title("Upload & retrain")
    label = st.selectbox("Class", SEVERITY_ORDER, format_func=lambda k: LABEL[k])
    files = st.file_uploader("Images", type=IMAGE_TYPES, accept_multiple_files=True)
    if st.button("Upload", disabled=not files, key="upload_btn"):
        with st.spinner("Uploading"):
            try:
                r = requests.post(
                    f"{API}/upload",
                    data={"label": label},
                    files=[("files", (x.name, x.getvalue(), x.type)) for x in files],
                    timeout=600,
                )
                if r.ok:
                    j = r.json()
                    msg = f"Saved {j['saved']}. Pending {j['pending_total']} / {j['auto_retrain_threshold']}."
                    if j.get("auto_retrain_started"):
                        msg += " Auto-retrain started."
                        st.session_state["_was_running"] = True
                    md(note("ok", esc(msg)))
                else:
                    md(note("bad", esc(error_text(r))))
            except Exception as e:
                md(note("bad", esc(e)))

    section("Retrain")
    interval = 4 if st.session_state.get("_was_running") else None
    if interval is None:
        try:
            if api_get("/status")["status"] == "retraining":
                interval = 4
        except Exception:
            pass
    st.fragment(retrain_panel, run_every=interval)()


def retrain_panel():
    try:
        s = api_get("/status")
    except Exception as e:
        md(note("bad", esc(e)))
        return
    running = s["status"] == "retraining"
    pending, thr = s["pending_uploads"], max(1, s["auto_retrain_threshold"])
    band = "".join([
        cell("Pending", f'<span class="mono">{pending}</span> <span class="of mono">/ {thr}</span>',
             f'<div class="meter"><i style="width:{min(100, 100 * pending / thr):.0f}%"></i></div>'),
        cell("State", marker(TONE["Mild_Moderate"] if running else NEUTRAL) + ("Retraining" if running else "Idle")),
    ])
    md(f'<div class="band two">{band}</div>')
    if s.get("last_error"):
        md(note("bad", esc(s["last_error"])))

    if st.button("Start retraining", type="primary", disabled=(pending == 0 or running), key="retrain_btn"):
        try:
            r = requests.post(f"{API}/retrain", timeout=30)
            if r.ok:
                st.session_state["_was_running"] = True
                st.rerun()
            else:
                md(note("bad", esc(error_text(r))))
        except Exception as e:
            md(note("bad", esc(e)))

    last = s.get("last_metrics")
    if last:
        md(kv([
            ("Accuracy", f'<span class="mono">{last["accuracy"]:.1%}</span>'),
            ("Macro-F1", f'<span class="mono">{last["macro_f1"]:.3f}</span>'),
        ]))

    if st.session_state.get("_was_running") and not running:
        st.session_state["_was_running"] = False
        st.rerun()


pages = [
    st.Page(overview, title="Overview", icon=":material/dashboard:", default=True),
    st.Page(predict, title="Predict", icon=":material/visibility:", url_path="predict"),
    st.Page(insights, title="Data insights", icon=":material/bar_chart:", url_path="insights"),
    st.Page(retrain_page, title="Upload & retrain", icon=":material/model_training:", url_path="retrain"),
]
st.navigation(pages).run()
