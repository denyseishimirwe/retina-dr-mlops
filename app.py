import os, requests
import pandas as pd
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
CLASSES = ["Mild_Moderate", "No_DR", "Severe_Proliferative"]

st.set_page_config(page_title="Retina DR", layout="wide")
page = st.sidebar.radio("Menu", ["Dashboard", "Predict", "Visualizations", "Upload & Retrain"])

if page == "Dashboard":
    st.title("Model Status")
    try:
        up = requests.get(f"{API}/uptime", timeout=10).json()["uptime_seconds"]
        st.metric("API up-time (seconds)", up)
        s = requests.get(f"{API}/status", timeout=10).json()
        st.metric("Retraining status", s["status"])
        st.metric("Pending uploads", s["pending_uploads"])
        if s.get("last_metrics"):
            st.subheader("Last retrain results (test-set evaluation)")
            st.json(s["last_metrics"])
        if s.get("last_error"):
            st.error(s["last_error"])
    except Exception as e:
        st.error(f"API unreachable: {e}")

elif page == "Predict":
    st.title("Predict one retina image")
    f = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if f:
        st.image(f, width=300)
        if st.button("Predict"):
            r = requests.post(f"{API}/predict", files={"file": (f.name, f.getvalue(), f.type)}, timeout=120)
            if r.ok:
                j = r.json()
                st.success(f"{j['prediction']} ({j['confidence']:.1%})")
                st.bar_chart(pd.Series(j["probabilities"]))
            else:
                st.error(r.text)

elif page == "Visualizations":
    st.title("Data insights")
    st.caption("Three image features from the training set, and what they say about diabetic retinopathy screening.")
    try:
        d = requests.get(f"{API}/insights", timeout=300).json()
        st.subheader("1. Images per class")
        st.bar_chart(pd.Series(d["counts"]))
        st.markdown(
            "**Story:** No_DR is the largest class and Severe_Proliferative is the smallest "
            "(about 9:1 in the original 5-grade data: 1,805 No_DR vs 193 Severe). "
            "A model that always guessed “healthy” would look decent on accuracy. "
            "That is why training uses class weights and why macro-F1 is the main metric, "
            "not accuracy."
        )
        st.subheader("2. Average brightness per class")
        st.bar_chart(pd.Series(d["brightness"]))
        st.markdown(
            "**Story:** Mean brightness sits in a tight band across classes (about 68–73 in the notebook sample) "
            "and the spreads overlap. Brightness is mostly camera exposure and lighting, not disease. "
            "It cannot separate healthy from DR, so the model has to learn local patterns instead of a global brightness cue."
        )
        st.subheader("3. Average red channel per class")
        st.bar_chart(pd.Series(d["red_mean"]))
        st.markdown(
            "**Story:** Mean red is slightly lower for Severe_Proliferative than for No_DR "
            "(about 108 vs 122 in the notebook sample), but the overlap is large. "
            "Haemorrhages and lesions are red, yet a single colour average washes them out. "
            "That is why a convolutional model is needed: it can pick up small red dots and yellow patches "
            "that a dataset-wide red mean cannot."
        )
    except Exception as e:
        st.error(e)

else:
    st.title("Upload new data and retrain")
    st.write("Upload several labelled images, then press **Trigger retraining**. Retraining also starts on its own once 50 images are waiting.")
    label = st.selectbox("Class of these images", CLASSES)
    files = st.file_uploader("Upload multiple images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if files and st.button("Upload"):
        r = requests.post(f"{API}/upload", data={"label": label},
                          files=[("files", (x.name, x.getvalue(), x.type)) for x in files], timeout=600)
        st.json(r.json() if r.ok else {"error": r.text})
    if st.button("Trigger retraining"):
        r = requests.post(f"{API}/retrain", timeout=30)
        st.json(r.json() if r.ok else {"error": r.text})
    if st.button("Refresh status"):
        st.json(requests.get(f"{API}/status", timeout=10).json())
