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
            st.subheader("Last retrain results"); st.json(s["last_metrics"])
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
    try:
        d = requests.get(f"{API}/insights", timeout=300).json()
        st.subheader("1. Images per class"); st.bar_chart(pd.Series(d["counts"]))
        st.subheader("2. Average brightness per class"); st.bar_chart(pd.Series(d["brightness"]))
        st.subheader("3. Average red channel per class"); st.bar_chart(pd.Series(d["red_mean"]))
        st.info("Write your interpretation of each chart here before submitting.")
    except Exception as e:
        st.error(e)

else:
    st.title("Upload new data and retrain")
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