import os, time, threading, json
from typing import List
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.preprocessing import *
from src import prediction
from src import model as trainer

app = FastAPI(title="Retina DR API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

START = time.time()
THRESHOLD = int(os.getenv("AUTO_RETRAIN_THRESHOLD", "50"))
state = {"status": "idle", "last_metrics": None, "last_error": None}
lock = threading.Lock()
_insights = {}


def _run_retrain():
    try:
        m = trainer.retrain()
        prediction.get_model(reload=True)
        state.update(status="idle", last_metrics=m, last_error=None)
    except Exception as e:
        state.update(status="idle", last_error=str(e))


def _start_retrain():
    with lock:
        if state["status"] == "retraining":
            return False
        state["status"] = "retraining"
    threading.Thread(target=_run_retrain, daemon=True).start()
    return True


def _pending():
    return len(list_images(UPLOAD_DIR)[0])


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/uptime")
def uptime():
    return {"uptime_seconds": round(time.time() - START, 1)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    try:
        return prediction.predict_bytes(data)
    except Exception as e:
        raise HTTPException(400, f"Could not process image: {e}")


@app.post("/upload")
async def upload(label: str = Form(...), files: List[UploadFile] = File(...)):
    if label not in CLASS_NAMES:
        raise HTTPException(400, f"label must be one of {CLASS_NAMES}")
    dest = os.path.join(UPLOAD_DIR, label)
    os.makedirs(dest, exist_ok=True)
    saved = 0
    for f in files:
        name = f"{int(time.time()*1000)}_{saved}_{os.path.basename(f.filename)}"
        with open(os.path.join(dest, name), "wb") as out:
            out.write(await f.read())
        saved += 1
    pending = _pending()
    auto = False
    if pending >= THRESHOLD:
        auto = _start_retrain()
    return {"saved": saved, "pending_total": pending,
            "auto_retrain_threshold": THRESHOLD, "auto_retrain_started": auto}


@app.post("/retrain")
def retrain():
    if _pending() == 0:
        raise HTTPException(400, "No uploaded data to retrain on")
    if not _start_retrain():
        raise HTTPException(409, "Retraining already running")
    return {"message": "Retraining started"}


@app.get("/status")
def status():
    return {**state, "pending_uploads": _pending(),
            "auto_retrain_threshold": THRESHOLD}


@app.get("/insights")
def insights():
    if _insights:
        return _insights
    counts, bright, red = {}, {}, {}
    for c in CLASS_NAMES:
        d = os.path.join(TRAIN_DIR, c)
        files = [f for f in os.listdir(d)] if os.path.isdir(d) else []
        counts[c] = len(files)
        b, r = [], []
        for f in files[:60]:
            arr = np.array(Image.open(os.path.join(d, f)).convert("RGB"))
            b.append(float(arr.mean()))
            r.append(float(arr[:, :, 0].mean()))
        bright[c] = float(np.mean(b)) if b else 0
        red[c] = float(np.mean(r)) if r else 0
    _insights.update({"counts": counts, "brightness": bright, "red_mean": red})
    return _insights