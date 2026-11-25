import numpy as np
import tensorflow as tf
from src.preprocessing import *

_model = None


def get_model(reload=False):
    global _model
    if _model is None or reload:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def predict_bytes(data: bytes):
    x = preprocess_bytes(data)
    prob = get_model()(x, training=False).numpy()[0]
    w = np.array([1, 1, load_meta().get("severe_weight", 1.0)])
    idx = int(np.argmax(prob * w))
    return {"prediction": CLASS_NAMES[idx],
            "confidence": float(prob[idx]),
            "probabilities": {c: float(p) for c, p in zip(CLASS_NAMES, prob)}}