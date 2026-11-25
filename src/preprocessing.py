import os, io, json
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "retina_model.keras")
META_PATH = os.path.join(MODELS_DIR, "meta.json")

IMG_SIZE = (224, 224)
CLASS_NAMES = ["Mild_Moderate", "No_DR", "Severe_Proliferative"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
EXTS = (".png", ".jpg", ".jpeg")


def load_meta():
    try:
        with open(META_PATH) as f:
            return json.load(f)
    except Exception:
        return {"class_names": CLASS_NAMES, "severe_weight": 1.0}


def list_images(root):
    paths, labels = [], []
    for c in CLASS_NAMES:
        d = os.path.join(root, c)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith(EXTS):
                paths.append(os.path.join(d, f))
                labels.append(LABEL_TO_IDX[c])
    return paths, labels


def _load(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img.set_shape((224, 224, 3))
    return img, label


def make_dataset(paths, labels, batch=32, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=42)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def preprocess_bytes(data: bytes):
    img = Image.open(io.BytesIO(data)).convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.array(img, dtype="float32"), 0)