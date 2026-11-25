import os, json, shutil, time, random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from src.preprocessing import *


def build_model():
    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomBrightness(0.1),
    ], name="augment")
    base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = augment(inputs)
    x = preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(3, activation="softmax",
                           kernel_regularizer=regularizers.l2(1e-4))(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def class_weights(labels):
    classes = np.unique(labels)
    w = compute_class_weight("balanced", classes=classes, y=np.array(labels))
    cw = {i: 1.0 for i in range(len(CLASS_NAMES))}
    cw.update({int(c): float(x) for c, x in zip(classes, w)})
    return cw


def evaluate(model):
    p, l = list_images(TEST_DIR)
    if not p:
        return {}
    prob = model.predict(make_dataset(p, l), verbose=0)
    w = np.array([1, 1, load_meta().get("severe_weight", 1.0)])
    pred = np.argmax(prob * w, axis=1)
    return {"accuracy": float(accuracy_score(l, pred)),
            "macro_f1": float(f1_score(l, pred, average="macro")),
            "n_test": len(l)}


def train(epochs=15):
    p, l = list_images(TRAIN_DIR)
    tr_p, va_p, tr_l, va_l = train_test_split(p, l, test_size=0.15, stratify=l, random_state=42)
    model = build_model()
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    model.fit(make_dataset(tr_p, tr_l, training=True), validation_data=make_dataset(va_p, va_l),
              epochs=epochs, class_weight=class_weights(tr_l), callbacks=[es])
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    return evaluate(model)


def retrain(epochs=5, per_class_old=150):
    new_p, new_l = list_images(UPLOAD_DIR)
    if not new_p:
        raise ValueError("No uploaded images found in data/uploads")
    old_p, old_l = list_images(TRAIN_DIR)

    rng = random.Random(42)
    keep = []
    for c in range(len(CLASS_NAMES)):
        idx = [i for i, x in enumerate(old_l) if x == c]
        rng.shuffle(idx)
        keep += idx[:per_class_old]
    paths = [old_p[i] for i in keep] + new_p
    labels = [old_l[i] for i in keep] + new_l

    try:
        tr_p, va_p, tr_l, va_l = train_test_split(paths, labels, test_size=0.15,
                                                  stratify=labels, random_state=42)
    except ValueError:
        tr_p, va_p, tr_l, va_l = train_test_split(paths, labels, test_size=0.15, random_state=42)

    model = tf.keras.models.load_model(MODEL_PATH)   # pre-trained custom model
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    model.fit(make_dataset(tr_p, tr_l, training=True), validation_data=make_dataset(va_p, va_l),
              epochs=epochs, class_weight=class_weights(tr_l), callbacks=[es], verbose=2)

    metrics = evaluate(model)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    shutil.copy(MODEL_PATH, os.path.join(MODELS_DIR, f"backup_{stamp}.keras"))
    model.save(MODEL_PATH)

    for p, l in zip(new_p, new_l):
        dest = os.path.join(TRAIN_DIR, CLASS_NAMES[l])
        os.makedirs(dest, exist_ok=True)
        shutil.move(p, os.path.join(dest, f"{stamp}_{os.path.basename(p)}"))

    metrics.update({"trained_on_new_images": len(new_p), "timestamp": stamp})
    with open(os.path.join(MODELS_DIR, "last_retrain.json"), "w") as f:
        json.dump(metrics, f)
    return metrics