import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import time
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
import io
import hashlib
from PIL import Image, UnidentifiedImageError

from model_def import build_densenet121
from preprocessing import _bg_scale, image_to_strip, shrink, windows_from_strip

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent / "model"))  # relatif ke main.py, bukan ke folder tempat uvicorn dijalankan
WINDOWS = int(os.getenv("WINDOWS", "9"))  # jumlah jendela per foto, sama dengan sel Testing (k=9)
TOP_K = int(os.getenv("TOP_K", "10"))
MAX_SIDE = int(os.getenv("MAX_SIDE", "0"))  # 0 = tanpa pengecilan, persis sel 5.14 (foto 12 MP bisa lambat)
MAX_BYTES = 10 * 1024 * 1024
ENSEMBLE = os.getenv("ENSEMBLE", "0") == "1"  # 1 = rata-rata semua fold (sama dengan pengujian di notebook); 0 = hanya fold terbaik

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = json.loads((MODEL_DIR / "config.json").read_text())
    labels = json.loads((MODEL_DIR / "labels.json").read_text())
    height, width = config["input_size"]

    folds = config["folds"]
    if not ENSEMBLE:
        folds = [max(folds, key=lambda f: f["val_acc"])]
    models = []
    for fold in folds:
        model = build_densenet121(height, width, len(labels))
        model.load_weights(MODEL_DIR / fold["file"])
        models.append(model)

    # Semua model dijalankan dalam SATU graph TensorFlow. Memanggil model(...) secara eager membuat Keras
    # mengeksekusi layer satu per satu lewat Python (lambat); tf.function mengompilasinya sekali saja.
    # Hasilnya identik bit-per-bit. Batch dibuat dinamis (None) supaya jumlah jendela yang berbeda
    # (strip pendek -> jendela lebih sedikit) tidak memicu kompilasi ulang.
    @tf.function(input_signature=[tf.TensorSpec([None, height, width, 1], tf.float32)])
    def run(x):
        return [m(x, training=False) for m in models]

    run(tf.zeros((WINDOWS, height, width, 1), tf.float32))  # warm-up: kompilasi terjadi sekarang, bukan di request pertama
    print("Model dimuat:", [f["file"] for f in folds], flush=True)

    state.update(models=models, run=run, labels=labels, size=(height, width), config=config)
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "input_size": state["size"], "classes": len(state["labels"]),
            "model": state["config"].get("model"), "n_models": len(state["models"])}


def infer(data: bytes):
    height, width = state["size"]
    t0 = time.perf_counter()
    img = Image.open(io.BytesIO(data))
    size = img.size
    strip = image_to_strip(shrink(img, MAX_SIDE), height, width)
    windows = windows_from_strip(strip, width, WINDOWS).astype("float32")
    t1 = time.perf_counter()
    # rata-rata antar jendela per model, lalu antar model (soft voting, sama dengan predict_strips_ensemble di notebook)
    outputs = state["run"](tf.constant(windows))
    probs = np.mean([np.asarray(o).mean(0) for o in outputs], axis=0)
    t2 = time.perf_counter()
    timing = {"preprocessing_s": round(t1 - t0, 2), "model_s": round(t2 - t1, 2), "n_models": len(state["models"]),
              "bg_downscale": _bg_scale(img.size[::-1])}  # 1 = jalur persis/lambat; >1 = jalur cepat
    print(f"[waktu] foto {size} | preprocessing {t1 - t0:.2f}s (bg 1/{timing['bg_downscale']}) | model x{len(state['models'])} {t2 - t1:.2f}s", flush=True)
    return size, strip, probs, timing


async def read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(400, "File kosong.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File terlalu besar (maks 10 MB).")
    return data


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await read_upload(file)
    try:
        # infer() berat (CPU) dan sinkron: jalankan di threadpool agar event loop (mis. /health) tidak ikut terblokir
        _, _, probs, timing = await run_in_threadpool(infer, data)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "File bukan gambar yang valid.")

    top = np.argsort(probs)[::-1][: min(TOP_K, len(probs))]
    return {
        "timing": timing,
        "predictions": [
            {"label": state["labels"][int(i)], "confidence": float(probs[i])} for i in top
        ]
    }


@app.post("/debug")
async def debug(file: UploadFile = File(...)):
    """Untuk membandingkan dengan Colab: hash strip, versi library, dan top-10."""
    import cv2, skimage, PIL, keras
    import tensorflow as tf

    data = await read_upload(file)
    try:
        size, strip, probs, _ = await run_in_threadpool(infer, data)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "File bukan gambar yang valid.")

    order = np.argsort(probs)[::-1][:10]
    return {
        "foto": list(size),
        "strip": list(strip.shape),
        "strip_sha256": hashlib.sha256(np.ascontiguousarray(strip).tobytes()).hexdigest()[:16],
        "label_awal": state["labels"][:3],
        "jumlah_kelas": len(state["labels"]),
        "model": state["config"],
        "versi": {"cv2": cv2.__version__, "skimage": skimage.__version__, "PIL": PIL.__version__,
                  "numpy": np.__version__, "tf": tf.__version__, "keras": keras.__version__},
        "top10": [{"label": state["labels"][int(i)], "prob": round(float(probs[i]), 4)} for i in order],
    }