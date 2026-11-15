import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
import io
import hashlib
from PIL import Image, UnidentifiedImageError

from model_def import build_densenet121
from preprocessing import image_to_strip, shrink, windows_from_strip

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent / "model"))  # relatif ke main.py, bukan ke folder tempat uvicorn dijalankan
WINDOWS = int(os.getenv("WINDOWS", "9"))  # jumlah jendela per foto, sama dengan sel Testing (k=9)
TOP_K = int(os.getenv("TOP_K", "10"))
MAX_SIDE = int(os.getenv("MAX_SIDE", "0"))  # 0 = tanpa pengecilan, persis sel 5.14 (foto 12 MP bisa lambat)
MAX_BYTES = 10 * 1024 * 1024

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = json.loads((MODEL_DIR / "config.json").read_text())
    labels = json.loads((MODEL_DIR / "labels.json").read_text())
    height, width = config["input_size"]

    model = build_densenet121(height, width, len(labels))
    model.load_weights(MODEL_DIR / "model.weights.h5")
    model(np.zeros((1, height, width, 1), dtype="float32"), training=False)  # warm-up

    state.update(model=model, labels=labels, size=(height, width), config=config)
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "input_size": state["size"], "classes": len(state["labels"]),
            "model": state["config"].get("model"), "seed": state["config"].get("seed")}


def infer(data: bytes):
    height, width = state["size"]
    img = Image.open(io.BytesIO(data))
    size = img.size
    strip = image_to_strip(shrink(img, MAX_SIDE), height, width)
    windows = windows_from_strip(strip, width, WINDOWS).astype("float32")
    probs = np.asarray(state["model"](windows, training=False)).mean(0)  # rata-rata antar jendela
    return size, strip, probs


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
        _, _, probs = infer(data)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "File bukan gambar yang valid.")

    top = np.argsort(probs)[::-1][: min(TOP_K, len(probs))]
    return {
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
        size, strip, probs = infer(data)
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
