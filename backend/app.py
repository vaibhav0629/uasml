import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from model_def import build_densenet121
from preprocessing import image_to_strip, windows_from_strip

HERE = Path(__file__).parent
# Cari folder model: ./model, atau ../backend/model (struktur repo uasml)
MODEL_DIR = Path(os.getenv("MODEL_DIR", "")) if os.getenv("MODEL_DIR") else next(
    (p for p in (HERE / "model", HERE.parent / "backend" / "model") if (p / "model.weights.h5").exists()),
    HERE / "model",
)
WINDOWS = 9          # jumlah jendela per foto (sama dengan notebook, k=9)
TOP_K = 10
LOW_CONFIDENCE = 0.25
MAX_MB = 10

st.set_page_config(page_title="Writer Identification", layout="wide")


@st.cache_resource(show_spinner="Memuat model DenseNet121...")
def load_model():
    config = json.loads((MODEL_DIR / "config.json").read_text())
    labels = json.loads((MODEL_DIR / "labels.json").read_text())
    h, w = config["input_size"]
    model = build_densenet121(h, w, len(labels))
    model.load_weights(MODEL_DIR / "model.weights.h5")
    model(np.zeros((1, h, w, 1), dtype="float32"), training=False)  # warm-up
    return model, labels, (h, w), config


def predict(img: Image.Image, model, labels, size):
    h, w = size
    t0 = time.perf_counter()
    strip = image_to_strip(img, h, w)
    windows = windows_from_strip(strip, w, WINDOWS).astype("float32")
    t1 = time.perf_counter()
    probs = np.asarray(model(windows, training=False)).mean(0)  # rata-rata antar jendela
    t2 = time.perf_counter()
    top = np.argsort(probs)[::-1][:TOP_K]
    return [(labels[i], float(probs[i])) for i in top], strip, t1 - t0, t2 - t1


model, labels, size, config = load_model()

st.title("Handwriting Writer Identification")
st.caption(f"{config['model']} (seed {config.get('seed')}, val acc {config.get('val_acc', 0):.1%}) · {len(labels)} penulis")

uploaded = st.file_uploader("Foto tulisan tangan", type=["png", "jpg", "jpeg"])
if not uploaded:
    st.info("Upload sebuah foto.")
    st.stop()
if uploaded.size > MAX_MB * 1024 * 1024:
    st.error(f"File terlalu besar (maks {MAX_MB} MB).")
    st.stop()

try:
    image = ImageOps.exif_transpose(Image.open(uploaded))
except (UnidentifiedImageError, OSError):
    st.error("File tidak bisa dibaca sebagai gambar.")
    st.stop()

left, right = st.columns(2)
left.image(image, caption=f"{image.width}×{image.height}", use_container_width=True)

if st.button("Identifikasi penulis", type="primary"):
    with st.spinner("Menganalisis..."):
        preds, strip, t_pre, t_inf = predict(image, model, labels, size)
    top_label, top_conf = preds[0]
    with right:
        st.subheader(top_label)
        st.caption(f"{top_conf:.1%} confidence · preprocessing {t_pre:.2f}s · model {t_inf:.2f}s")
        if top_conf < LOW_CONFIDENCE:
            st.warning("Confidence rendah. Mungkin penulis ini tidak dikenal model.")
        for label, p in preds:
            st.progress(min(max(p, 0.0), 1.0), text=f"{label}  {p:.1%}")
    with st.expander("Lihat strip hasil preprocessing"):
        st.image(strip, use_container_width=True, clamp=True)
