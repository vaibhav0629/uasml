import hashlib
import io
import os
import time

import requests
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
LOW_CONFIDENCE = 0.25
MAX_UPLOAD_MB = 10
PREVIEW_SIDE = 1200  # pratinjau cukup kecil; foto asli tetap dikirim utuh ke backend

st.set_page_config(page_title="Writer Identification", layout="wide")
st.title("Handwriting Writer Identification")
st.caption("Upload a photo of handwriting and the model will guess who wrote it.")


@st.cache_resource
def http_session():
    return requests.Session()  # koneksi ke backend dipakai ulang


def predict(name, data, mime):
    response = http_session().post(
        f"{BACKEND_URL}/predict",
        files={"file": (name, data, mime)},
        timeout=int(os.getenv("REQUEST_TIMEOUT", "300")),
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False, max_entries=8)
def make_preview(data: bytes):
    """Decode + perkecil sekali saja per file; rerun Streamlit tidak mengulanginya."""
    with Image.open(io.BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image)
        size = image.size
        image.thumbnail((PREVIEW_SIDE, PREVIEW_SIDE))
        return image.copy(), size


uploaded = st.file_uploader("Handwriting photo", type=["png", "jpg", "jpeg"])

if not uploaded:
    st.info("Upload a photo.")
    st.stop()

if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
    st.error(f"File is too large. Please use an image under {MAX_UPLOAD_MB}MB.")
    st.stop()

data = uploaded.getvalue()

try:
    preview, original_size = make_preview(data)
except (UnidentifiedImageError, OSError):
    st.error("This file could not be read as an image. Please try another one.")
    st.stop()

st.image(preview, caption=f"{original_size[0]}×{original_size[1]}")

# Hasil disimpan per file: klik ulang / rerun Streamlit tidak mengulang analisis yang sama.
file_key = hashlib.sha1(data).hexdigest()
cache = st.session_state.setdefault("predictions", {})

if st.button("Identify writer", type="primary") and file_key not in cache:
    try:
        with st.spinner("Analyzing..."):
            started = time.perf_counter()
            result = predict(uploaded.name, data, uploaded.type)
            result["total_s"] = round(time.perf_counter() - started, 2)
            cache[file_key] = result
    except requests.ConnectionError:
        st.error("Cannot reach the backend. Is it running?")
    except requests.Timeout:
        st.error("The backend took too long to respond.")
    except requests.HTTPError as error:
        st.error(f"Backend error: {error.response.status_code}")
    except (KeyError, ValueError):
        st.error("The backend returned an unexpected response.")

result = cache.get(file_key)
if result is not None:
    predictions = result["predictions"]
    if not predictions:
        st.error("The backend returned no prediction.")
    else:
        top = predictions[0]
        st.subheader(top["label"])
        st.caption(f"{top['confidence']:.1%} confidence")
        timing = result.get("timing") or {}
        st.caption(
            f"Total {result['total_s']}s (upload + analisis) · preprocessing {timing.get('preprocessing_s')}s "
            f"(bg 1/{timing.get('bg_downscale')}) · model {timing.get('model_s')}s x{timing.get('n_models')}"
        )
        if top["confidence"] < LOW_CONFIDENCE:
            st.warning("Low confidence. This may be a writer the model does not know.")

        st.write("Top matches")
        for item in predictions:
            score = min(max(item["confidence"], 0.0), 1.0)
            st.progress(score, text=f"{item['label']}  {score:.1%}")