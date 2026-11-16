import os

import requests
import streamlit as st

import io
from PIL import Image, ImageOps, UnidentifiedImageError

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
LOW_CONFIDENCE = 0.25
MAX_UPLOAD_MB = 10

st.set_page_config(page_title="Writer Identification", layout="wide")
st.title("Handwriting Writer Identification")
st.caption("Upload a photo of handwriting and the model will guess who wrote it.")


def predict(file):
    response = requests.post(
        f"{BACKEND_URL}/predict",
        files={"file": (file.name, file.getvalue(), file.type)},
        timeout=int(os.getenv("REQUEST_TIMEOUT", "300")),
    )
    response.raise_for_status()
    return response.json()["predictions"]

def load_image(data: bytes) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data)))


uploaded = st.file_uploader("Handwriting photo", type=["png", "jpg", "jpeg"])

if uploaded:
    st.image(uploaded)

    if st.button("Identify writer", type="primary"):
        try:
            with st.spinner("Analyzing..."):
                predictions = predict(uploaded)
        except requests.ConnectionError:
            st.error("Cannot reach the backend. Is it running?")
        except requests.Timeout:
            st.error("The backend took too long to respond.")
        except requests.HTTPError as error:
            st.error(f"Backend error: {error.response.status_code}")
        except (KeyError, ValueError):
            st.error("The backend returned an unexpected response.")
        else:
            if not predictions:
                st.error("The backend returned no prediction.")
            else:
                top = predictions[0]
                st.subheader(top["label"])
                st.caption(f"{top['confidence']:.1%} confidence")
                if top["confidence"] < LOW_CONFIDENCE:
                    st.warning("Low confidence. This may be a writer the model does not know.")

                st.write("Top matches")
                for item in predictions:
                    score = min(max(item["confidence"], 0.0), 1.0)
                    st.progress(score, text=f"{item['label']}  {score:.1%}")

if not uploaded:
    st.info("Upload a photo.")
    st.stop()

if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
    st.error(f"File is too large. Please use an image under {MAX_UPLOAD_MB}MB.")
    st.stop()

data = uploaded.getvalue()

try:
    image = load_image(data)
except (UnidentifiedImageError, OSError):
    st.error("This file could not be read as an image. Please try another one.")
    st.stop()
