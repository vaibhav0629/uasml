import random

from fastapi import FastAPI, File, UploadFile

app = FastAPI()

WRITERS = ["A250002", "A250013", "A250019", "A250032", "A250043"]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    await file.read()
    scores = sorted((random.random() for _ in WRITERS), reverse=True)
    total = sum(scores)
    labels = random.sample(WRITERS, len(WRITERS))
    top3 = list(zip(labels, scores))[:3]
    return {"predictions": [{"label": l, "confidence": s / total} for l, s in top3]}
