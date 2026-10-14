"""FastAPI backend.  Run: uvicorn app:app --reload  ->  http://localhost:8000"""
import json, time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).parent
MODELS = BASE / "models"
FRONTEND = BASE.parent / "frontend"

app = FastAPI(title="Biome-Locked Forest Cover Classifier", version="2.0")


def metrics():
    p = MODELS / "metrics.json"
    if not p.exists():
        raise HTTPException(503, "Models not trained yet. Run: python train.py --quick")
    return json.loads(p.read_text())


@lru_cache(maxsize=8)
def load_model(name: str):
    path = MODELS / f"{name.replace(' ', '_').lower()}.joblib"
    if not path.exists():
        raise HTTPException(404, f"Model '{name}' not found")
    return joblib.load(path)


class Row(BaseModel):
    elevation: float = Field(..., ge=1800, le=4000)
    aspect: float = Field(..., ge=0, le=360)
    slope: float = Field(..., ge=0, le=90)
    h_dist_hydrology: float = Field(..., ge=0)
    v_dist_hydrology: float
    h_dist_roadways: float = Field(..., ge=0)
    hillshade_9am: float = Field(..., ge=0, le=255)
    hillshade_noon: float = Field(..., ge=0, le=255)
    hillshade_3pm: float = Field(..., ge=0, le=255)
    h_dist_fire: float = Field(..., ge=0)
    wilderness_area: int = Field(..., ge=1, le=4)
    soil_type: int = Field(..., ge=1, le=40)


class PredictRequest(Row):
    model: str = "Random Forest"


class BatchRequest(BaseModel):
    model: str = "Random Forest"
    rows: list[Row] = Field(..., min_length=1, max_length=5000)


def vectorise(r: Row):
    wild = [0.0] * 4; wild[r.wilderness_area - 1] = 1.0
    soil = [0.0] * 40; soil[r.soil_type - 1] = 1.0
    return np.array([[r.elevation, r.aspect, r.slope, r.h_dist_hydrology, r.v_dist_hydrology,
                      r.h_dist_roadways, r.hillshade_9am, r.hillshade_noon, r.hillshade_3pm,
                      r.h_dist_fire, *wild, *soil]])


def run(name, x, classes):
    model = load_model(name)  # load time is excluded from the latency figure
    t = time.perf_counter()
    proba = model.predict_proba(x)[0]
    ms = (time.perf_counter() - t) * 1000
    top = int(np.argmax(proba))
    return {"model": name, "prediction": classes[top], "cover_type_id": top + 1,
            "confidence": round(float(proba[top]) * 100, 2), "latency_ms": round(ms, 2),
            "probabilities": {c: round(float(p) * 100, 2) for c, p in zip(classes, proba)}}


@app.get("/api/metrics")
def get_metrics():
    return metrics()


@app.post("/api/predict")
def predict(req: PredictRequest):
    return run(req.model, vectorise(req), metrics()["classes"])


@app.post("/api/predict_all")
def predict_all(req: PredictRequest):
    m, x = metrics(), vectorise(req)
    return {"results": [run(r["model"], x, m["classes"]) for r in m["results"]]}


@app.post("/api/predict_batch")
def predict_batch(req: BatchRequest):
    classes = metrics()["classes"]
    X = np.vstack([vectorise(r) for r in req.rows])
    model = load_model(req.model)
    t = time.perf_counter()
    proba = model.predict_proba(X)
    ms = (time.perf_counter() - t) * 1000
    top = proba.argmax(1)
    return {"model": req.model, "count": len(top), "total_ms": round(ms, 1),
            "ms_per_1k": round(ms / len(top) * 1000, 1),
            "results": [{"row": i + 1, "prediction": classes[k], "cover_type_id": int(k) + 1,
                         "confidence": round(float(proba[i, k]) * 100, 2)} for i, k in enumerate(top)]}


@app.get("/api/health")
def health():
    return {"status": "ok", "trained": (MODELS / "metrics.json").exists()}


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")