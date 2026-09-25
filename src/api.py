import os
import json
import time
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
import psycopg2
import redis
import joblib
import numpy as np

app = FastAPI(
    title="P0 Text Classifier",
    description="AG News classifier: ML vs Transformer vs LLM comparison",
    version="0.1.0"
)

# Global model holder
model_pipeline = None
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

def get_db_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL", "postgresql://uma:password@localhost:5432/classifier"))

redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True)

class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text cannot be empty")
        return v.strip()

class PredictResponse(BaseModel):
    label: str
    confidence: float
    model_used: str
    latency_ms: float
    cached: bool = False

def setup_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            input_text TEXT NOT NULL,
            label VARCHAR(50) NOT NULL,
            confidence FLOAT NOT NULL,
            model_used VARCHAR(50) NOT NULL,
            latency_ms FLOAT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

@app.on_event("startup")
def on_startup():
    global model_pipeline
    setup_db()
    
    # Load model artifact
    model_path = os.getenv("MODEL_PATH", "models/tfidf_logreg.joblib")
    if os.path.exists(model_path):
        model_pipeline = joblib.load(model_path)
        print(f"Loaded ML model pipeline from {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}. Placeholder mode active.")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model_pipeline is not None
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start = time.perf_counter()
    cache_key = f"pred:{request.text}"

    # 1. Check Redis Cache
    try:
        cached_result = redis_client.get(cache_key)
        if cached_result:
            data = json.loads(cached_result)
            latency_ms = (time.perf_counter() - start) * 1000
            return PredictResponse(
                label=data["label"],
                confidence=data["confidence"],
                model_used=data["model_used"],
                latency_ms=round(latency_ms, 2),
                cached=True
            )
    except Exception as e:
        print(f"Redis cache check failed: {e}")

    # 2. Real Model Inference
    if model_pipeline is not None:
        # predict_proba returns confidence scores for each class
        probs = model_pipeline.predict_proba([request.text])[0]
        pred_idx = int(np.argmax(probs))
        label = LABEL_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        model_name = "sklearn-tfidf-logreg"
    else:
        label = "Business"
        confidence = 0.50
        model_name = "placeholder"

    latency_ms = (time.perf_counter() - start) * 1000

    # 3. Cache in Redis (TTL: 1 hour)
    try:
        redis_client.setex(
            cache_key,
            3600,
            json.dumps({"label": label, "confidence": round(confidence, 4), "model_used": model_name})
        )
    except Exception as e:
        print(f"Redis cache set failed: {e}")

    # 4. Log to PostgreSQL
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO predictions (input_text, label, confidence, model_used, latency_ms) VALUES (%s, %s, %s, %s, %s)",
            (request.text, label, confidence, model_name, latency_ms)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"DB log failed: {e}")

    return PredictResponse(
        label=label,
        confidence=round(confidence, 4),
        model_used=model_name,
        latency_ms=round(latency_ms, 2),
        cached=False
    )
