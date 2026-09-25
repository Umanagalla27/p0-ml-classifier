# P0: Production Text Classifier — Classic ML vs Transformer vs LLM

[![CI Pipeline](https://github.com/Umanagalla27/p0-ml-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Umanagalla27/p0-ml-classifier/actions)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1.svg?logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-7-DC382D.svg?logo=redis&logoColor=white)

A benchmarked, containerized classification microservice comparing **Classic ML (TF-IDF + Logistic Regression)**, **Fine-Tuned Transformers (DistilBERT)**, and **In-Context Few-Shot LLMs** on the AG News dataset (120,000 training samples across 4 classes).

---

## 🏛️ System Architecture

```text
User / Client Request
        │
        ▼
┌──────────────────┐
│ FastAPI Gateway  │ ◄── [Pydantic v2 Schema Validation]
└─────────┬────────┘
          │
    [Cache Check]
          ├──────────────► [Redis 7] (Cache Hit: Latency < 2ms)
          │
     (Cache Miss)
          ▼
┌──────────────────┐
│ scikit-learn /   │ ───► Predicted Label + Confidence Score
│ DistilBERT Model │
└─────────┬────────┘
          │
     [Async Log]
          ▼
┌──────────────────┐
│  PostgreSQL 16   │ ───► Logs (id, text, label, confidence, latency_ms, timestamp)
└──────────────────┘
```

---

## 📊 Paradigm Benchmark & Trade-Off Comparison

| Metric | Classic ML (TF-IDF + LogReg) | DistilBERT (Fine-Tuned) | Few-Shot LLM (In-Context) |
|---|---|---|---|
| **Accuracy** | **92.09%** | **94.61%** | 91.50% - 93.50% |
| **Macro F1** | **92.07%** | **94.61%** | 91.80% |
| **p50 Latency** | **~5 ms** | ~25 ms (GPU) / ~80 ms (CPU) | ~450 ms - 900 ms |
| **Cost / 1M queries**| **$0 (CPU)** | ~$15 (Hosted GPU instance) | ~$150 - $400 (API Tokens) |
| **Cold Start** | **< 0.1s** | ~2 - 5s | Instant (Managed API) |
| **Training Time** | **45 sec (CPU)** | 7 min (T4 GPU) | 0 (Prompting only) |

### Engineering Decision Matrix
1. **Choose Classic ML** for high-throughput, latency-critical applications (<10ms SLA) where 92% accuracy satisfies the business case and cloud spend must stay near-zero.
2. **Choose Fine-Tuned Transformer** for mission-critical domain workflows requiring maximum accuracy (94.6%+) with strict data privacy within an enterprise VPC.
3. **Choose LLM Prompting** for rapid prototyping, cold-start domains with zero labeled training data, or workflows with frequently shifting label taxonomies.

---

## 🚀 Quick Start (Local Docker Stack)

### Prerequisites
- Docker Engine & Docker Compose
- Python 3.12+ (for local development)

### Run with One Command
```bash
docker compose up --build -d
```

### Endpoints
- **Health Check**: `GET http://localhost:8000/health`
- **Inference**: `POST http://localhost:8000/predict`
- **Swagger Docs**: `http://localhost:8000/docs`
