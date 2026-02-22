# System Architecture

## Overview

**ARIA** (Autonomous Risk Intelligence Advisor) is a full-stack insurance bundle prediction system. Users submit client profiles through a conversational web UI; predictions are served by a FastAPI backend running a trained LightGBM classifier.

```
┌─────────────────────────────────────────────────┐
│                   Browser                        │
│          Next.js 14 Frontend (front/)            │
│   Landing │ Predict │ Batch │ Dashboard          │
└─────────────────────┬───────────────────────────┘
                      │ HTTP (REST)
                      │ NEXT_PUBLIC_API_URL
                      ▼
┌─────────────────────────────────────────────────┐
│         FastAPI Backend (app/)                   │
│         Uvicorn · Port 8000                      │
│                                                  │
│  Routes → Services → Core → model.pkl           │
└─────────────────────────────────────────────────┘
```

---

## Backend

### Entrypoint

`app/main.py` creates the FastAPI application, registers CORS middleware (allow all origins), mounts the four route groups, and manages the model lifecycle through an async lifespan context:

```
startup  → ModelStore.load()   # reads model.pkl from disk
shutdown → ModelStore.clear()  # releases memory
```

### Layer Diagram

```
HTTP Request
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  API Layer  (app/api/)                               │
│  deps.py → get_model()      raises 503 if not loaded │
│  routes/health.py           GET  /health             │
│  routes/metrics.py          GET  /metrics            │
│  routes/model.py            GET  /model/info         │
│                             GET  /model/features     │
│  routes/predict.py          POST /predict            │
│                             POST /predict/explain    │
│                             POST /predict/batch      │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  Service Layer  (app/services/)                      │
│  preprocessor.py   feature engineering (mirrors      │
│                    solution.py training pipeline)    │
│  predictor.py      target encoding + LightGBM infer  │
│  explainer.py      SHAP top-k feature extraction    │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  Core Layer  (app/core/)                             │
│  model_loader.py   ModelStore / LoadedModel          │
│  metrics.py        MetricsStore (in-memory)          │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  Config  (app/config.py)                             │
│  MODEL_PATH, N_CLASSES, confidence thresholds,       │
│  metrics rolling window                              │
└──────────────────────────────────────────────────────┘
```

### Key Components

#### ModelStore (`app/core/model_loader.py`)

Singleton-like class-level store. On startup it deserializes `model.pkl` (via `joblib`) into a `LoadedModel` dataclass containing:

| Field | Type | Purpose |
|---|---|---|
| `booster` | `lgb.Booster` | LightGBM model for inference |
| `feature_cols` | `list[str]` | Ordered feature column names |
| `employer_enc_map` | `dict` | Target encoding map for `Employer_ID` |
| `global_mean` | `float` | Fallback for unseen employer IDs |
| `global_prior` | `list[float]` | Per-class prior for unseen levels |
| `broker_multi` | `dict` | Per-class broker target encoding |
| `region_multi` | `dict` | Per-class region target encoding |
| `model_version` | `str` | Version tag (default `v7`) |
| `shap_explainer` | `shap.TreeExplainer` | Pre-initialized SHAP explainer |

#### MetricsStore (`app/core/metrics.py`)

In-memory, class-level counters. Updated after every prediction. Tracks total predictions, per-bundle distribution, rolling latency (window = 1000), and error count.

#### Prediction Pipeline

```
PredictRequest (JSON)
        │
        ▼
   preprocess(df)                 # app/services/preprocessor.py
        │  derive family-size, income brackets,
        │  claim ratios, seasonal flags, etc.
        ▼
   encode_and_predict(df, model)  # app/services/predictor.py
        │  Employer_ID  → target encoding
        │  Broker_ID    → per-class target encoding (10 columns)
        │  Region_Code  → per-class target encoding (10 columns)
        │  reindex to feature_cols
        │  booster.predict(X) → proba [N, 10]
        ▼
   argmax → predicted_bundle (0–9)
   max    → confidence  →  High / Medium / Low
        │
        ▼  (explain endpoint only)
   shap_explainer.shap_values(X)
   top_shap_features(k=5)        # app/services/explainer.py
```

#### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Model loaded status, uptime, feature/class count |
| `GET` | `/metrics` | Total preds, latency, bundle distribution, error rate |
| `GET` | `/model/info` | Version, feature count, hyperparameters, encoding strategy |
| `GET` | `/model/features` | Full feature importance map + top-10 list |
| `POST` | `/predict` | Single prediction → bundle, probabilities, confidence |
| `POST` | `/predict/explain` | Single prediction + top-5 SHAP features |
| `POST` | `/predict/batch` | CSV upload → per-row predictions + aggregate distribution |

#### Schemas (`app/schemas/schemas.py`)

All request/response shapes are Pydantic models. `PredictRequest` is open (`extra = "allow"`) to be tolerant of additional fields from different client profiles.

---

## Frontend

### Stack

| Concern | Technology |
|---|---|
| Framework | Next.js 14 (App Router) |
| Language | TypeScript |
| Styling | Tailwind CSS + custom CSS variables |
| Component library | shadcn/ui (Radix primitives) |
| Data fetching | SWR (stale-while-revalidate) |
| Charts | Recharts |
| Package manager | pnpm |

### Pages

```
front/app/
├── page.tsx         # Landing — ARIA greeting + live metrics cards
├── predict/
│   └── page.tsx     # Chat-style multi-step prediction form
├── batch/
│   └── page.tsx     # CSV drag-and-drop → batch results + pie chart
└── dashboard/
    └── page.tsx     # Live metrics, feature importance, model info
```

#### Landing (`/`)

Displays ARIA's animated avatar and a typewriter greeting. Polls `/metrics` every 10 s via SWR to show live prediction counts and latency. Links to `/predict` and `/batch`.

#### Predict (`/predict`)

Conversational, step-by-step form driven by ARIA. Three sequential widget steps:

1. **Demographics** — family composition, income, employment
2. **Risk** — claims history, years without claims, existing policyholder
3. **Broker** — acquisition channel, broker type, deductible tier, policy details

After the final step the form calls `POST /predict/explain`, renders the predicted bundle with probabilities, confidence label, and top-5 SHAP features. ARIA's avatar state (`idle | thinking | happy | nervous | squinting`) changes dynamically based on prediction confidence.

#### Batch (`/batch`)

Drag-and-drop CSV uploader. Sends the file to `POST /predict/batch` as `multipart/form-data`. Displays paginated results table (20 rows/page) and a Recharts pie chart of bundle distribution.

#### Dashboard (`/dashboard`)

Real-time observability panel (polls every 5 s):
- Prediction count, average latency, error rate, active bundles
- Bar chart of bundle distribution (coloured per bundle)
- Bar chart of top-10 feature importances
- Model metadata cards (version, classes, encoding strategy)

### API Client (`front/lib/api.ts`)

Single module exposing typed wrappers around every backend endpoint. Base URL configured via `NEXT_PUBLIC_API_URL` environment variable (defaults to `http://localhost:8000`).

```
api.getHealth()          → HealthResponse
api.getMetrics()         → MetricsResponse
api.getModelInfo()       → ModelInfoResponse
api.getModelFeatures()   → ModelFeaturesResponse
api.predict(req)         → PredictResponse
api.predictExplain(req)  → ExplainResponse
api.batchPredict(file)   → BatchResponse
```

### Component Tree (Predict page)

```
PredictPage
├── Starfield            (canvas background)
├── AriaAvatar           (animated SVG, state-driven)
├── ChatMessage[]        (scrollable message list)
│   ├── DemographicsInput   (step 1 widget)
│   ├── RiskInput           (step 2 widget)
│   ├── BrokerInput         (step 3 widget)
│   └── PredictionResult    (final result widget)
```

---

## Data Flow — Single Prediction

```
User fills Demographics / Risk / Broker forms
          │
          ▼
api.predictExplain(req)  [POST /predict/explain]
          │
          ▼  FastAPI
preprocess(df)    →  encode_and_predict()  →  SHAP
          │
          └──► MetricsStore.record_prediction()
          │
          ▼
ExplainResponse {bundle, probabilities, confidence, shap[5]}
          │
          ▼
PredictionResult component renders result card
AriaAvatar switches state based on confidence_label
```

---

## Infrastructure

### Docker (Backend only)

```dockerfile
FROM python:3.11-slim
# installs libgomp (OpenMP required by LightGBM)
# pip installs requirements.txt
# copies app/ + model.pkl
# CMD: uvicorn app.main:app --host 0.0.0.0 --port 8000
EXPOSE 8000
```

`docker-compose.yml` mounts `./model.pkl` into the container at `/app/model.pkl` and passes `MODEL_PATH` as an environment variable.

The frontend is not containerised — it runs as a standard Next.js dev/build server separately.

### Environment Variables

| Variable | Service | Default | Purpose |
|---|---|---|---|
| `MODEL_PATH` | Backend | `./model.pkl` | Path to serialised model artifact |
| `NEXT_PUBLIC_API_URL` | Frontend | `http://localhost:8000` | Backend base URL |

---

## ML Model

| Property | Value |
|---|---|
| Algorithm | LightGBM (gradient-boosted trees) |
| Task | Multi-class classification (10 bundles, 0–9) |
| Artifact | `model.pkl` serialised with `joblib` |
| Explainability | SHAP `TreeExplainer` (pre-initialised at startup) |
| Encoding | Target encoding for `Employer_ID`, `Broker_ID`, `Region_Code` |
| Confidence bands | High ≥ 0.6 · Medium ≥ 0.35 · Low < 0.35 |

Feature engineering in `app/services/preprocessor.py` mirrors the training pipeline in `solution (3).py` to avoid training/serving skew.

---

## Directory Reference

```
back/
├── app/                    Backend (Python / FastAPI)
│   ├── main.py             App factory, lifespan, CORS, router mounts
│   ├── config.py           Centralised constants
│   ├── api/
│   │   ├── deps.py         get_model() dependency
│   │   └── routes/         One file per route group
│   ├── core/
│   │   ├── model_loader.py ModelStore + LoadedModel
│   │   └── metrics.py      MetricsStore
│   ├── schemas/
│   │   └── schemas.py      Pydantic request/response models
│   └── services/
│       ├── preprocessor.py Feature engineering
│       ├── predictor.py    Encoding + inference
│       └── explainer.py    SHAP extraction
├── front/                  Frontend (Next.js / TypeScript)
│   ├── app/                App Router pages
│   ├── components/         Shared + page-level components
│   ├── hooks/              use-mobile, use-toast, use-typewriter
│   └── lib/
│       ├── api.ts          Typed API client
│       └── utils.ts        Tailwind helpers
├── model.pkl               Serialised LightGBM artifact (not in repo)
├── Dockerfile              Backend container image
├── docker-compose.yml      Single-service compose file
└── requirements.txt        Python dependencies
```

## Entry Points

- **Backend (Docker):** `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- **Backend (local):** `python -m app.main` or `uvicorn app.main:app --reload`
- **Backend (legacy):** `api.py` re-exports `app` for `uvicorn api:app`
- **Frontend:** `pnpm dev` (Next.js dev server, default port 3000)
