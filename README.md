# ARIA — Insurance Bundle Predictor

**ARIA** (Autonomous Risk Intelligence Advisor) is a full-stack machine learning application that predicts which insurance coverage bundle a client will purchase. A conversational web UI guides users through collecting a client profile, which is passed to a trained LightGBM classifier that returns a predicted bundle with confidence scoring and SHAP-based explanations.

🚀 **Live Demo:** (<https://amusing-connection-production.up.railway.app>)

---

## Features

- **Conversational predict UI** — ARIA guides brokers through a step-by-step client intake form and returns a prediction with a natural-language explanation
- **Batch predictions** — drag-and-drop CSV uploader for bulk inference with paginated results and a bundle distribution chart
- **Live dashboard** — real-time observability panel showing prediction counts, latency, error rate, feature importances, and model metadata
- **SHAP explanations** — top-5 SHAP features are returned alongside every prediction to explain the model's reasoning
- **Confidence bands** — predictions are labelled High (≥ 60%), Medium (≥ 35%), or Low (< 35%)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, Radix UI, Recharts |
| Backend | FastAPI, Uvicorn, Python 3.11 |
| ML | LightGBM, SHAP, scikit-learn, pandas |
| Containerisation | Docker, Docker Compose |

---

## Project Structure

```
├── app/
│   ├── backend/                    FastAPI application
│   │   ├── app/
│   │   │   ├── main.py             App factory & lifespan
│   │   │   ├── config.py           Centralised constants
│   │   │   ├── api/
│   │   │   │   ├── deps.py         Shared dependencies
│   │   │   │   └── routes/         health · metrics · model · predict
│   │   │   ├── core/
│   │   │   │   ├── model_loader.py LightGBM + preprocessor loader
│   │   │   │   └── metrics.py      In-memory metrics store
│   │   │   ├── schemas/
│   │   │   │   └── schemas.py      Pydantic request/response models
│   │   │   └── services/
│   │   │       ├── preprocessor.py 62-feature pipeline
│   │   │       ├── predictor.py    Inference + confidence bands
│   │   │       └── explainer.py    SHAP TreeExplainer wrapper
│   │   ├── model.pkl               Trained LightGBM (1.86 MB)
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── requirements-dev.txt
│   ├── frontend/                   Next.js 14 application
│   │   ├── app/                    App Router
│   │   │   ├── page.tsx            Home / landing
│   │   │   ├── predict/            Conversational predict UI
│   │   │   ├── batch/              CSV batch upload & results
│   │   │   └── dashboard/          Live observability panel
│   │   ├── components/
│   │   │   ├── aria-avatar.tsx     Animated ARIA avatar
│   │   │   ├── nav.tsx             Top navigation bar
│   │   │   ├── starfield.tsx       Background particle effect
│   │   │   ├── predict/            Step-form input components
│   │   │   └── ui/                 Radix-based component library
│   │   ├── hooks/                  use-mobile · use-toast · use-typewriter
│   │   ├── lib/
│   │   │   ├── api.ts              Typed API client
│   │   │   └── utils.ts            Shared utilities
│   │   └── public/                 Icons & static assets
│   ├── retrain/
│   │   ├── retrain.py              Full retraining pipeline
│   │   └── requirements.txt
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_api.py
│   │   ├── test_predict.py
│   │   └── test_preprocess.py
│   ├── .github/workflows/
│   │   ├── ci.yml                  Lint & test on push / PR
│   │   ├── cd.yml                  Deploy backend (Railway) + frontend (Vercel)
│   │   └── retrain.yml             Manual model retrain trigger
│   ├── ARCHITECTURE.md             Layer diagrams & full directory reference
│   └── docker-compose.yml
├── models/
│   ├── README.md                   Full technical report (ML side)
│   ├── SOLUTION.md                 One-page key-decisions summary
│   ├── DataQuest_Technical_Report.tex  LaTeX technical report
│   ├── Data/
│   │   ├── raw/                    train.csv · test.csv (competition data)
│   │   └── processed/              train_clean.csv · test_clean.csv
│   ├── notebook/
│   │   ├── assets/                 EDA & model visualisation PNGs
│   │   ├── data_visualization.ipynb  EDA — 69 cells of analysis
│   │   ├── cleaning.ipynb          Cleaning pipeline → processed CSVs
│   │   ├── modelling.ipynb         Initial exploration (v1–v2)
│   │   ├── modelling_best_model_v7.ipynb  Best model + SHAP explainability
│   │   ├── technical_report.ipynb  Narrative report with inline visualisations
│   │   └── submit.ipynb            Submission generation helper
│   ├── submissions/
│   │   ├── submission_v1/          150 trees + threshold tuning  (score 0.3331)
│   │   ├── submission_v2/          75 trees, raw argmax          (score 0.4583)
│   │   ├── submission_v3/          50 trees, latency test        (score 0.4253)
│   │   ├── submission_v4/          150 shallow trees depth=4     (score 0.4188)
│   │   ├── submission_v5/          Full-train encoding fix        (score 0.4640)
│   │   ├── submission_v6/          Intermediate iteration
│   │   └── submission_v7/          All improvements — best model  (score 0.4875)
│   └── test/
│       └── test_latency.py         Local latency benchmark
├── README.md                       ← You are here
├── technical_report.pdf            Compiled PDF of the LaTeX report
└── DataQuest-Brief-Document.pdf    Original competition brief
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Model status, uptime, feature/class count |
| `GET` | `/metrics` | Total predictions, latency, bundle distribution, error rate |
| `GET` | `/model/info` | Version, hyperparameters, encoding strategy |
| `GET` | `/model/features` | Full feature importance map + top-10 list |
| `POST` | `/predict` | Single prediction → bundle, probabilities, confidence |
| `POST` | `/predict/explain` | Single prediction + top-5 SHAP features |
| `POST` | `/predict/batch` | CSV upload → per-row predictions + aggregate distribution |

---

## ML Model

| Property | Value |
|---|---|
| Algorithm | LightGBM (multiclass softmax) |
| Task | 10-class classification (insurance bundles 0–9) |
| Features | 62 (44 cleaned + 20 target-encoded) |
| Resampling | 3-step ROS → SMOTE for minority classes |
| CV | 5-fold StratifiedKFold — OOF Macro F1 = **0.6418** |
| Explainability | SHAP TreeExplainer |
| Final Score | **0.4875** (Macro F1 = 0.5577 on leaderboard) |

The 10 target bundles are: `Auto_Comprehensive`, `Auto_Liability_Basic`, `Basic_Health`, `Family_Comprehensive`, `Health_Dental_Vision`, `Home_Premium`, `Home_Standard`, `Premium_Health_Life`, `Renter_Basic`, `Renter_Premium`.

---

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) & Docker Compose
- [Node.js 18+](https://nodejs.org/) and [pnpm](https://pnpm.io/) (for local frontend dev)
- Python 3.11+ (for local backend dev)

### Run with Docker

```bash
# From the app/ directory
docker compose up --build
```

The API will be available at `http://localhost:8000`.

### Run Locally

**Backend:**

```bash
cd app/backend
pip install -r requirements.txt
# Place model.pkl in app/backend/
uvicorn app.main:app --reload --port 8000
```

**Frontend:**

```bash
cd app/frontend
pnpm install
pnpm dev
```

The frontend will be available at `http://localhost:3000`. Set `NEXT_PUBLIC_API_URL=http://localhost:8000` in a `.env.local` file if the default doesn't match.

---

## Running Tests

```bash
cd app
pip install -r backend/requirements-dev.txt
pytest tests/
```

---

## Notebooks

See [models/notebook/MODELLING_README.md](models/notebook/MODELLING_README.md) for a guide to the notebooks.

| Notebook | Purpose |
|---|---|
| `cleaning.ipynb` | Data cleaning and feature engineering pipeline |
| `data_visualization.ipynb` | Exploratory data analysis |
| `modelling.ipynb` | Initial model exploration (v1–v2) |
| `modelling_best_model_v7.ipynb` | Final model training + SHAP explainability |
| `technical_report.ipynb` | Full technical report with visualisations |

---

## MLOps

Three GitHub Actions workflows live in `app/.github/workflows/`.

### `ci.yml` — Lint & Tests

Triggers on every push to `main` or `develop`, and on pull requests targeting `main`.

| Step | What it does |
|---|---|
| Setup | Checks out code, sets up Python 3.11 |
| Install | Installs `requirements.txt` + `requirements-dev.txt` |
| Lint | Runs **Ruff** over `backend/`, `retrain/`, and `tests/` |
| Test | Runs the full pytest suite with verbose output |

### `cd.yml` — Deploy

Triggers on every push to `main` (after CI passes). Runs two independent jobs in parallel:

| Job | Target | Tool |
|---|---|---|
| `deploy-backend` | Railway | Railway CLI (`railway up --detach`) |
| `deploy-frontend` | Vercel | Vercel CLI (`vercel --prod`) |

Required secrets: `RAILWAY_TOKEN`, `VERCEL_TOKEN`, `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID`.

### `retrain.yml` — Retrain Model

Manually triggered via `workflow_dispatch`. Accepts an optional `train_csv` input (path or URL; defaults to `data/train.csv`).

| Step | What it does |
|---|---|
| Setup | Checks out code, sets up Python 3.11, installs `retrain/requirements.txt` |
| Retrain | Runs `retrain/retrain.py` and writes the new artifact to `backend/model.pkl` |
| Commit | Commits and pushes the updated `model.pkl` if it changed (tagged `[skip ci]` to avoid a loop) |

---

## Deployment

The application is deployed and publicly accessible. (<https://amusing-connection-production.up.railway.app>)

---

## Further Documentation

Detailed backend architecture — layer diagrams, service descriptions, API schemas, and the full directory reference — is documented in [app/ARCHITECTURE.md](app/ARCHITECTURE.md).

---

## License

ARIA — Insurance Bundle Predictor was built for the DataQuest competition. All rights reserved.
