# Retail Agentic AI

A modular, agentic retail decision-support system powered by machine learning. Each
module provides a distinct intelligence layer — pricing, customer health, demand,
basket affinity, and inventory — that feeds actionable recommendations to retail
operations teams through a web dashboard.

---

## Architecture

A FastAPI backend serves six ML-backed modules over a REST API; a React/TypeScript
frontend consumes them as an interactive dashboard.

```
backend/     FastAPI app — routes/, utils/, config.py, app.py
frontend/    React 19 + TypeScript + Vite SPA (Zustand, Tailwind, Recharts)
models/      Pre-trained model artifacts (.pkl) loaded by the backend at startup
sample-data/ Example CSVs for trying each module
```

## Modules

### 1. 🏷️ Pricing Intelligence
- **Model**: XGBoost Classifier (200–800 trees)
- **Inputs**: Product rating, rating count, current vs competitor price ratio
- **Actions**: `increase` / `discount` / `hold` / `decrease`
- **Logic**: Confidence-weighted price adjustment with ±20% guardrails

### 2. 👥 Customer Health Intelligence *(Unified)*
- **Models**: KMeans Clustering (k=4) + XGBoost Churn Classifier (400 trees)
- **Inputs**: Recency (1–365 days), Frequency (1–500 orders), Monetary (₹1–100 K)
- **Segmentation output**: Core Actives / Regular Contributors / Lapsing High-Potential / Dormant Low-Yield
- **Churn output**: Probability score + risk tier (Safe / Low / Medium / High)
- **Recommendation**: Segment × risk-tier matrix → specific retention action
- **Analytics**: Segment distribution · Churn-by-segment · Feature importance · Risk heat matrix

### 3. 📈 Demand Forecasting
- **Model**: Facebook Prophet (time-series)
- **Output**: Point forecasts + uncertainty intervals, spike detection
- **Use case**: Inventory planning, 4–24 week horizon

### 4. 🧺 Market Basket Analysis
- **Method**: FP-Growth association-rule mining, run live on the uploaded transaction CSV
- **Output**: Cross-category product association rules

### 5. 📦 Inventory & Purchase-Order Recommendations
- Combines outputs from churn, demand, basket, and pricing into Purchase Order
  recommendations with ML-style risk scoring, priority tiers, and summary KPIs

### 6. 🔎 Model Compatibility Check
- Inspects an uploaded CSV's columns and reports which of the modules above it can run

---

## Technology Stack

```
Backend:  Python 3.11 · FastAPI · Uvicorn · Scikit-learn · XGBoost · Prophet · mlxtend
Frontend: React 19 · TypeScript · Vite · Zustand · Tailwind CSS · Recharts
```

## Model Files

| File | Type | Size |
|------|------|------|
| `models/pricing_model.pkl` | XGBoost Classifier | ~904 KB |
| `models/pricing_scaler.pkl` | RobustScaler | ~1 KB |
| `models/kmeans.pkl` | KMeans (k=4) | ~24 KB |
| `models/rfm_scaler.pkl` | RobustScaler | ~1 KB |
| `models/churn_model.pkl` | XGBoost Classifier | ~354 KB |
| `models/forecast_prophet.pkl` | Prophet | ~22 KB |

---

## Running locally

**Backend** (FastAPI, port 8000):
```bash
cd backend
cp .env.example .env      # adjust as needed
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend** (Vite dev server, port 5173):
```bash
cd frontend
cp .env.example .env      # set VITE_API_BASE_URL to the backend URL
npm install
npm run dev
```

The backend exposes interactive API docs at `/docs` and a health check at `/health`.

> Note: `VITE_API_BASE_URL` is baked into the frontend at **build time** (Vite convention) —
> set it before running `npm run build` for a production build, not afterwards.

---

## Deployment

Deployed as two independently hosted pieces:
- **Backend**: containerized FastAPI service (EC2 or ECS) — stateful, holds the loaded models in memory.
- **Frontend**: static build (`npm run build`) served from S3 + CloudFront.

Set `CORS_ORIGINS` on the backend to the deployed frontend's origin, and `VITE_API_BASE_URL`
on the frontend build to the deployed backend's URL/domain.
