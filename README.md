# Healthcare Provider Fraud Detection

Machine learning system to detect potential healthcare provider fraud using Medicare claims data.

## 🎯 Overview

This project uses XGBoost to classify healthcare providers as potentially fraudulent or legitimate based on aggregated claims, beneficiary, and billing data. It includes:

- **ML Pipeline**: End-to-end training with hyperparameter tuning (Optuna)
- **Experiment Tracking**: MLflow integration for metrics and model versioning
- **API Serving**: FastAPI backend with Gradio web interface
- **Docker Ready**: Containerized deployment for AWS Fargate

## 📁 Project Structure

```
Healthcare_Project/
├── src/
│   ├── app/                 # FastAPI + Gradio application
│   │   ├── app.py          # Development version
│   │   └── main.py         # Production version
│   ├── data/               # Data loading & preprocessing
│   ├── features/           # Feature engineering
│   ├── models/             # Training, tuning, evaluation
│   ├── serving/            # Inference module
│   │   ├── inference.py    # Model loading & prediction
│   │   └── model/          # Exported model (for Docker)
│   └── utils/              # Data validation
├── scripts/
│   ├── run_pipeline.py     # Full ML pipeline
│   └── export_model.py     # Export model for Docker
├── data/                   # Raw data (gitignored)
├── artifacts/              # Local model artifacts (gitignored)
├── mlruns/                 # MLflow tracking (gitignored)
└── notebooks/              # Jupyter notebooks
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
python -m venv health
source health/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Download Data

Place the following files in `data/`:
- `Train_Inpatientdata-1542865627584.csv`
- `Train_Outpatientdata-1542865627584.csv`
- `Train_Beneficiarydata-1542865627584.csv`
- `Train-1542865627584.csv`

### 3. Train Model

```bash
python scripts/run_pipeline.py
```

This will:
- Load and validate data
- Engineer features
- Tune hyperparameters (30 Optuna trials)
- Train XGBoost model
- Log to MLflow
- Save model to `artifacts/`

### 4. Export Model for Docker

```bash
python scripts/export_model.py
```

### 5. Run API

```bash
# Development
uvicorn src.app.main:app --reload

# Production
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

Access:
- **API Docs**: http://localhost:8000/docs
- **Gradio UI**: http://localhost:8000/ui

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.955 |
| Recall | 0.880 |
| F1 Score | 0.610 |

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | POST | Fraud prediction (JSON) |
| `/ui` | GET | Gradio web interface |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "count_unique_beneficiary": 100,
    "count_unique_claims": 500,
    ...
  }'
```

## 🐳 Docker Deployment

```bash
# Build
docker build -t healthcare-fraud .

# Run
docker run -p 8000:8000 healthcare-fraud
```

## 📈 MLflow Tracking

```bash
mlflow ui
```

View experiments at http://localhost:5000

## 📝 License

MIT License
