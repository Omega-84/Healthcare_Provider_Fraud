# Healthcare Provider Fraud Detection

Machine learning system to detect potential healthcare provider fraud using Medicare claims data.

![CI/CD](https://github.com/Omega-84/Healthcare_Provider_Fraud/actions/workflows/ci.yml/badge.svg)

## 🎯 Overview

This project uses XGBoost to classify healthcare providers as potentially fraudulent or legitimate based on aggregated claims, beneficiary, and billing data. It includes:

- **ML Pipeline**: End-to-end training with hyperparameter tuning (Optuna)
- **Experiment Tracking**: MLflow integration for metrics and model versioning
- **API Serving**: FastAPI backend with Gradio web interface
- **Containerized**: Docker with optimized multi-stage builds
- **Cloud Deployment**: AWS ECS Fargate with Application Load Balancer
- **CI/CD**: GitHub Actions for automated testing and deployment

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
├── tests/                  # Unit tests
├── .github/workflows/      # CI/CD pipeline
├── .aws/                   # AWS task definitions
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
- Engineer 44 predictive features
- Tune hyperparameters (30 Optuna trials)
- Train XGBoost model
- Log to MLflow
- Save model to `artifacts/`

### 4. Export Model for Docker

```bash
python scripts/export_model.py
```

### 5. Run API Locally

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
| `/docs` | GET | Swagger API documentation |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "count_unique_beneficiary": 100,
    "count_unique_claims": 500,
    "count_dead_beneficiary": 5,
    "mean_hospital_stay_days": 5.0,
    ...
  }'
```

## 🐳 Docker

### Build & Run Locally

```bash
# Build
docker build -t healthcare-fraud .

# Run
docker run -p 8000:8000 healthcare-fraud
```

### Pull from Docker Hub

```bash
docker pull <your-dockerhub-username>/healthcare-fraud:latest
docker run -p 8000:8000 <your-dockerhub-username>/healthcare-fraud:latest
```

---

## ☁️ AWS Deployment Guide

Complete step-by-step guide to deploy this application to AWS ECS Fargate.

### Prerequisites

- AWS Account
- AWS CLI installed and configured
- Docker installed

### Step 1: Configure AWS CLI

```bash
aws configure
# Enter:
#   AWS Access Key ID: <your-access-key>
#   AWS Secret Access Key: <your-secret-key>
#   Default region: us-east-1
#   Default output format: json
```

### Step 2: Set Up Billing Alerts

1. Go to [AWS Billing Console](https://console.aws.amazon.com/billing/)
2. **Billing Preferences** → Enable:
   - ✅ Receive Free Tier Usage Alerts
   - ✅ Receive Billing Alerts
3. **Budgets** → Create budget:
   - Type: Cost budget
   - Amount: $20/month
   - Alert threshold: 80%

### Step 3: Create IAM User for GitHub Actions

1. Go to [IAM Console](https://console.aws.amazon.com/iam/) → **Users** → **Create user**
2. Name: `github-actions`
3. Attach policies:
   - `AmazonEC2ContainerRegistryFullAccess`
   - `AmazonECS_FullAccess`
4. Create access key (Application outside AWS)
5. Save credentials securely

### Step 4: Create ECR Repository

```bash
aws ecr create-repository --repository-name healthcare-fraud --region us-east-1
```

### Step 5: Create ECS Service-Linked Role

```bash
aws iam create-service-linked-role --aws-service-name ecs.amazonaws.com
```

### Step 6: Push Docker Image to ECR

```bash
# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build and tag
docker build -t healthcare-fraud .
docker tag healthcare-fraud:latest $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/healthcare-fraud:latest

# Push
docker push $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/healthcare-fraud:latest
```

### Step 7: Create Security Group

1. Go to [EC2 → Security Groups](https://console.aws.amazon.com/ec2/v2/home#SecurityGroups)
2. **Create security group**:
   - Name: `healthcare-fraud-sg`
   - VPC: Default VPC
   - Inbound rules:
     | Type | Port | Source |
     |------|------|--------|
     | HTTP | 80 | 0.0.0.0/0 |
     | Custom TCP | 8000 | 0.0.0.0/0 |

### Step 8: Create ECS Cluster

1. Go to [ECS Console](https://console.aws.amazon.com/ecs/) → **Create cluster**
2. Configure:
   - Name: `healthcare-cluster`
   - Infrastructure: **AWS Fargate**
3. Create

### Step 9: Create Task Definition

1. Go to **Task definitions** → **Create new task definition**
2. Configure:
   - Family: `healthcare-fraud-task`
   - Launch type: **Fargate**
   - OS: Linux/X86_64
   - CPU: `0.5 vCPU`
   - Memory: `1 GB`
3. Container:
   - Name: `healthcare-fraud`
   - Image: `<account-id>.dkr.ecr.us-east-1.amazonaws.com/healthcare-fraud:latest`
   - Port: `8000`
4. Create

### Step 10: Create ECS Service with Load Balancer

1. Go to cluster → **Services** → **Create**
2. Configure:
   - Launch type: **Fargate**
   - Task definition: `healthcare-fraud-task`
   - Service name: `healthcare-fraud-service`
   - Desired tasks: `1`
3. Networking:
   - VPC: Default VPC
   - Subnets: Select 2+ subnets
   - Security group: `healthcare-fraud-sg`
4. Load balancer:
   - Type: **Application Load Balancer**
   - Create new ALB: `healthcare-ALB`
   - Listener port: 80
   - Target group: `healthcare-fraud-target` (port 8000)
5. Create

### Step 11: Configure ALB Security Group

Ensure ALB security group allows inbound port 80:

```bash
# Get ALB security group ID
SG_ID=$(aws elbv2 describe-load-balancers --names healthcare-ALB --query "LoadBalancers[0].SecurityGroups[0]" --output text)

# Add port 80 rule
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 80 --cidr 0.0.0.0/0
```

### Step 12: Access Your Application

Get the ALB DNS name:

```bash
aws elbv2 describe-load-balancers --names healthcare-ALB --query "LoadBalancers[0].DNSName" --output text
```

Access:
- **Health check**: `http://<alb-dns>/`
- **Gradio UI**: `http://<alb-dns>/ui`
- **API Docs**: `http://<alb-dns>/docs`

### Step 13: Set Up GitHub Actions Secrets

Go to GitHub repo → **Settings** → **Secrets** → **Actions** → Add:

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `DOCKERHUB_USERNAME` | Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |

---

## 🛑 Teardown (Stop Charges)

### Stop Service (Saves Fargate costs ~$15/month)

```bash
aws ecs update-service --cluster healthcare-cluster --service <service-name> --desired-count 0
```

### Full Teardown (Saves all costs ~$30/month)

```bash
# Delete ECS service
aws ecs delete-service --cluster healthcare-cluster --service <service-name> --force

# Delete ECS cluster
aws ecs delete-cluster --cluster healthcare-cluster

# Delete Load Balancer (main cost!)
aws elbv2 delete-load-balancer --load-balancer-arn $(aws elbv2 describe-load-balancers --names healthcare-ALB --query "LoadBalancers[0].LoadBalancerArn" --output text)

# Wait for ALB deletion, then delete target group
sleep 30
aws elbv2 delete-target-group --target-group-arn $(aws elbv2 describe-target-groups --names healthcare-fraud-target --query "TargetGroups[0].TargetGroupArn" --output text)

# Delete ECR repository (optional, minimal cost)
aws ecr delete-repository --repository-name healthcare-fraud --force
```

### Estimated Monthly Costs

| Resource | Running | Stopped |
|----------|---------|---------|
| Fargate (0.5 vCPU, 1GB) | ~$15 | $0 |
| ALB | ~$16 | ~$16 |
| ECR | ~$0.20 | ~$0.20 |
| **Total** | **~$31** | **~$16** |

---

## 🔄 CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`):

1. **Build** Docker image
2. **Push** to Docker Hub
3. **Push** to AWS ECR
4. **Deploy** to ECS Fargate

Triggers on every push to `main` branch.

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src
```

## 📈 MLflow Tracking

```bash
mlflow ui
```

View experiments at http://localhost:5000

## 🛠️ Technologies

- **ML**: XGBoost, Optuna, Scikit-learn, Pandas
- **API**: FastAPI, Gradio, Pydantic
- **MLOps**: MLflow, Docker, GitHub Actions
- **Cloud**: AWS ECS Fargate, ECR, ALB

## 📝 License

MIT License
