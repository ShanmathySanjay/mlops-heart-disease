# Heart Disease Prediction API (MLOps Project)

## Overview
This project implements an end-to-end MLOps pipeline for predicting heart disease using machine learning.  
It includes data processing, model training, CI/CD automation, containerization, Kubernetes deployment, and monitoring.

## Tech Stack
- Python, FastAPI, Uvicorn
- Scikit-learn (ML models)
- MLflow (experiment tracking)
- Docker, DockerHub
- Kubernetes (K3s on AWS EC2)
- GitHub Actions (CI/CD)
- Prometheus & Grafana (Monitoring)

## Dataset
Dataset Source:
UCI Heart Disease Dataset
https://archive.ics.uci.edu/dataset/45/heart+disease

Subset used:
processed.cleveland.data

### Setup
Place dataset in:

data/heart.csv

## EDA & Preprocessing
- Missing values handled
- Feature scaling using StandardScaler
- Exploratory analysis:
  - Correlation heatmap
  - Class balance
  - Feature distributions

## Model Development
Two models trained:
- Logistic Regression
- Random Forest

### Evaluation Metrics:
- Accuracy
- Precision
- Recall
- ROC-AUC

### Model Selection:
Best model selected automatically based on ROC-AUC score and saved as:

model.pkl

## Experiment Tracking
MLflow used to track:
- Parameters
- Metrics
- Artifacts (plots, models)

## CI/CD Pipeline

Pipeline steps:
1. Lint code (flake8)
2. Run unit tests (pytest)
3. Train models
4. Select best model
5. Save model.pkl
6. Build Docker image
7. Push to DockerHub (commit SHA tagged)
8. Deploy to Kubernetes (EC2)

## Docker

### Build

docker build -t heart-api .

### Run

docker run -p 8000:8000 heart-api


## Kubernetes Deployment

Apply:

kubectl apply -f k8s-deployment.yaml

kubectl apply -f ingress.yaml

kubectl apply -f prometheus.yaml

kubectl apply -f grafana.yaml


## API Endpoints

| Endpoint   | Description        |
|------------|--------------------|
| `/docs`    | Swagger UI         |
| `/predict` | Prediction API     |
| `/metrics` | Metrics |


## Monitoring
- Prometheus scrapes `/metrics`
- Grafana dashboards visualize performance


## Testing

Run:

pytest tests/


## Deployment URLs

- API: http://<EC2-IP>/docs  
- Prometheus: http://<EC2-IP>:30009  
- Grafana: http://<EC2-IP>:30010  