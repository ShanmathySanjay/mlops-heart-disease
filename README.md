# Heart Disease Prediction API (MLOps Project)

An end-to-end MLOps pipeline for predicting heart disease, covering model training, CI/CD, deployment, and monitoring.


## Links
- Report: [Assignment Report](Assignment_Report.pdf)  
- Demo: https://drive.google.com/file/d/1wqkRPKkklxRqb8Sl096AhdB8Pzri8gwO/view?usp=drive_link
- Repo: https://github.com/ShanmathySanjay/mlops-heart-disease


## Tech Stack
Python, FastAPI, Scikit-learn, MLflow, Docker, DockerHub, Kubernetes (K3s), GitHub Actions, Prometheus, Grafana, AWS EC2


## Dataset
- UCI Heart Disease Dataset  
- Place file at:
```
data/heart.csv
````

## Setup

``` 
git clone https://github.com/ShanmathySanjay/mlops-heart-disease.git
cd mlops-heart-disease
pip install -r requirements.txt
```

## Train model

```
PYTHONPATH=. python src/models/train.py
```

### Run API

```bash
uvicorn src.api.app:app --reload
```

Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)


## MLflow

```bash
mlflow ui
```

[http://127.0.0.1:5000](http://127.0.0.1:5000)


## Docker

```bash
docker build -t heart-api .
docker run -p 8000:8000 heart-api
```

## Kubernetes

```bash
kubectl apply -f k8s-deployment.yaml
kubectl apply -f ingress.yaml
kubectl apply -f prometheus.yaml
kubectl apply -f grafana.yaml
```

## API

**POST /predict**

```json
{
  "features": [60, 1, 3, 140, 260, 1, 2, 120, 1, 2.5, 1, 2, 3]
}
```


## Monitoring

* Prometheus → `/metrics`
* Grafana → dashboards
* Query:

```
rate(request_count_total[1m])
```

## Tests

```bash
pytest tests/
```

## Deployment

* API: http://<EC2-IP>/docs
* Prometheus: http://<EC2-IP>:30009
* Grafana: http://<EC2-IP>:30010