# master-thesis-project

## Setup Run Scripts for FastAPI

1. Go to `Run` -> `Edit Configurations` -> `+` -> `FastAPI`

2. Add the full filepath to main.py in the field `Application file`

## Run FastAPI

### DEV

````text
fastapi dev main.py
````

### PROD

````text
fastapi run main.py
````

### API Documentation

Open the browser and navigate to [localhost:8000/docs](http://localhost:8000/docs) to access the API documentation.

## Running Kubernetes services locally

````bash
kubectl port-forward svc/minio -n minio 9000:9000
kubectl port-forward svc/flamenco-manager -n flamenco 8080:8080
````