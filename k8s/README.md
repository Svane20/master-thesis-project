# Deployment on Kubernetes

## Flamenco

### Create namespace

````bash
kubectl create namespace flamenco
````

### Deploy NFS Server

````bash
kubectl apply -f flamenco-storage-pvc.yaml
````

### Deploy Flamenco Manager

````bash
kubectl apply -f flamenco-manager.yaml
````

### Deploy Flamenco Worker

````bash
kubectl apply -f flamenco-worker.yaml
````

### Access Flamenco Manager

````bash
kubectl port-forward svc/flamenco-manager 8080:8080 --namespace flamenco
````

### Add Blender file to NFS storage

1. Add directory in NFS storage

````bash
kubectl exec -it flamenco-manager-0 -n flamenco -- mkdir -p /var/flamenco/output/jobs
````

2. Copy Blender file to NFS storage

````bash
kubectl cp test.blend flamenco-manager-0:/var/flamenco/output/jobs/test.blend -n flamenco
````

## Minio

### Create namespace

```bash
kubectl create namespace minio
```

### Install Minio

```bash
helm repo add minio https://charts.min.io/
helm repo update
```

### Deploy secrets

```bash
kubectl apply -f minio-secret.yaml
```

### Deploy Minio

```bash
helm install minio minio/minio --namespace minio -f minio-values.yaml
```

### Upgrade Minio

```bash
helm upgrade minio minio/minio --namespace minio -f minio-values.yaml
```

### Access Minio

```bash
kubectl port-forward svc/minio-console 9001:9001 --namespace minio
```

### Setup Minio Ingress

```bash
kubectl apply -f minio-ingress.yaml
```

### Uninstall Minio

```bash
helm uninstall minio --namespace minio
```

# Monitoring

## Prometheus & Grafana

1. Enable metrics-server in Minikube

```bash
minikube addons enable metrics-server
```

2. Install Prometheus & Grafana

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update
```

3. Deploy Prometheus

```bash
helm install prometheus prometheus-community/prometheus --namespace monitoring --create-namespace
```

4. Deploy Grafana

```bash
helm install grafana grafana/grafana --namespace monitoring --create-namespace
```

### Setup Grafana Ingress

```bash
kubectl apply -f monitoring-ingress.yaml
```

# WebAPI

## Deploy

```bash
kubectl apply -f web-service.yaml
```

## Upgrade

```bash
kubectl rollout restart deployment blender-web-api
```