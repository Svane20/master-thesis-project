# Deployment on Kubernetes

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
kubectl apply -f minio-secrets.yaml
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