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