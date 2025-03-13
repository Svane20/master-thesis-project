# Running Kubernetes

1. Start Docker Desktop

2. Run this command to start Minikube with the Flannel CNI:

````bash
minikube start --cni=flannel
````

# Prometheus & Grafana

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

## Minikube Tunnel (Ingress)

1. Enable Ingress and Ingress-DNS:

````bash
minikube addons enable ingress
minikube addons enable ingress-dns
````

2. Add the following line to the `C:\Windows\System32\drivers\etc\hosts` file:

````text
127.0.0.1 grafana.local
127.0.0.1 prometheus.local
````

3. Run this command to enable the MiniKube tunnel:

````bash
minikube tunnel
````

4. Visit the following URLs:

- [Grafana](https://grafana.local)
- [Prometheus](https://prometheus.local)