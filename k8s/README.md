# Install Minikube

1. Run the following command to install Minikube:

```text
choco install minikube
````

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
helm upgrade --install prometheus prometheus-community/prometheus -f monitoring/prometheus-values.yaml
```

4. Deploy Grafana

```bash
helm upgrade --install grafana grafana/grafana -f monitoring/grafana-values.yaml
```

### Setup Grafana Ingress

```bash
kubectl apply -f monitoring/monitoring-ingress.yaml
```

## Minikube Tunnel (Ingress)

1. Enable Ingress and Ingress-DNS:

````bash
minikube addons enable ingress
minikube addons enable ingress-dns
````

2. Add the following line to the `C:\Windows\System32\drivers\etc\hosts` file:

````text
127.0.0.1 grafana
127.0.0.1 prometheus
127.0.0.1 resnet-onnx
127.0.0.1 resnet-torchscript
127.0.0.1 resnet-pytorch
127.0.0.1 swin-onnx
127.0.0.1 swin-torchscript
127.0.0.1 swin-pytorch
127.0.0.1 dpt-onnx
127.0.0.1 dpt-torchscript
127.0.0.1 dpt-pytorch
````

3. Run this command to enable the MiniKube tunnel:

````bash
minikube tunnel
````

4. Visit the following URLs:

## Monitoring

- [Grafana](https://grafana)
- [Prometheus](https://prometheus)

## ResNet
- [ResNet ONNX](http://resnet-onnx/docs)
- [ResNet TorchScript](http://resnet-torchscript/docs)
- [ResNet Pytorch](http://resnet-pytorch/docs)

## Swin
- [Swin ONNX](http://swin-onnx/docs)
- [Swin TorchScript](http://swin-torchscript/docs)
- [Swin Pytorch](http://swin-pytorch/docs)

## DPT
- [DPT ONNX](http://dpt-onnx/docs)
- [DPT TorchScript](http://dpt-torchscript/docs)
- [DPT Pytorch](http://dpt-pytorch/docs)