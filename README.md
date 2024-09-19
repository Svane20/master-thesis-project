# Running Kubernetes

1. Start Docker Desktop

2. Run this command to start Minikube with the Flannel CNI:

````bash
minikube start --cni=flannel
````

## Port Forwarding

### Minio

1. Run this command to forward the Minio WebUI service to the local machine:

````bash
kubectl port-forward svc/minio-console 9001:9001 --namespace minio
````

2. Run this command to forward the Minio service to the local machine:

````bash
kubectl port-forward svc/minio 9000:9000 --namespace minio
````

### WebAPI

1. Run this command to forward the FastAPI service to the local machine:

````bash
kubectl port-forward svc/blender-web-api-service 8000:80
````

## MiniKube Tunnel (Ingress)

1. Enable Ingress and Ingress-DNS:

````bash
minikube addons enable ingress
minikube addons enable ingress-dns
````

2. Add the following line to the `C:\Windows\System32\drivers\etc\hosts` file:

````text
127.0.0.1 blender.local
127.0.0.1 flamenco-manager.local
127.0.0.1 minio-console.local
127.0.0.1 grafana.local
127.0.0.1 prometheus.local
````

3. Run this command to enable the MiniKube tunnel:

````bash
minikube tunnel
````

4. Visit the following URLs:

- [Blender WebAPI](http://blender.local)
- [Flamenco Manager](https://flamenco-manager.local)
- [Minio Console](https://minio-console.local)
- [Grafana](https://grafana.local)
- [Prometheus](https://prometheus.local)

# Running Docker Compose

1. Run this command

````bash
docker compose up --build -d
````