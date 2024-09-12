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

1. Add the following line to the `C:\Windows\System32\drivers\etc\hosts` file:

````text
127.0.0.1 blender.local
127.0.0.1 minio.local
127.0.0.1 grafana.local
127.0.0.1 prometheus.local
````

2. Run this command to enable the MiniKube tunnel:

````bash
minikube tunnel
````



# Running Docker Compose

1. Run this command

````bash
docker compose up --build -d
````