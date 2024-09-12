# Running Kubernetes

1. Start Docker Desktop

2. Run this command to start Minikube with the Flannel CNI:

````bash
minikube start --cni=flannel
````

## Port Forwarding

1. Run this command to forward the Minio WebUI service to the local machine:

````bash
kubectl port-forward svc/minio-console 9001:9001 --namespace minio
````

2. Run this command to forward the Minio service to the local machine:

````bash
kubectl port-forward svc/minio 9000:9000 --namespace minio
````



# Running Docker Compose

1. Run this command

````bash
docker compose up --build -d
````