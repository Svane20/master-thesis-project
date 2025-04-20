## Docker Compose

### Local development

1. Run the following command:

````text
docker compose -f .\docker-compose.local.yml up --build -d
````

2. Stop the containers

````text
docker compose -f .\docker-compose.local.yml down
````

### Dev development

1. Run the following command:

````text
docker compose -f .\docker-compose.dev.yml up --build -d
````

2. Stop the containers

````text
docker compose -f .\docker-compose.dev.yml down
````

### Production

1. Run the following command:

````text
docker compose -f .\docker-compose.prod.yml up --build -d
````

2. Stop the containers

````text
docker compose -f .\docker-compose.prod.yml down
````

### Demo

1. Run the following command:

````text
docker compose -f .\docker-compose.demo.yml up --build -d
````

2. Stop the containers

````text
docker compose -f .\docker-compose.demo.yml down
````

#### Performance testing

1. Run the following command to start the performance testing environment:

````text
docker compose -f .\docker-compose.performance.yml up -d --scale swin-onnx=3
````

2. Run the following command to run the performance tests:

````text
locust -f tests/performance/locustfile.py --headless -u 50 -r 5 --run-time 2m --host=http://localhost --csv=./reports/locust_report
````

3. Run the following command to stop the performance testing environment:

````text
docker compose -f .\docker-compose.performance.yml down
````

## Swagger UI

### SWIN

[Swin ONNX](http://localhost:80/swin/onnx/docs)
[Swin TorchScript](http://localhost:80/swin/torchscript/docs)
[Swin Pytorch](http://localhost:80/swin/pytorch/docs)

### Resnet

[Resnet ONNX](http://localhost:80/resnet/onnx/docs)
[Resnet TorchScript](http://localhost:80/resnet/torchscript/docs)
[Resnet Pytorch](http://localhost:80/resnet/pytorch/docs)

### DPT

[DPT ONNX](http://localhost:80/dpt/onnx/docs)
[DPT TorchScript](http://localhost:80/dpt/torchscript/docs)
[DPT Pytorch](http://localhost:80/dpt/pytorch/docs)

## Grafana

1. Go to Grafana

````text
http://localhost:3000
````

2. Login

````text
username: admin
password: admin
````

## Build and push docker images

1. Open Git Bash

2. Go to the root of the project

````text
cd Desktop/Thesis/master-thesis-project/web-services
````

3. Export the following variables

````text
export DOCKER_HUB_USERNAME=
export DOCKER_HUB_PASSWORD=
````

4. Make the script executable:

````text
chmod +x build_and_push.sh
````

5. Run the script:

````text
./build_and_push.sh
````