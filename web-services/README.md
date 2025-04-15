## Docker Compose

1. Go to the root of the project
2. Run the following command

````bash
docker compose up --build -d
````

### Local development

1. Run the following command:

````text
docker compose -f .\docker-compose.dev.yml up --build -d
````

2. Stop the containers

````text
docker compose -f .\docker-compose.dev.yml down
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

#### Test scaling

````text
docker compose -f .\docker-compose.dev.yml up --build --scale unet-onnx=2 -d
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

## Add tracking to Grafana

1. Go to Grafana

````text
http://localhost:3000
````

2. Login

````text
username: admin
password: admin
````

3. Click on Connections -> Data sources
4. Click on Add data source
5. Select Prometheus
6. Fill in the URL of the Prometheus server

````text
http://prometheus:9090
````

7. Click on Save & Test