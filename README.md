# Sky Replacement Pipeline

## Table of Contents
**[Dataset Converter](#dataset-converter)**<br>
**[Demos](#dataset-converter)**<br>
**[Kubernetes](#kubernetes)**<br>
**[Machine Learning](#machine-learning)**<br>
**[Synthetic Data Generation](#synthetic-data-generation)**<br>
**[Web Services](#web-services)**<br>

## Dataset Converter
The [dataset-converter](./dataset-converter) takes the versioned generated synthetic data and builds a training-ready corpus with validation steps to guarantee integrity. <br> 
Additionally, generates key auxiliary inputs:
- **Foreground** (`fg/`) & **Background** (`bg/`) layers (by applying each mask to its source) 
- **Trimaps** (`trimaps/`) via stochastic erosion of the mask

## Demos
The [demos](./demos) contain interactive demos hosted on HuggingFace Spaces for each exported ONNX model.
The live demos are available at:

- [U-Net Resnet Sky Replacement](https://huggingface.co/spaces/Svane20/unet-resnet-sky-replacement)
- [U-Net SWIN Sky Replacement](https://huggingface.co/spaces/Svane20/unet-swin-sky-replacement)
- [DPT Sky Replacement](https://huggingface.co/spaces/Svane20/unet-dpt-sky-replacement)

## Kubernetes
The [k8s](./k8s) contain the Kubernetes deployment files for the models.

## Machine Learning
The [machine-learning](./machine-learning) contains:
- ML training
- Experiment tracking with Weights & Biases
- Model evaluation and validation
- Model export to minimal Pytorch checkpoint, ONNX and TorchScript

## Synthetic Data Generation
The [synthetic-data-generation](./synthetic-data-generation) contains the scripts to generate synthetic data for training the models.

## Web Services
The [web-services](./web-services) contain the web services for the models. <br>
This includes: 
- The REST API's for the models
- Dockerfiles for containerization
- Docker-compose files for local, dev and prod environments
- Benchmarking and performance tests
