# Sky Replacement Pipeline

## Table of Contents
**[Dataset Converter](#dataset-converter)**<br>
**[Demos](#dataset-converter)**<br>
**[Kubernetes](#kubernetes)**<br>
**[Machine Learning](#machine-learning)**<br>
**[Synthetic Data Generation](#synthetic-data-generation)**<br>
**[Web Services](#web-services)**<br>

## Dataset Converter
The [dataset-converter](./dataset-converter) takes the versioned Blender renders (RGBA + α-masks) 
and builds a training-ready corpus. 
It also generates key auxiliary inputs—foreground, background and trimaps—and provides validation 
steps to guarantee integrity.

## Demos
The [demos](./demos) contain the deployed ONNX models in HuggingFace Spaces.
The live demos are available at:

- [U-Net Resnet Sky Replacement](https://huggingface.co/spaces/Svane20/unet-resnet-sky-replacement)
- [U-Net SWIN Sky Replacement](https://huggingface.co/spaces/Svane20/unet-swin-sky-replacement)
- [DPT Sky Replacement](https://huggingface.co/spaces/Svane20/unet-dpt-sky-replacement)

## Kubernetes
The [k8s](./k8s) folder contains the Kubernetes deployment files for the models.

## Machine Learning
The [machine-learning](./machine-learning) folder contains the training, experiment tracking and evaluation for the ML models.

## Synthetic Data Generation
The [synthetic-data-generation](./synthetic-data-generation) folder contains the scripts to generate synthetic data for training the models.

## Web Services
The [web-services](./web-services) folder contains the web services for the models.
This includes the REST API's and the associated benchmarks and performance tests.