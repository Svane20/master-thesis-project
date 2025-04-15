ARG USE_GPU=false
ARG BASE_IMAGE=python:3.10-slim

FROM ${BASE_IMAGE}

ARG USE_GPU

WORKDIR /app

# Install dependencies based on the USE_GPU flag
RUN if [ "$USE_GPU" = "true" ]; then \
      echo "Building GPU version..."; \
      apt-get update && apt-get install -y python3 python3-pip && \
      rm -rf /var/lib/apt/lists/* && \
      pip3 install --upgrade pip && \
      pip3 install --no-cache-dir \
         --index-url https://download.pytorch.org/whl/cu124 \
         --extra-index-url https://pypi.org/simple \
         torch==2.6.0+cu124 \
         torchvision==0.21.0+cu124 \
         torchaudio==2.6.0+cu124 \
         pydantic-settings==2.8.1 \
         pydantic==2.10.6 \
         numpy==1.26.3 \
         pillow==11.1.0 \
         "fastapi[all]==0.115.9" \
         uvicorn==0.34.0 \
         onnxruntime-gpu==1.21.0 \
         prometheus-fastapi-instrumentator==7.0.2 \
         prometheus-client==0.21.1 \
         pymatting==1.1.13 \
         opencv-python-headless==4.10.0.84 \
         python-json-logger==3.3.0; \
    else \
      echo "Building CPU version..."; \
      pip install --upgrade pip && \
      pip install --no-cache-dir \
         torch==2.6.0 \
         torchvision==0.21.0 \
         torchaudio==2.6.0 \
         pydantic-settings==2.8.1 \
         pydantic==2.10.6 \
         numpy==1.26.3 \
         pillow==11.1.0 \
         "fastapi[all]==0.115.9" \
         uvicorn==0.34.0 \
         onnxruntime==1.21.0 \
         prometheus-fastapi-instrumentator==7.0.2 \
         prometheus-client==0.21.1 \
         pymatting==1.1.13 \
         opencv-python-headless==4.10.0.84 \
         python-json-logger==3.3.0; \
    fi

# Copy project related files
COPY resnet/onnx/configs /app/configs
COPY resnet/onnx/models /app/models
COPY resnet/onnx/main.py /app/main.py

# Copy the libs directory
COPY libs /app/libs

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]