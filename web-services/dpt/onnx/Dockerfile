ARG USE_GPU=false
ARG BASE_IMAGE=python:3.10-slim

FROM ${BASE_IMAGE}

ARG USE_GPU

WORKDIR /app

# Install curl so Docker healthchecks work
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.cpu.txt /app/requirements.cpu.txt
COPY requirements.gpu.txt /app/requirements.gpu.txt

# Install dependencies based on the USE_GPU flag
RUN if [ "$USE_GPU" = "true" ]; then \
      echo "Building GPU version..."; \
      apt-get update && apt-get install -y python3 python3-pip && \
      rm -rf /var/lib/apt/lists/* && \
      pip3 install --upgrade pip && \
      pip3 install --no-cache-dir -r requirements.gpu.txt; \
    else \
      echo "Building CPU version..."; \
      apt-get update && apt-get install -y curl python3 python3-pip && \
      rm -rf /var/lib/apt/lists/* && \
      pip install --upgrade pip && \
      pip install --no-cache-dir -r requirements.cpu.txt; \
    fi

# Install dependencies based on the USE_GPU flag
RUN if [ "$USE_GPU" = "true" ]; then \
      echo "Building GPU version..."; \
      apt-get update && apt-get install -y python3 python3-pip && \
      rm -rf /var/lib/apt/lists/* && \
      pip3 install --upgrade pip && \
      pip3 install --no-cache-dir -r requirements.gpu.txt; \
    else \
      echo "Building CPU version..."; \
      pip install --upgrade pip && \
      pip install --no-cache-dir -r requirements.cpu.txt; \
    fi

# Copy project related files
COPY dpt/onnx/configs /app/configs
COPY dpt/onnx/models /app/models
COPY dpt/onnx/main.py /app/main.py

# Copy the libs directory
COPY libs /app/libs

EXPOSE 8007

CMD ["uvicorn", "main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8007"]