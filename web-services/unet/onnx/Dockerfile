# -----------------------------------------------------------------------------
# 1) BUILDER STAGE
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python commands consistent
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --target="/install" -r requirements.txt

# -----------------------------------------------------------------------------
# 2) FINAL STAGE (RUNTIME)
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install Python, pip, and libgl1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Make python commands consistent
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Ensure binaries and Python modules are found
ENV PATH="/usr/local/bin:${PATH}"
ENV PYTHONPATH="/usr/local/lib/python3.10/site-packages:${PYTHONPATH}"

COPY src/ ./src/

ENV MODEL_PATH="/app/src/models/unet_v1.onnx"

EXPOSE 8001

# Override the NVIDIA entrypoint if needed
ENTRYPOINT []

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
