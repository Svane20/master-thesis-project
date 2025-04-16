#!/usr/bin/env bash
set -euo pipefail

# ——— CONFIG ———
# Export these before running
#   export DOCKER_HUB_USERNAME=svane222
#   export DOCKER_HUB_PASSWORD=
: "${DOCKER_HUB_USERNAME:?Need to set DOCKER_HUB_USERNAME}"
: "${DOCKER_HUB_PASSWORD:?Need to set DOCKER_HUB_PASSWORD}"


services=(
  "resnet/onnx"
  "resnet/torchscript"
  "resnet/pytorch"
  "swin/onnx"
  "swin/torchscript"
  "swin/pytorch"
  "dpt/onnx"
  "dpt/torchscript"
  "dpt/pytorch"
)

# Build modes
declare -A modes=(
  [gpu]="USE_GPU=true BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 suffix=gpu"
  [cpu]="USE_GPU=false BASE_IMAGE=python:3.10-slim suffix=cpu"
)

# Log in to Docker Hub
echo "🔑 Logging in to Docker Hub as '${DOCKER_HUB_USERNAME}'..."
echo "${DOCKER_HUB_PASSWORD}" | docker login -u "${DOCKER_HUB_USERNAME}" --password-stdin

for mode in "${!modes[@]}"; do
  eval "${modes[$mode]}"
  echo
  echo "🛠️  Building & pushing all images for MODE=${mode^^}"
  for svc in "${services[@]}"; do
    img_base="${svc//\//-}"
    local_tag="${img_base}:${suffix}"
    remote_tag="${DOCKER_HUB_USERNAME}/${img_base}:${suffix}"

    echo "  ▶ Building ${local_tag}..."
    docker build \
      --build-arg USE_GPU="${USE_GPU}" \
      --build-arg BASE_IMAGE="${BASE_IMAGE}" \
      -f "${svc}/Dockerfile" \
      -t "${local_tag}" \
      .

    echo "  ▶ Tagging as ${remote_tag}..."
    docker tag "${local_tag}" "${remote_tag}"

    echo "  ▶ Pushing ${remote_tag}..."
    docker push "${remote_tag}"
  done
done

echo
echo "✅ All images built, tagged, and pushed to Docker Hub."