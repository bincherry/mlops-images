#!/bin/bash

RAY_VERSION=2.36.1
VLLM_VERSION=v0.6.1.post2

# CPU
DOCKERFILE=./Dockerfile.cpu
TAG=${RAY_VERSION}-py310-cpu-vllm
BASE_IMAGE=rayproject/ray:${RAY_VERSION}-py310-cpu

# GPU
# DOCKERFILE=./Dockerfile.cuda
# TAG=${RAY_VERSION}-py310-cu123-vllm
# BASE_IMAGE=rayproject/ray:${RAY_VERSION}-py310-cu123

REGISTRY=${REGISTRY:-images.neolink-ai.com/matrixdc}

DOCKER_BUILDKIT=1

EXTRA_ARGS="$@"

BOLD_YELLOW='\033[1;33m'
RESET="\033[0m"

echo -e "${BOLD_YELLOW}Building image ${REGISTRY}/ray:${TAG} ...${RESET}"

docker buildx build \
    --platform=linux/amd64 \
    --build-arg BASE_IMAGE=${BASE_IMAGE} \
    --build-arg VLLM_VERSION=${VLLM_VERSION} \
    $EXTRA_ARGS \
    -t ${REGISTRY}/ray:${TAG} \
    -f ${DOCKERFILE} \
    .
