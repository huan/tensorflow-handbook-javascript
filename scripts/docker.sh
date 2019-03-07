#!/usr/bin/env bash
set -e

IMAGE_NAME=tensorflow/tensorflow
TAG_NAME_CPU=latest-py3
TAG_NAME_GPU=latest-gpu-py3
# for CUDA v10
# TAG_NAME_GPU=2.0.0a0-gpu-py3

DOCKER_CMD_CPU=docker
DOCKER_CMD_GPU=nvidia-docker

if [[ ! "$NV_GPU" ]]; then
  NV_GPU=0
fi

if [[ $DOCKER_CMD ]]; then
  if [[ ! $TAG_NAME ]]; then
    TAG_NAME=$TAG_NAME_CPU
  fi
elif [[ $(which nvidia-docker) ]]; then
  DOCKER_CMD=${DOCKER_CMD_GPU}
  TAG_NAME=${TAG_NAME_GPU}
else
  DOCKER_CMD=${DOCKER_CMD_CPU}
  TAG_NAME=${TAG_NAME_CPU}
fi

if [[ "$PULL" ]]; then
  docker pull ${IMAGE_NAME}:${TAG_NAME}
fi

if [[ ! "$CONTAINER_NAME" ]]; then
  # CONTAINER_NAME="$(id -nu)"-concise-chit-chat
  CONTAINER_NAME="${USER}"-javascript-concise-chit-chat
fi

cat <<_MSG_

Starting Docker Container ...
========================

NV_GPU=$NV_GPU
DOCKER_CMD=$DOCKER_CMD
IMAGE_NAME=$IMAGE_NAME
TAG_NAME=$TAG_NAME
CONTAINER_NAME=$CONTAINER_NAME

------------------------
_MSG_

DOCKER_CONTAINER_ID=$(docker ps -q -a -f name="$CONTAINER_NAME")

if [[ ! "$DOCKER_CONTAINER_ID" ]]; then
  echo "Creating new docker container: ${CONTAINER_NAME} ..."
  # -u $(id -u):$(id -g) \
  $DOCKER_CMD run \
      -t -i \
      --name "$CONTAINER_NAME" \
      --mount type=bind,source="$(pwd)",target=/notebooks \
      -p 6007:6006 \
      -p 8889:8888 \
      "${IMAGE_NAME}:${TAG_NAME}" \
      /bin/bash
else
  echo "Resuming exiting docker container: ${CONTAINER_NAME}, press [Enter] to continue ..."
  $DOCKER_CMD start "${CONTAINER_NAME}"
  $DOCKER_CMD attach "${CONTAINER_NAME}"
fi
