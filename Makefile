ISAACLAB_ALGO ?= sac
ISAACLAB_NUM_ENVS ?= 512
ISAACLAB_MAX_ITERATIONS ?= 200
ISAACLAB_DEVICE ?= cuda:0

.PHONY: setup gpu train smoke test

setup:
	./scripts/setup_host.sh

gpu:
	./scripts/test_docker_gpu.sh

train:
	ISAACLAB_ALGO=$(ISAACLAB_ALGO) ISAACLAB_NUM_ENVS=$(ISAACLAB_NUM_ENVS) ISAACLAB_MAX_ITERATIONS=$(ISAACLAB_MAX_ITERATIONS) ISAACLAB_DEVICE=$(ISAACLAB_DEVICE) ./scripts/train_isaaclab_docker.sh

smoke:
	ISAACLAB_MAX_ITERATIONS=1 ISAACLAB_NUM_ENVS=16 ./scripts/train_isaaclab_docker.sh

test:
	python3 -m unittest discover -s tests
