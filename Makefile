SAC_TIMESTEPS ?= 1000
SAC_DEVICE ?= cpu
SAC_BUFFER_SIZE ?= 50000

.PHONY: setup smoke demo train test preview

setup:
	./scripts/setup_host.sh

smoke:
	./scripts/isaac_docker_run.sh 10

demo:
	./scripts/run_latest_lunar_demo.sh

train:
	SAC_TIMESTEPS=$(SAC_TIMESTEPS) SAC_DEVICE=$(SAC_DEVICE) SAC_BUFFER_SIZE=$(SAC_BUFFER_SIZE) ./scripts/train_sac_docker.sh

test:
	python3 -m unittest discover -s tests

preview:
	python3 app/preview.py --seed 42 --size 1400
