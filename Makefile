.PHONY: lint test lint-fix code-quality run build-image run-image clean

SRC_DIR = .

IMAGE_NAME ?= quay.io/rose/rose-game-ai-reference
DRIVER_PATH ?= mydriver.py
PORT ?= 8081

# By default, run both linting and tests
all: lint test

lint:
	@echo "Running flake8 linting..."
	flake8 --show-source --statistics .
	black --check --diff .

lint-fix:
	@echo "Running lint fixing..."
	@black --verbose --color .

code-quality:
	@echo "Running static code quality checks..."
	radon cc .
	radon mi .

test:
	@echo "Running unittests..."
	pytest

run:
	@echo "Running driver logic server ..."
	PYTHONPATH=$(SRC_DIR):$$PYTHONPATH python rose/main.py --port $(PORT) --driver $(DRIVER_PATH)

build-image:
	@echo "Building container image ..."
	podman build -t $(IMAGE_NAME) .

run-image:
	@echo "Running container image ..."
	podman run --rm -it --network host -e PORT=$(PORT) $(IMAGE_NAME)

clean:
	-rm -rf .coverage
	-rm -rf htmlcov
	-rm -rf .pytest_cache
	-find . -name '*.pyc' -exec rm {} \;
	-find . -name '__pycache__' -exec rmdir {} \;
