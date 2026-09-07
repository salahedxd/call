SRC_DIR = src
PYTHON = python3
DEBUG = --debug

install:
	uv sync

run:
	uv run python -m $(SRC_DIR)

run-debug:
	uv run python -m $(SRC_DIR) $(DEBUG)

debug:
	$(PYTHON) -m pdb -m $(SRC_DIR)

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "*_cache" -exec rm -rf {} +

lint:
	flake8 src
	mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs