.PHONY: clean venv dev install lint test help

PYTHON := python3
VENV_DIR := .venv
VENV_BIN := $(VENV_DIR)/bin
VENV_PIP := $(VENV_BIN)/pip
VENV_PYTHON := $(VENV_BIN)/python

# Default target
.DEFAULT_GOAL := help

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV_DIR)
	@$(VENV_PIP) install --upgrade pip setuptools wheel
	@echo "Virtual environment created at $(VENV_DIR)"

# Install dependencies from pyproject.toml
install: venv
	@echo "Installing dependencies from pyproject.toml..."
	@$(VENV_PIP) install pdm
	@$(VENV_BIN)/pdm install
	@echo "Dependencies installed"

# Install development dependencies
dev: venv
	@echo "Installing development dependencies..."
	@$(VENV_PIP) install -e ".[dev]"
	@echo "Development dependencies installed"

# Run linting
lint: venv
	@echo "Running linters..."
	@$(VENV_BIN)/black --check .
	@$(VENV_BIN)/flake8 .
	@$(VENV_BIN)/isort --check-only --profile black .

# Run tests
test: venv
	@echo "Running tests..."
	@$(VENV_BIN)/pytest

# Clean environment
clean:
	@echo "Cleaning project..."
	@rm -rf $(VENV_DIR) build dist *.egg-info .pytest_cache .coverage .mypy_cache __pycache__ **/__pycache__
	@echo "Cleanup complete"

# Display help information
help:
	@echo "Available targets:"
	@echo "  venv     - Create virtual environment"
	@echo "  install  - Install dependencies from pyproject.toml"
	@echo "  dev      - Install development dependencies"
	@echo "  lint     - Run linters (black, flake8, isort)"
	@echo "  test     - Run tests with pytest"
	@echo "  clean    - Remove virtual environment and build artifacts"
	@echo "  help     - Display this help message"
