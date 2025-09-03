# PCM-LLM Development Makefile
# Provides common development tasks and shortcuts

.PHONY: help install install-dev clean test test-unit test-integration lint format type-check docs build dist clean-build clean-pyc clean-test clean-cache run run-reasoning run-summarization run-classification run-all cache-info clear-cache

# Default target
help:
	@echo "🚀 PCM-LLM Development Commands"
	@echo "================================"
	@echo ""
	@echo "Installation:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  run              Run default benchmark (reasoning)"
	@echo "  run-reasoning    Run reasoning benchmark"
	@echo "  run-summarization Run summarization benchmark"
	@echo "  run-classification Run classification benchmark"
	@echo "  run-all          Run all benchmarks"
	@echo ""
	@echo "Cache Management:"
	@echo "  cache-info       Show cache status"
	@echo "  clear-cache      Clear entire cache"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             Run linting checks"
	@echo "  format           Format code with black and isort"
	@echo "  type-check       Run type checking with mypy"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo ""
	@echo "Documentation:"
	@echo "  docs             Build documentation"
	@echo ""
	@echo "Packaging:"
	@echo "  build            Build package"
	@echo "  dist             Create distribution"
	@echo ""
	@echo "Cleaning:"
	@echo "  clean            Clean all generated files"
	@echo "  clean-build      Clean build artifacts"
	@echo "  clean-pyc        Clean Python cache files"
	@echo "  clean-test       Clean test artifacts"
	@echo "  clean-cache      Clean cache directories"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Development commands
run:
	python main.py

run-reasoning:
	python main.py reasoning

run-summarization:
	python main.py summarization

run-classification:
	python main.py classification

run-all:
	python main.py all

# Cache management
cache-info:
	python main.py cache-info

clear-cache:
	python main.py clear-cache

# Code quality
lint:
	flake8 core/ compressors/ llms/ data_loaders/ evaluation/ utils/
	pylint core/ compressors/ llms/ data_loaders/ evaluation/ utils/

format:
	black core/ compressors/ llms/ data_loaders/ evaluation/ utils/
	isort core/ compressors/ llms/ data_loaders/ evaluation/ utils/

type-check:
	mypy core/ compressors/ llms/ data_loaders/ evaluation/ utils/

# Testing
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# Documentation
docs:
	cd docs && make html

# Packaging
build:
	python -m build

dist: build
	python -m twine check dist/*

# Cleaning
clean: clean-build clean-pyc clean-test clean-cache

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

clean-pyc:
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find . -name '*.so' -delete

clean-test:
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/
	rm -rf .mypy_cache/

clean-cache:
	rm -rf compressed_cache/*
	rm -rf results/*
	rm -rf logs/*
	rm -rf models/*
	touch compressed_cache/.gitkeep results/.gitkeep logs/.gitkeep models/.gitkeep

# Development workflow
dev-setup: install-dev
	pre-commit install

pre-commit: format lint type-check test

# Quick development cycle
dev-cycle: format lint type-check test run

# Environment setup
venv:
	python3 -m venv .venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source .venv/bin/activate  # Linux/macOS"
	@echo "  .venv\\Scripts\\Activate.ps1  # Windows PowerShell"

# Docker support (if needed)
docker-build:
	docker build -t pcm-llm .

docker-run:
	docker run -it --rm pcm-llm

# Performance profiling
profile:
	python -m cProfile -o profile.prof main.py
	python -c "import pstats; pstats.Stats('profile.prof').sort_stats('cumulative').print_stats(20)"

# Memory profiling
memory-profile:
	python -m memory_profiler main.py

# Security checks
security-check:
	bandit -r core/ compressors/ llms/ data_loaders/ evaluation/ utils/
	safety check

# Dependencies
deps-outdated:
	pip list --outdated

deps-update:
	pip-review --auto

# Git helpers
git-hooks:
	pre-commit install

git-status:
	git status --porcelain

git-clean:
	git clean -fd
	git reset --hard HEAD
