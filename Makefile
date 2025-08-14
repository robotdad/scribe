.PHONY: help install doctor test lint format clean ai-context-files activate

# Default goal is to show help
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo ""
	@echo "Scribe MCP Server"
	@echo "=================="
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Usage: make install  # Set up project dependencies"
	@echo ""

install: ## Install/update all project dependencies
	@echo ""
	@echo "Installing project dependencies..."
	uv sync --extra dev
	@echo "Dependencies installed"
	@echo ""
	@echo "Available commands:"
	@ls .venv/bin/ 2>/dev/null | grep -E "(scribe)" | sed 's/^/  /' || echo "  scribe (available after activation)"
	@echo ""
	@$(MAKE) -s activate

doctor: ## Check project health
	@echo ""
	@echo "Checking project health..."
	@uv --version || (echo "❌ uv not found" && exit 1)
	@test -f uv.lock && echo "✅ Lock file exists" || echo "❌ Lock file missing"
	@test -d .venv && echo "✅ Virtual environment exists" || echo "❌ Virtual environment missing"
	@test -f pyproject.toml && echo "✅ Project config exists" || echo "❌ Project config missing"
	@source .venv/bin/activate && python -c "import scribe; print('✅ Scribe package imports successfully')" || echo "❌ Scribe package import failed"
	@echo "Health check completed"
	@echo ""

test: ## Run tests
	@echo "Running tests..."
	uv run pytest tests/ -v

lint: ## Lint code
	@echo "Linting code..."
	uv run ruff check src/
	uv run mypy src/

format: ## Format code
	@echo "Formatting code..."
	uv run black src/
	uv run ruff format src/

clean: ## Clean generated files and caches
	@echo "Cleaning generated files..."
	rm -rf ai_context/generated/*
	rm -rf ai_context/git_collector/*
	rm -rf build/
	rm -rf dist/
	rm -rf src/*.egg-info/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Generated files cleaned"

ai-context-files: ## Build AI context files for development
	@echo ""
	@echo "Building AI context files..."
	uv run python tools/build_ai_context_files.py
	uv run python tools/build_git_collector_files.py
	@echo "AI context files generated"
	@echo ""

activate: ## Show command to activate virtual environment
	@if [ -n "$$VIRTUAL_ENV" ]; then \
		echo "\033[32m✓ Virtual environment already active\033[0m"; \
		echo ""; \
	elif [ -f .venv/bin/activate ]; then \
		echo "\033[33m→ Run this command: source .venv/bin/activate\033[0m"; \
		echo ""; \
	else \
		echo "\033[31m✗ No virtual environment found. Run 'make install' first.\033[0m"; \
	fi