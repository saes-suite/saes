.PHONY: help install start-api build lint format test ci docs docs-serve docs-preview start-docs release-dry-run release-stamp version branch

# ── Variables ────────────────────────────────────────────────
BUMP      ?= minor
AUTHOR    ?=
DOCS_PORT ?= 8000

# ── Default ──────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────
install: ## Install saes dependencies with all extras via uv
	uv sync --extra dev

# ── Run ──────────────────────────────────────────────────────
start-api: ## Launch FastAPI REST server
	uv run uvicorn saes.server.app:app --reload --port 8060

build: ## Build wheel and sdist packages
	uv build

# ── Quality ──────────────────────────────────────────────────
lint: ## Run ruff and mypy
	uv run ruff check .
	uv run mypy src

format: ## Auto-format code
	uv run ruff format .

test: ## Run test suite
	uv run pytest

ci: lint test ## Run full CI quality gate

# ── Documentation (GitHub Pages) ─────────────────────────────
docs: ## Build MkDocs Material static site in strict mode (exact GitHub Pages output)
	uv run --extra viz --extra api --extra docs mkdocs build --strict --site-dir site

docs-serve: ## Preview GitHub Pages documentation locally with live reload (:8000)
	@echo "📚 Serving SAES documentation at http://127.0.0.1:$(DOCS_PORT)"
	uv run --extra viz --extra api --extra docs mkdocs serve -a 127.0.0.1:$(DOCS_PORT)

docs-preview: docs ## Build strict static site and serve exact GitHub Pages bundle (:8000)
	@echo "📚 Serving static GitHub Pages build at http://127.0.0.1:$(DOCS_PORT)"
	@python3 -m http.server $(DOCS_PORT) --bind 127.0.0.1 --directory site

start-docs: docs-serve ## Alias for docs-serve (launch local GitHub Pages preview)

# ── Release ──────────────────────────────────────────────────
version: ## Show current version from version.txt
	@cat version.txt

release-dry-run: ## Preview release stamp without modifying files
	@node scripts/release_stamp.js --dry-run --bump $(BUMP) $(if $(AUTHOR),--author $(AUTHOR),)

release-stamp: ## Stamp changelog and bump version (BUMP=major|minor|patch AUTHOR=@user)
	@node scripts/release_stamp.js --bump $(BUMP) $(if $(AUTHOR),--author $(AUTHOR),)

# ── Branching ────────────────────────────────────────────────
branch: ## Create a new branch with the correct version prefix (interactive)
	@echo ""; \
	echo "Select version bump type for this branch:"; \
	echo "  1) minor  (default — new feature, X.Y+1.0)"; \
	echo "  2) patch  (bugfix / hotfix, X.Y.Z+1)"; \
	echo "  3) major  (breaking change, X+1.0.0)"; \
	echo ""; \
	printf "Choice [1]: "; \
	read CHOICE; \
	case "$${CHOICE:-1}" in \
		1) PREFIX="feat" ;; \
		2) PREFIX="fix" ;; \
		3) PREFIX="major" ;; \
		*) echo "Invalid choice"; exit 1 ;; \
	esac; \
	echo ""; \
	printf "Branch name (e.g. add-feature): "; \
	read NAME; \
	if [ -z "$$NAME" ]; then \
		echo "ERROR: branch name cannot be empty"; \
		exit 1; \
	fi; \
	BRANCH="$$PREFIX/$$NAME"; \
	echo ""; \
	echo "Creating branch: $$BRANCH"; \
	git checkout -b "$$BRANCH"; \
	echo ""; \
	echo "✅ Branch '$$BRANCH' created."; \
	echo "📝 Remember to add your changes to CHANGELOG.md under ## [Unreleased]"
