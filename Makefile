# ============================================================
# Environment detection (42 cluster vs local machine)
# ============================================================
USER_NAME = $(shell whoami)
HAS_SGOINFRE = $(shell if [ -d /sgoinfre/goinfre ]; then echo "yes"; else echo "no"; fi)

ifeq ($(HAS_SGOINFRE),yes)
	VENV_TARGET = /sgoinfre/goinfre/Perso/$(USER_NAME)/envs/RAG_env
	export HF_HOME = /sgoinfre/goinfre/Perso/$(USER_NAME)/hf-cache
	export UV_LINK_MODE = copy
else
	VENV_TARGET = .venv_local
endif


# ============================================================
# Project / tooling variables
# ============================================================
MAIN				= src
PYTHON				= python3
VENV				= .venv
VENV_BIN			= $(VENV)/bin
V_PYTHON			= $(VENV_BIN)/python
MYPY_FLAGS			= --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
FLAKE				= $(VENV_BIN)/flake8
MYPY				= $(VENV_BIN)/mypy
ARGS ?=


# ============================================================
# Pipeline parameters (overridable, e.g. make index CHUNK_SIZE=1000)
# ============================================================
CHUNK_SIZE	 		?= 2000
K					?= 10
DATASET_DIR	 		= data/datasets/UnansweredQuestions
OUTPUT_DIR	 		= data/output/search_results/UnansweredQuestions

QUESTIONS_DATASET 	= $(DATASET_DIR)/dataset_docs_public.json
CODE_DATASET	 	= $(DATASET_DIR)/dataset_code_public.json


# ============================================================
# Default target
# ============================================================
all: run


# ============================================================
# Environment setup
# ============================================================
setup-venv:
	@if [ "$(HAS_SGOINFRE)" = "yes" ]; then \
		if [ ! -L $(VENV) ] && [ ! -d $(VENV) ]; then \
			echo "[INFO] Cluster 42 détecté : Création du lien symbolique vers le sgoinfre..."; \
			mkdir -p $(VENV_TARGET); \
			ln -s $(VENV_TARGET) $(VENV); \
			echo "HF_HOME=/sgoinfre/goinfre/Perso/$(USER_NAME)/hf-cache" > .env; \
			echo "UV_LINK_MODE=copy" >> .env; \
		fi \
	else \
		if [ ! -d $(VENV) ]; then \
			echo "[INFO] Machine maison détectée : Initialisation d'un .venv standard..."; \
			uv venv $(VENV); \
		fi \
	fi

install: setup-venv
	uv sync

add: setup-venv
	uv add $(ARGS)

# ============================================================
# Generic run / debug entrypoints
# ============================================================
run: install
	uv run python3 -m $(MAIN) $(ARGS)

debug: install
	uv run python3 -m pdb -m $(MAIN) $(ARGS)

# ============================================================
# Cleanup
# ============================================================
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache dist uv.lock .cache/
	@if [ -L $(VENV) ]; then rm -f $(VENV); else rm -rf $(VENV); fi

# ============================================================
# Linting
# ============================================================
lint: install
	$(FLAKE) .
	$(MYPY) $(MYPY_FLAGS) .

lint-strict: install
	$(FLAKE) .
	$(MYPY) --strict .

# ============================================================
# Pipeline steps (each runnable independently)
# ============================================================
index: install
	uv run python3 -m $(MAIN) index --max_chunk_size $(CHUNK_SIZE)

search-questions: install
	uv run python3 -m $(MAIN) search_dataset \
		--dataset_path $(QUESTIONS_DATASET) \
		--k $(K) \
		--save_directory $(OUTPUT_DIR)

search-code: install
	uv run python3 -m $(MAIN) search_dataset \
		--dataset_path $(CODE_DATASET) \
		--k $(K) \
		--save_directory $(OUTPUT_DIR)

search-all: search-questions search-code

# ============================================================
# Full pipeline (index + all searches)
# ============================================================
pipeline: index search-all

# ============================================================
# Tests
# ============================================================
test: install
	uv run python3 -m pytest tests/

# ============================================================
# Phony targets
# ============================================================
.PHONY: all install run debug clean lint lint-strict setup-venv test \
		index search-questions search-code search-all pipeline add
