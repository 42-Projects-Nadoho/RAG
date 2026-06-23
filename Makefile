USER_NAME = $(shell whoami)
HAS_SGOINFRE = $(shell if [ -d /sgoinfre/goinfre ]; then echo "yes"; else echo "no"; fi)

ifeq ($(HAS_SGOINFRE),yes)
    VENV_TARGET = /sgoinfre/goinfre/Perso/$(USER_NAME)/envs/RAG_env
    export HF_HOME = /sgoinfre/goinfre/Perso/$(USER_NAME)/hf-cache
    export UV_LINK_MODE = copy
else
    VENV_TARGET = .venv_local
endif

MAIN			= src
PYTHON			= python3
VENV			= .venv
VENV_BIN		= $(VENV)/bin
V_PYTHON		= $(VENV_BIN)/python
MYPY_FLAGS		= --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
FLAKE			= $(VENV_BIN)/flake8
MYPY			= $(VENV_BIN)/mypy
EXCLUDE			= $(VENV),data/raw/,vllm-0.10.1/
ARGS ?=

all: run


setup-venv:
	@if [ "$(HAS_SGOINFRE)" = "yes" ]; then \
		if [ ! -L $(VENV) ] && [ ! -d $(VENV) ]; then \
			echo "[INFO] Cluster 42 détecté : Création du lien symbolique vers le sgoinfre..."; \
			mkdir -p $(VENV_TARGET); \
			ln -s $(VENV_TARGET) $(VENV); \
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

run: install
	uv run -m $(MAIN) $(ARGS)

debug: install
	uv run -m $(MAIN) $(ARGS)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache dist uv.lock .cache/
	@if [ -L $(VENV) ]; then rm -f $(VENV); else rm -rf $(VENV); fi

lint: install
	$(FLAKE) . --exclude $(EXCLUDE)
	$(MYPY) --exclude $(EXCLUDE) $(MYPY_FLAGS) src

lint-strict: install
	$(FLAKE) . --exclude $(EXCLUDE)
	$(MYPY) --exclude $(EXCLUDE) $(MYPY_FLAGS) --strict src

.PHONY: all install run debug clean lint lint-strict setup-venv
