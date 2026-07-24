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
ARGS ?=

all: run

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

run: install
	uv run python3 -m $(MAIN) $(ARGS)

debug: install
	uv run python3 -m pdb -m $(MAIN) $(ARGS)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache dist uv.lock .cache/
	@if [ -L $(VENV) ]; then rm -f $(VENV); else rm -rf $(VENV); fi

lint: install
	$(FLAKE) .
	$(MYPY) $(MYPY_FLAGS) .

lint-strict: install
	$(FLAKE) .
	$(MYPY) --strict .

.PHONY: all install run debug clean lint lint-strict setup-venv
