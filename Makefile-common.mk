MAKEFLAGS += --warn-undefined-variables
SHELL = /bin/bash -o pipefail
.DEFAULT_GOAL := help
.PHONY: help install clean clean-env

## display help message
help:
	@awk '/^##.*$$/,/^[~\/\.0-9a-zA-Z_-]+:/' $(MAKEFILE_LIST) | awk '!(NR%2){print $$0p}{p=$$0}' | awk 'BEGIN {FS = ":.*?##"}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' | sort

venv ?= .venv
pip := $(venv)/bin/pip

$(pip): .python-version
# create venv using system python even when another venv is active
	PATH=$${PATH#$${VIRTUAL_ENV}/bin:} python3 -m venv --clear $(venv)
	$(venv)/bin/python --version

$(venv): requirements.txt $(pip)
	$(pip) install -r requirements.txt
	$(pip) install -e .
	$(pip) install -U lbforaging rware
	touch $(venv)

## delete the venv
clean-env:
	rm -rf $(venv)

## clean pycache and log files
clean:
	@echo "Cleaning up log files, __pycache__ folders, and .pyc files..."
	find . -name '*.log' -type f -delete
	find . -name '__pycache__' -type d -exec rm -r {} +
	find . -name '*.pyc' -type f -delete
	@echo "Cleanup complete."

## create venv and install this package
install: $(venv)
