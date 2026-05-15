.PHONY: build force clean check fextract mamba

build:
	python setup.py build_ext --inplace

force:
	python setup.py build_ext --inplace -f

clean:
	@echo "TODO: clean build artifacts"

check:
	python demo_emd.py
	python qrs_test.py -r mitdb/111

fextract:
	python feature_extraction.py -o features/features_s8.dat -s 8

mamba:
	mamba install Cython numpy scipy matplotlib joblib Pyro4 scikit-learn setuptools wheel pip


# VENV_DIR ?= .venv
# PYTHON ?= python3
# VENV_PYTHON := $(VENV_DIR)/bin/python
# PIP := $(VENV_PYTHON) -m pip
#
# setup:
# 	$(PYTHON) -m venv $(VENV_DIR)
# 	$(PIP) install --upgrade pip setuptools wheel
# 	$(PIP) install Cython numpy scipy matplotlib joblib Pyro4 scikit-learn
# 	$(PIP) install -e .
# 	$(VENV_PYTHON) setup.py build_ext --inplace

