.PHONY: build force debug trace release clean check fextract mamba

build:
	python setup.py build_ext --inplace

force:
	python setup.py build_ext --inplace -f

# ---------------------------------------------------------------------------
# Debug builds
# ---------------------------------------------------------------------------

# debug: compile with -O0 -g and Cython gdb_debug=True.
# Generates .pyx.gdb XML files alongside the .so files so gdb can map
# C stack frames back to Cython source lines.
# Use with:  cygdb . -- --args python extract_one.py -r mitdb/100 -n 0
# or:        gdb -ex run --args python extract_one.py -r mitdb/100 -n 0
debug:
	DEBUG=1 python setup.py build_ext --inplace -f

# trace: compile with -O0 -g and Cython linetrace=True + CYTHON_TRACE=1.
# Makes every Cython line visible to Python's sys.settrace(), so pdb and
# IDE debuggers (VS Code / PyCharm) can step through .pyx files line by line.
# Use with:  python -m pdb extract_one.py -r mitdb/100 -n 0
# or set a breakpoint in VS Code / PyCharm and run extract_one.py normally.
trace:
	CYTHON_TRACE=1 python setup.py build_ext --inplace -f

# release: restore the normal optimised build after a debug/trace session.
release:
	python setup.py build_ext --inplace -f

# ---------------------------------------------------------------------------

clean:
	find . -maxdepth 2 \( -name '*.so' -o -name '*.pyd' -o -name '*.c' \
	    -o -name '*.pyx.gdb' \) -not -path './osea/*' -delete
	rm -rf build/

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

