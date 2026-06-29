"""Minimal build of the two libwfdb-free reference extensions for Phase 4
ground-truth: signal_processing + vf_features.  Build in place with:

    python setup_ref.py build_ext --inplace

Does NOT build wfdb_reader / qrs_detect / vf_data (those need libwfdb).
"""
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="signal_processing",
        sources=["signal_processing.pyx"],
        extra_compile_args=["-O3"],
    ),
    Extension(
        name="vf_features",
        sources=["vf_features.pyx", "vf_features_native.c"],
        libraries=["m"],
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="vfong_ref",
    ext_modules=cythonize(extensions, language_level="3"),
    include_dirs=[np.get_include()],
)
