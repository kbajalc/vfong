from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension(
        name='qrs_detect',
        sources=[
            'qrs_detect.pyx',
            'osea20-gcc/analbeat.c',
            'osea20-gcc/bdac.c',
            'osea20-gcc/bxbep.c',
            'osea20-gcc/classify.c',
            'osea20-gcc/match.c',
            'osea20-gcc/noisechk.c',
            'osea20-gcc/postclas.c',
            'osea20-gcc/qrsdet.c',
            'osea20-gcc/qrsfilt.c',
            'osea20-gcc/rythmchk.c',
        ],
        libraries=['wfdb', 'm'],
        extra_compile_args=['-Wimplicit-function-declaration', '-O3'],
    ),
    Extension(
        name='signal_processing',
        sources=['signal_processing.pyx'],
    ),
    Extension(
        name='vf_data',
        sources=['vf_data.pyx'],
    ),
    Extension(
        name='vf_features',
        sources=['vf_features.pyx', 'vf_features_native.c'],
        libraries=['m'],
        extra_compile_args=['-Wimplicit-function-declaration', '-O3'],
    ),
    Extension(
        name='wfdb_reader',
        sources=['wfdb_reader.pyx'],
        libraries=['wfdb'],
        extra_compile_args=['-Wimplicit-function-declaration', '-O3'],
    ),
]


setup(
    name='vfong',
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()]
)
