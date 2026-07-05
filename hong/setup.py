import os
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

# Build modes (set via environment variables before running setup.py):
#
#   make build          normal optimised release build
#   make debug          DEBUG=1  →  -O0 -g, gdb_debug=True  (use with cygdb/gdb)
#   make trace          CYTHON_TRACE=1  →  Python-level line tracing  (use with pdb/IDE)
#
# The two modes are mutually exclusive; TRACE takes precedence if both are set.

_debug = os.environ.get("DEBUG") == "1"
_trace = os.environ.get("CYTHON_TRACE") == "1"

# C compiler flags
_warn = ['-Wimplicit-function-declaration']
if _debug or _trace:
    _cflags = _warn + ['-O0', '-g']
else:
    _cflags = _warn + ['-O3']

# Cython compiler directives
_directives = {}
if _trace:
    # linetrace makes every Cython line report to sys.settrace() so pdb /
    # IDE debuggers can step through .pyx source files.
    # profile=True is required for linetrace to work.
    _directives['linetrace'] = True
    _directives['profile'] = True

# C preprocessor macros consumed by the Cython-generated C code
_macros = [('CYTHON_TRACE', '1'), ('CYTHON_TRACE_NOGIL', '1')] if _trace else []

if _debug:
    print("setup.py: DEBUG build  (-O0 -g, gdb_debug=True)")
elif _trace:
    print("setup.py: TRACE build  (-O0 -g, linetrace=True)")
else:
    print("setup.py: RELEASE build  (-O3)")

extensions = [
    Extension(
        name='qrs_detect',
        sources=[
            'qrs_detect.pyx',
            'osea/analbeat.c',
            'osea/bdac.c',
            'osea/bxbep.c',
            'osea/classify.c',
            'osea/match.c',
            'osea/noisechk.c',
            'osea/postclas.c',
            'osea/qrsdet.c',
            'osea/qrsfilt.c',
            'osea/rythmchk.c',
        ],
        libraries=['wfdb', 'm'],
        extra_compile_args=_cflags,
        define_macros=_macros,
    ),
    Extension(
        name='signal_processing',
        sources=['signal_processing.pyx'],
        extra_compile_args=_cflags,
        define_macros=_macros,
    ),
    Extension(
        name='vf_data',
        sources=['vf_data.pyx'],
        extra_compile_args=_cflags,
        define_macros=_macros,
    ),
    Extension(
        name='vf_features',
        sources=['vf_features.pyx', 'vf_features_native.c'],
        libraries=['m'],
        extra_compile_args=_cflags,
        define_macros=_macros,
    ),
    Extension(
        name='wfdb_reader',
        sources=['wfdb_reader.pyx'],
        libraries=['wfdb'],
        extra_compile_args=_cflags,
        define_macros=_macros,
    ),
]


setup(
    name='vfong',
    ext_modules=cythonize(
        extensions,
        gdb_debug=_debug,
        compiler_directives=_directives,
    ),
    include_dirs=[np.get_include()]
)
