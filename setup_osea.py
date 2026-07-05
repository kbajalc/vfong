"""Build the OSEA qrs_detect extension WITHOUT libwfdb (Phase 4 QRS validation).

    conda activate dev
    python setup_osea.py build_ext --inplace

qrs_detect.pyx needs only three OSEA symbols (BeatDetectAndClassify, ResetBDAC,
amap). The compiled OSEA sources include <wfdb/ecgcodes.h> for annotation-code
constants only; tests/_osea/wfdb/ecgcodes.h stubs those. bxbep.c (which does real
WFDB I/O) is replaced by tests/_osea/osea_amap.c, which provides amap + fflag.
No libwfdb link required.
"""
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

OSEA = "osea"
osea_sources = [
    f"{OSEA}/analbeat.c", f"{OSEA}/bdac.c", f"{OSEA}/classify.c",
    f"{OSEA}/match.c", f"{OSEA}/noisechk.c", f"{OSEA}/postclas.c",
    f"{OSEA}/qrsdet.c", f"{OSEA}/qrsfilt.c", f"{OSEA}/rythmchk.c",
    "tests/_osea/osea_amap.c",   # replaces bxbep.c (no libwfdb I/O)
]

ext = Extension(
    name="qrs_detect",
    sources=["hong/qrs_detect.pyx"] + osea_sources,
    libraries=["m"],                       # NOTE: no "wfdb"
    include_dirs=[np.get_include(), "tests/_osea", OSEA],
    extra_compile_args=["-O3"],
)

setup(
    name="vfong_osea",
    ext_modules=cythonize([ext], language_level="3"),
)
