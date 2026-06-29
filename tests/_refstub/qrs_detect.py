"""Stub for the OSEA ``qrs_detect`` Cython module.

Placed on ``sys.path`` *ahead* of ``qrs_detect.pyx`` so the reference
``vf_features`` extension imports without building ``qrs_detect`` (which links
libwfdb). Returns no beats, so the reference's QRS features (RR/RR_Std/RR_CV/
UR/VR, idx 22-26) come out 0.0 — those are validated separately against xqrs.
"""


def qrs_detect(samples, sampling_rate):
    return []
