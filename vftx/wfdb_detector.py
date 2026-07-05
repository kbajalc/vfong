"""
Concrete QRSDetector implementation using wfdb's XQRS algorithm.

XQRS returns beat positions only — no beat-type classification.
All detected beats are therefore returned as 'N' (normal).
As a consequence UR and VR features will always be 0.0.

Contrast with the reference OSEA detector which classifies each beat as
'N' (normal), 'V' (VPC), or 'Q' (unknown/unclassified).  OSEA-only metrics
(UR, VR) are thus not replicable with this detector until a beat classifier
is added.
"""

import numpy as np
import wfdb.processing

from vftx.types import QRSDetector  # noqa: F401 — used for isinstance checks


class WfdbXqrsDetector:
    """QRSDetector backed by wfdb.processing.xqrs_detect.

    Parameters
    ----------
    verbose:
        Pass through to xqrs_detect.  Default False to suppress console output.
    """

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose
    pass #def

    def detect(
        self,
        signal_mv: np.ndarray,
        sampling_rate: float,
    ) -> list[tuple[int, str]]:
        """Detect QRS complexes and return (sample_index, 'N') tuples.

        Beat type is always 'N' — xqrs provides no beat classification.
        Sample indices are in the signal's native sampling rate.
        """
        qrs_inds = wfdb.processing.xqrs_detect(
            sig=signal_mv,
            fs=sampling_rate,
            verbose=self._verbose,
        )
        return [(int(idx), "N") for idx in qrs_inds]
    pass #def
pass #class
