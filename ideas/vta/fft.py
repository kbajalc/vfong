import numpy as np
from scipy import signal, stats

from pxg import plot
from pxg import FS, MS, Record

### FOURIER #####################################

# FS = 250     # Sampling Rate (Hz)
# MS = 4       # Tick size 4 ms (250 Hz)

FFT_WIN = 1024             # Buffer Size (4.096 seconds)
FFT_BIN = FS / FFT_WIN     # Frequency resolution (Hz per bin)

FFT_HANN = np.hanning(FFT_WIN)
FFT_FREQS = np.fft.rfftfreq(FFT_WIN, 1/FS)

# FFT_LIMIT = int(24.5 / FFT_BIN)
# FFT_LOW = int(2 / FFT_BIN)
# FFT_HIGH = int(15 / FFT_BIN)
# FFT_WINDOW = signal.windows.tukey(FN, alpha=0.5)

# WELCH_NPERSEG = 512     # Sub-window size for Welch (controls freq resolution vs. smoothing)
# WELCH_NOVERLAP = 384    # 50% overlap is standard
# WELCH_WINDOW = 'hann'   # Window function

# PEAK_HEIGHT = 0.01
PEAK_DIST = 3
# HARM_TOLER = 0.5 # Hz tolerance for matching harmonics
FREQ_SPLIT = 5.0 # Hz split for band power ratio (e.g., LF/HF)
EDGE_PCTIL = 0.95 # Percentile for spectral edge frequency


def FFT(seg: np.ndarray) -> np.ndarray:
    # Apply window to reduce leakage
    tran = np.fft.rfft(seg * FFT_HANN)
    # Magnitude (Absolute value)
    # Normalization (Divide by sum of window)
    # This corrects for the energy loss of the windowing
    tran = np.abs(tran) / np.sum(FFT_HANN)
    # Symmetry Compensation (Multiply by 2 for positive freqs)
    # We only keep the first half (0 to N/2)
    mags = tran * 2
    # Fix DC component (Index 0 should not be doubled)
    mags[0] = mags[0] / 2
    # Power is magnitude squared
    # power = mags ** 2
    return mags
pass #def


class Spectral:
    def __init__(self, data: np.ndarray, sig = False, short = False):
        magna = FFT(data) if sig else data
        power = magna ** 2
        total = np.sum(power)
        freqs = FFT_FREQS

        self.Freqs = freqs
        self.Magna = magna
        self.Power = power
        self.Total = total

        ###################################################################

        # Dominant frequency in the 1.66-8.33 Hz range (VFL band)
        mask = (freqs > 1.66) & (freqs <= 8.33)
        self.Prominent: float = freqs[mask][np.argmax(power[mask])]

        # VFL Frequency Bands (0.5-3.5 Hz, 3.5-8 Hz, 8-20 Hz)
        self.LowBand: float = np.sum(power[(freqs > 0.5) & (freqs <= 3.5)])
        self.MidBand: float = np.sum(power[(freqs > 3.5) & (freqs <= 8.0)])
        self.HighBand: float = np.sum(power[(freqs > 8.0) & (freqs <= 20.0)])
        self.OutBand: float = np.sum(power[(freqs > 1.5) & (freqs <= 24)])

        ####################################################################

        ### Dominant Frequency is the frequency with max power
        mask = (freqs > 0.5) & (freqs < 100)
        self.Dominant: float = freqs[mask][np.argmax(power[mask])]

        # Spectral Centroid (cent) is the "center of mass" of the spectrum
        if total == 0: total = 1e-10 # Avoid division by zero
        cenf = np.sum(freqs * power) / total
        self.Centroid: float = cenf # type: ignore

        # Spectral Spread (spread) is the standard deviation of the spectrum around the centroid
        spread = np.sqrt(np.sum(((freqs - cenf) ** 2) * power) / total)
        self.Spread: float = spread # type: ignore

        ### Normalize power for Entropy/Probabilistic calcs
        normal = power / total if total > 0 else power
        self.Normal: np.ndarray = normal 

        ### Spectral Entropy (H) measures the "disorder" of the spectrum
        # Filter out zeros to avoid log(0)
        p = normal[normal > 0]
        entropy = -np.sum(p * np.log2(p))
        # Normalize by log2 of number of bins (0 to 1 scale)
        entropy = entropy / np.log2(len(normal))
        self.Entropy: float = entropy # type: ignore

        # Gini index measures inequality in the spectrum (0 to 1 scale)
        n = len(power)
        sorted = np.sort(power)
        multipliers = 2 * np.arange(n) - n + 1
        gini_sum = np.dot(multipliers, sorted)
        gini = float(gini_sum / (n * total))
        self.Gini: float = gini # type: ignore

        # Spectral Flatness (SF) measures how "noise-like" the spectrum is
        # Geometric Mean / Arithmetic Mean
        g_mean = stats.gmean(power + 1e-10) # Add small epsilon to avoid log(0)
        a_mean = np.mean(power) + 1e-10
        self.Flatness: float = g_mean / a_mean # type: ignore

        ### Edge Frequency (f_edge) is the frequency below which a certain 
        # percentage of total power is contained
        cumsum = np.cumsum(normal)
        # Find index where cumsum crosses 0.95
        idx = np.searchsorted(cumsum, EDGE_PCTIL)
        # Handle edge case where index might be out of bounds
        idx = min(idx, len(freqs) - 1)
        self.EdgeFreq = freqs[idx]

        ### Band Power Ratio (BPR) is the ratio of power in a high-frequency band to a low-frequency band
        mask_low = freqs <= FREQ_SPLIT
        mask_high = freqs > FREQ_SPLIT        
        p_low = np.sum(power[mask_low])
        p_high = np.sum(power[mask_high])
        if p_low == 0: 
            self.Ratio = 10.0 # High value indicates HF dominance
        else:
            self.Ratio = p_high / p_low
        pass #if

        ### Peak Detection for Harmonics
        # dist is in 'bins'. Adjust based on your resolution.
        # peak_idxs, _ = signal.find_peaks(self.Normal, height=PEAK_HEIGHT, distance=PEAK_DIST)
        # self.Peaks = peak_idxs
        
        # Prominence must be at least 5% of the highest peak
        # This catches the small "parent" in a seesaw, but ignores tiny static
        min_prom = 0.05 * np.max(magna)
        
        # Height must be above the average noise floor
        min_height = np.mean(magna) * 0.5

        # The Core Detection
        peak_idxs, _ = signal.find_peaks(
            magna,
            prominence=min_prom,
            distance=PEAK_DIST,   # ~1 Hz separation minimum
            # width=1.5,          # Reject 1-bin digital spikes
            height=min_height     # Reject absolute background static
        )
        self.Peaks = peak_idxs
    pass #def
pass #class
