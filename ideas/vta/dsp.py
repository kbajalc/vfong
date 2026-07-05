from collections.abc import Callable

import numpy as np
from scipy import signal
from scipy import ndimage

# from pxg import FS, MS, Record

### CONFIGURATION ###############################

FS = 250     # Sampling Rate (Hz)
MS = 4       # Tick size 4 ms (250 Hz)

LYN_WIND = 40 // MS   # 40 ms window for the Lynn filter, aprox 15 Hz cutoff frequency
LYN_HIGH = 00 // MS   # 0 ms for the high pass filter (not used)
MED_WIND = 596 // MS  # 596 ms window for the median filter

### FILTERS #####################################

def LynnFilter(sig: np.ndarray, W: int) -> np.ndarray:
    W -= W % 2
    # cf = 0.3 * FS / (W/2)
    # print(cf)

    a = np.array([1, -2, 1])
    b = np.zeros(W + 1)
    b[0] = 1
    b[W//2] = -2
    b[W] = 1
    g = (len(b) // 2) ** 2
    shift = W // 2 - 1

    sig = signal.lfilter(b, a, sig) / g # type: ignore
    sig = np.roll(sig, -shift)    
    return sig
pass #def

def BaselFilter(sig: np.ndarray, M: int) -> np.ndarray:
    if M <= 1: return sig
    # calculate moving median
    med = ndimage.median_filter(sig, size=M, mode="nearest")
    # med = np.convolve(med, np.ones(M)/M, mode="same")
    sig = sig - LynnFilter(med, W = M)
    return sig
pass #def

def ButterFilter(sig: np.ndarray, Low: int, High: int, order: int = 4) -> np.ndarray:
    nyq = 0.5 * FS
    low = Low / nyq
    high = High / nyq
    b, a = signal.butter(order, [low, high], btype='band') # type: ignore
    sig = signal.filtfilt(b, a, sig) 
    return sig
pass #def

def SignalFilter(sig: np.ndarray, L = LYN_WIND, M = MED_WIND * 1, H = LYN_HIGH * 0) -> np.ndarray:
    sig = BaselFilter(sig, M)
    sig = LynnFilter(sig, W = L)
    if H > 0:
        hi = np.convolve(sig, np.ones(H)/H, mode="same")
        sig = sig - hi
    pass #if
    return sig
pass #def

def FilterSignal(L = LYN_WIND, M = MED_WIND * 1, H = LYN_HIGH * 0) -> Callable[[np.ndarray], np.ndarray]:
    def filter_func(sig: np.ndarray) -> np.ndarray:
        return SignalFilter(sig, L, M, H)
    pass #def
    return filter_func
pass #def

# 61 may be enough for 2-15 Hz, but 91 is safer, must be odd.
def HilbertKernel(taps=91):
    if taps % 2 == 0: taps += 1
    center = (taps - 1) // 2
    h = np.zeros(taps)
    for i in range(taps):
        k = i - center
        # Only odd positions are non-zero
        if k % 2 == 0: continue 
        h[i] = (2 / (np.pi * k)) * (0.54 - 0.46 * np.cos(2 * np.pi * i / (taps - 1)))
    pass #for       
    return h, int(center)
pass #def

def HilbertFilter(sig: np.ndarray, taps=91):
    kernel, delay = HilbertKernel(taps)
    imag = np.convolve(sig, kernel, mode='same')
    return imag, delay
pass #def



def CosineWindow(Ls = 3):
    t = np.linspace(0, Ls, Ls * FS)
    w = np.zeros_like(t)
    # Region 1: 0 <= t <= 1/4
    m1 = (t >= 0) & (t <= 1 / 4)
    w[m1] = 0.5 * (1 - np.cos(4 * np.pi * t[m1]))
    # Region 2: 1/4 <= t <= Ls - 1/4
    m2 = (t >= 1 / 4) & (t <= Ls - 1 / 4)
    w[m2] = 1.0
    # Region 3: Ls - 1/4 <= t <= Ls
    m3 = (t >= Ls - 1 / 4) & (t <= Ls)
    w[m3] = 0.5 * (1 - np.cos(4 * np.pi * t[m3]))

    g = signal.windows.tukey(Ls * FS, alpha=(0.5 / Ls)) # type: ignore

    # plt.figure(figsize=(8, 4))
    # plt.plot(t, w, color='#1f77b4', linewidth=2)
    # plt.plot(t, g, color='#ff7f0e', linewidth=2)
    # plt.show()

    return w
pass #def



