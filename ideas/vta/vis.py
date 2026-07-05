import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from pxg import FS, MS, MV
from pxg import Record
from pxg import plot

from .fft import Spectral, FFT_WIN

from IPython.display import Markdown, display

## PLOT #########################################

def PlotRecord(rec: Record, page: int | list[int] = -1, off = 0, marker = False, title = ""):
    offset = off*plot.FS
    display(Markdown(f"## {rec.DB.upper()} {rec.RID}\n---"))
    if title:
        display(Markdown(f"```text\n{title}\n```"))
    pass #if
    if isinstance(page, list):
        for p in page:
            PlotPage(rec, p, offset, marker = marker)
        pass #for
    elif page != -1:
        PlotPage(rec, page, offset, marker = marker)
    else:
        for page in range(0, (len(rec.Signal) + plot.CHUNK // 2) // plot.CHUNK, 1):
            PlotPage(rec, page, offset, marker = marker)
        pass #for
    pass #if
    return rec
pass #def


def PlotPage(
    rec: Record, 
    page = 0, 
    offset = 0,
    marker = False,
    include = []
):
    res = plot.Page(
        rec, 
        page, 
        offset,

        SENSOR = False,
        SIGNAL = True,
        DECTOR = False,
        EXTEND = "HILBER",

        LOW = -700,
        ZEROS = [0, -400],

        label = "BPM",
        angle = 0,

        rrqs  = True, 
        qsvl  = False, 
        
        punts = True,  
        trig  = False,
        onoff = False,
        letra = False,

        grid  = False,
        anref = True,
        simple = True,

        # simple = False,
        # marker = marker and page not in include,

        show = False,
    )

    if res:
        PON, POF = plot.Range(rec, page, offset)

        # plot.Signal(rec.Local[PON:POF] / MV, zero=-400, color="tab:blue", format="-", linewidth=0.6)
        # plot.Signal(rec.Cover[PON:POF] / MV, zero=-400, color="tab:orange", format="-", linewidth=0.5)

        # plot.Signal(rec.Maxim[PON:POF] / MV, zero=-400, color="tab:red", format="-", linewidth=0.8)
        # plot.Signal(rec.Minim[PON:POF] / MV, zero=-400, color="tab:red", format="-", linewidth=0.8)

        plot.Signal(rec.Digit[PON:POF] * 90, zero=-700, color="tab:gray", format="-", linewidth=0.5, fill=True, alpha=0.2)
        plot.Signal(rec.Class[PON:POF] * 90, zero=-700, color="tab:red", format="-", linewidth=0.5, fill=True, alpha=0.2)
    pass #if

    plot.Show()

    # PlotSTFT(rec, page, offset)
    
    PlotFFT(rec, page, offset)

    return res
pass #def


def PlotFFT(rec: Record, page: int, offset: int):
    PON, POF = plot.Range(rec, page, offset)
    data = rec.Basic[PON:POF]

    # PlotSTFT(rec.Local[PON:POF+250*2])

    MAXFQ = 20

    steps = 8
    plt.figure(figsize=(30, 3))
    # plt.title(f"FFT {rec.DB.upper()}:{rec.RID}")

    for sub in range(steps):
        z = sub * 925
        seg = data[z:z+FFT_WIN]
        if len(seg) < FFT_WIN:
            continue
        pass #if

        s = Spectral(seg, True)

        title = (
            f"En:{s.Entropy:.2f} Fl:{s.Flatness:.2f} Sp:{s.Spread:.2f} Gi:{s.Gini:.2f}"
        )

        pwr = s.Normal
        
        plt.subplot(1, steps, sub + 1)
        plt.title(title, fontsize=13)
        plt.xlim(0, MAXFQ)
        plt.ylim(0, np.max(pwr)*1.1)
        plt.yticks([])
        plt.xticks(np.arange(0, MAXFQ, 1), labels=[str(i) if i % 5 == 0 else '' for i in range(0, MAXFQ, 1)])
    
        # Centroid
        plt.axvline(s.Centroid, color='orange', linestyle='--', label='Centroid')
        base = s.Centroid
        fixed = s.Spread
        plt.axvspan(base - fixed, base + fixed, color='orange', alpha=0.1, label='Dominant Range')

        # Dominant Frequency
        plt.axvline(s.Dominant, color='blue', linestyle='-', label='Dominant', linewidth=0.8)
        # Edge Frequency
        plt.axvline(s.EdgeFreq, color='tab:purple', linestyle='--', label='Edge', linewidth=0.3)

        plt.plot(s.Freqs, pwr, color='black', linewidth=1)

        pf = s.Freqs[s.Peaks]
        pv = pwr[s.Peaks]
        plt.plot(pf, pv, "x", color='red', label='Peaks')
    pass #for

    plt.tight_layout()
    plt.show()
pass #def


def PlotSTFT(rec: Record, page: int, offset: int):
    PON, POF = plot.Range(rec, page, offset)
    data = rec.Local[PON:POF]

    # 1. Setup Parameters
    noverlap = FFT_WIN // 2       # 50% overlap is standard for STFT

    # 2. Compute STFT
    # 'hamming' window is specified here to minimize spectral leakage
    # f, t_segments, Zxx = signal.stft(sig, fs=fs, window='hamming', 
    #                                 nperseg=nperseg, noverlap=noverlap)
    
    f, t_segments, Zxx = signal.spectrogram(data, 
                               fs=FS, 
                               window='hamming', 
                               nperseg=FFT_WIN, 
                               noverlap=noverlap, 
                               scaling='density')
    
    freq_mask = (f >= 0) & (f <= 20)
    f_trimmed = f[freq_mask]
    Zxx_trimmed = Zxx[freq_mask, :]

    # power_zxx = np.abs(Zxx_trimmed)**2
    # power_db = 10 * np.log10(power_zxx + 1e-10)

    # 3. Visualization
    plt.figure(figsize=(30, 4))
    plt.pcolormesh(t_segments, f_trimmed, Zxx_trimmed, shading='gouraud')
    plt.title('STFT Magnitude (Hamming Window)')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    plt.ylim(0, 20) # Focusing on the relevant frequency range
    # plt.colorbar(label='Magnitude')
    plt.yticks([])
    plt.show()
pass #def