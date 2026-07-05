import time
from typing import Callable

import numpy as np
from tqdm import tqdm

from pxg import FS, MS, MV
from pxg import EXG, Record, Rids
from pxg import plot

from .vis import PlotRecord
from .dsp import SignalFilter

ALIGN = False
REPORTW = True

FILTER: Callable[[np.ndarray], np.ndarray] = SignalFilter

def Use(filt: Callable[[np.ndarray], np.ndarray]):
    global FILTER
    FILTER = filt
pass #def

### DETECTOR ####################################

DASH =   "--------------------------------------------------------------------------------------"
HEADER_IDN = "RID\t     TP\t     FP\t     FN\t     TN\t    ACC\t    SPC\t    FNR\t    SEN\t    PPV\t     F1\n" + DASH
HEADER_TAB = "RID\tTP\tFP\tFN\tTN\tACC\tSPC\tFNR\tSEN\tPPV\tF1"

def Line(rid: str, TP: int, FP: int, FN: int, TN: int) -> str:
    ACC = (TP + TN) * 100 / (TP + FP + FN + TN) if TP + FP + FN + TN > 0 else 0
    SPC = TN * 100 / (TN + FP) if TN + FP > 0 else 0
    FNR = FN * 100 / (TP + FN) if TP + FN > 0 else 0
    SEN = TP * 100 / (TP + FN) if TP + FN > 0 else 0
    PPV = TP * 100 / (TP + FP) if TP + FP > 0 else 0
    F1V = SEN * PPV * 2 / (SEN + PPV) if SEN + PPV > 0 else 0

    if TP + FP + FN == 0: F1V = 100

    if ALIGN:
        line = (f"{rid}\t{TP:7}\t{FP:7}\t{FN:7}\t{TN:7}\t{ACC:7.2f}\t{SPC:7.2f}\t{FNR:7.2f}\t{SEN:7.2f}\t{PPV:7.2f}\t{F1V:7.2f}")
    else:
        line = (f"{rid}\t{TP}\t{FP}\t{FN}\t{TN}\t{ACC:.2f}\t{SPC:.2f}\t{FNR:.2f}\t{SEN:.2f}\t{PPV:.2f}\t{F1V:.2f}")
    return line
pass #def

def CFM(rid: str, res: np.ndarray) -> tuple[tuple[int, int, int, int], str]:
    TP = np.sum(res == 3)
    FP = np.sum(res == 1)
    FN = np.sum(res == 2)
    TN = np.sum(res == 0)
    line = Line(rid, TP, FP, FN, TN)
    return (TP, FP, FN, TN), line
pass #def

class Detector:
    def __init__(self, SEG: int, WIN: int):
        self.SEG = SEG
        self.WIN = WIN

        self.SEGFS = self.SEG * FS
        self.WINFS = self.WIN * FS
        self.STEPS = self.WIN - self.SEG + 1

        self.Refs = {}
    pass #def

    def Clone(self):
        return Detector(self.SEG, self.WIN)
    pass #def

    def Detect(self, rec: Record, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(len(rec.Sensor), dtype=int), np.zeros(len(rec.Sensor), dtype=int)
    pass #def
pass #class

class EVAL():
    def __init__(self, ALGO: Detector, DB = "", chart = False, FILT: Callable[[np.ndarray], np.ndarray] | None = None):
        global FILTER
        if FILT is None: FILT = FILTER

        self.FILT = FILT
        self.ALGO = ALGO

        self.DB = DB
        self.Records = {}

        self.xReport = HEADER_IDN if ALIGN else HEADER_TAB
        self.dReport = HEADER_IDN if ALIGN else HEADER_TAB

        if DB: self.Process(DB, chart = chart)
    pass #def
        
    def Eval(self, rec: Record, chart = False) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
        secs = len(rec.Sensor) // FS

        Ref = rec.Flags
        Fit, Res = self.ALGO.Detect(rec, Ref)
        rec.Class = Fit

        ### Stat by sample ##############

        Ref = Ref[:secs*FS]
        Fit = Fit[:secs*FS]

        (TPx, FPx, FNx, TNx), linex = CFM(rec.RID, Ref * 2 + Fit)
        self.xReport += "\n" + linex

        ### Stat by decision #############

        (TPd, FPd, FNd, TNd), lined = CFM(rec.RID, Res)
        self.dReport += "\n" + lined

        if chart:
            pages = []
            for p in range((len(rec.Local) + plot.CHUNK - 1) // plot.CHUNK):
                PON, POF = p * plot.CHUNK, (p + 1) * plot.CHUNK
                if np.sum(Ref[PON:POF]) + np.sum(Fit[PON:POF]) > 0:
                    pages.append(p)
                pass #if
            pass #for
            if len(pages) > 0:
                PlotRecord(rec, pages, title = HEADER_IDN + "\n" +linex)
            pass #if
        else:
            print(linex)
        pass #if

        return (TPx, FPx, FNx, TNx), (TPd, FPd, FNd, TNd)
    pass #def

    def Read(self, db: str, rid: str) -> Record:
        rec = Record(db, rid, bfx = True)
        rec.Local = self.FILT(rec.Sensor)

        mask = np.zeros(len(rec.Sensor), dtype=int)
        for ep in rec.RefEpi:
            # print(epi.Name, epi.End, epi.Len)
            if ep.Name in ["VFL", "VF", "WF", "#VT"]:
                mask[ep.Time:ep.End] = 1
            pass #if
        pass #for
        rec.Flags = mask

        return rec
    pass #def

    def Load(self, db: str, rids: list[str] = [], exe = False):
        if exe:
            EXG(db, learn=True, cfm=True, bfx = True, devx=False)
        pass #if
        rids = rids if len(rids) > 0 else Rids(db)
        self.DB = db
        self.Records = {}
        # start time in millis
        start = time.time() * 1000
        # print(f"{db.upper()} [", end="")
        for rid in tqdm(rids, ncols=120, desc=f'> {db.upper()}'):
            rec = self.Read(db, rid)
            self.Records[rid] = rec
            # print(">", end="")
        pass #for
        elapsed = (time.time() * 1000 - start)
        # print(f"] {elapsed:.0f} ms")
        return self
    pass #def

    def Process(self, db: str = "", rid: str | list[str] = [], exe = False, chart = False):
        if exe:
            EXG(db, learn=True, cfm=True, bfx = True, var_qrx=True, devx=False)
        pass #if

        self.xReport = HEADER_IDN if ALIGN else HEADER_TAB
        self.dReport = HEADER_IDN if ALIGN else HEADER_TAB
        if not chart:
            if REPORTW:
                print()
                print("Continuous Report, per sample") 
                print(DASH)
            pass #if
            print(HEADER_IDN if ALIGN else HEADER_TAB)
        pass #if

        if db == "": db = self.DB

        rids = []
        if rid and isinstance(rid, str):
            rids = [rid]
        elif rid and isinstance(rid, list):
            rids = rid
        else:
            rids = Rids(db)
        pass #if

        TPx = FPx = FNx = TNx = 0
        TPd = FPd = FNd = TNd = 0
        
        rids = rids if len(rids) > 0 else Rids(db)
        for rid in rids:
            if rid in self.Records:
                rec = self.Records[rid]
            else:
                rec = self.Read(db, rid)
            pass #if
            (tpx, fpx, fnx, tnx), (tpd, fpd, fnd, tnd) = self.Eval(rec, chart)
            TPx += tpx; FPx += fpx; FNx += fnx; TNx += tnx
            TPd += tpd; FPd += fpd; FNd += fnd; TNd += tnd
        pass #for

        linex = Line("ALL", TPx, FPx, FNx, TNx)
        self.xReport += "\n" + linex

        if db in self.ALGO.Refs: self.dReport += "\n" + self.ALGO.Refs[db]
        lined = Line("ALL", TPd, FPd, FNd, TNd)
        self.dReport += "\n" + lined

        if chart:
            print("\n\n")
            print("Continuous Report, per sample") 
            print(DASH)
            print(self.xReport)
        else:
            print(linex)
        pass #if

        if REPORTW:
            print("\n\n")
            print("Window Report, size:", self.ALGO.WIN, "s")
            print(DASH)
            print(self.dReport)
    pass #def
pass #class