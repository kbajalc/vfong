
### EXPERIMENTAL ################################

# def Tester(rec: Record):
#     rec.Signal = SignalFilter(rec.Sensor, L = LYN_WIND, M = MED_WIND * 1, H = LYN_HIGH * 0)
#     rec.Local = rec.Signal.copy()

#     hb, delay = HilbertFilter(rec.Local)
#     rec.Cover = np.sqrt(rec.Local**2 + hb**2)
#     rec.Maxim = np.sqrt(np.convolve(rec.Local ** 2, np.ones(500) / 500, mode="same")) 
#     rec.Digit = np.sign(np.maximum(0, rec.Cover - rec.Maxim))
    
#     rec.Class = np.convolve(rec.Digit, np.ones(500), mode="same") / 500 * 100
#     rec.Class = np.sign(np.maximum(rec.Class - 40, 0))

#     return rec.Class
# pass #def