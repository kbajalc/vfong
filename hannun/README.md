# hannun/

Material for reproducing the LDIAED design (a small residual 1-D CNN, descended from Hannun et al. 2019).

## pdf/
- `RAJPURKAR-2017.pdf`: arXiv 1707.01836, the preprint of the Hannun network (34-layer ResNet, 200 Hz, one label per 256 samples).
- `LDIAED-2022.pdf`: Nasimi and Yazdchi, PLoS ONE 17(2):e0264405. Copy of `docs/papers/journal.pone.0264405.pdf`.
- `HAI-2021.pdf`: arXiv 2107.06176, kNN shock advice on CUDB+VFDB (8 s, 250 Hz, random 5-fold).
- Hannun 2019 (Nature Medicine 25:65-69) is not here: the open-access download was blocked. Add it by hand if wanted.

## ref/
- `awni-ecg/`: snapshot of github.com/awni/ecg at commit fc2d0e26ad4d (2022-04-26), `.git` removed. Python 2.7 / Keras. Reference only, not run. `ecg/network.py` is the architecture; `examples/cinc17/config.json` holds the hyperparameters LDIAED reuses.
