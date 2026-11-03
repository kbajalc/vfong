"""Training loop and fold runner for vfnn.

Recipe from the design: He-normal init, Adam, learning rate divided by 10 when the dev loss
stalls for two epochs, best dev-loss model kept, binary cross-entropy with ``pos_weight``.

    python -m vfnn.train --split group --seconds 8 --dbs vfdb cudb mitdb
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from vfnn import data
from vfnn.data import Store, Windows
from vfnn.eval import best_f1_threshold, false_positive_table, per_record, scores, table
from vfnn.model import B_LITE, B_MID, Config, VFNet, macs, n_params


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    pass #if
    if torch.backends.mps.is_available():
        return torch.device("mps")
    pass #if
    return torch.device("cpu")
pass #def


@dataclass
class Fit:
    epochs: int = 30
    batch: int = 64
    lr: float = 1e-3
    augment: bool = True
    patience: int = 6                 # epochs without a new best dev loss before stopping
    per_epoch: int | None = None      # random windows per epoch (None = all)
    seed: int = 0
    verbose: bool = False             # one progress line per epoch
    tag: str = ""                     # label for the progress lines (the fold name)
    log: list = field(default_factory=list)
pass #class


def predict(model: nn.Module, win: Windows, device, store: Store | None = None, batch: int = 1024) -> np.ndarray:
    """Positive-class probability for every window."""
    store = store or Store(win.recs, device)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(win), batch):
            x, _ = store.batch(win, np.arange(i, min(i + batch, len(win))))
            out.append(torch.sigmoid(model(x)).cpu().numpy())
        pass #for
    pass #with
    return np.concatenate(out) if out else np.zeros(0)
pass #def


def dev_loss(model: nn.Module, win: Windows, loss_fn, device, store: Store, batch: int = 1024) -> float:
    model.eval()
    tot = torch.zeros((), device=device)
    with torch.no_grad():
        for i in range(0, len(win), batch):
            x, y = store.batch(win, np.arange(i, min(i + batch, len(win))))
            tot += loss_fn(model(x), y) * len(y)
        pass #for
    pass #with
    return float(tot.item() / max(len(win), 1))
pass #def


def fit(
    train: Windows, dev: Windows, cfg: Config = B_LITE, opt: Fit = Fit(), device=None,
    store: Store | None = None,
) -> VFNet:
    """Train on ``train``, keep the weights with the lowest ``dev`` loss."""
    device = device or pick_device()
    store = store or Store(train.recs, device)
    torch.manual_seed(opt.seed)
    rng = np.random.default_rng(opt.seed)

    model = VFNet(cfg).to(device)
    pos = max(int(train.y.sum()), 1)
    pos_weight = torch.tensor([min((len(train) - pos) / pos, 20.0)], device=device)
    loss_train = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_dev = nn.BCEWithLogitsLoss()
    adam = torch.optim.Adam(model.parameters(), lr=opt.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(adam, factor=0.1, patience=1)

    best, best_state, stale = float("inf"), None, 0
    for epoch in range(opt.epochs):
        t0 = time.time()
        idx = rng.permutation(len(train))
        if opt.per_epoch and opt.per_epoch < len(idx):
            idx = idx[:opt.per_epoch]
        pass #if
        model.train()
        tr = torch.zeros((), device=device)
        n = 0
        for i in range(0, len(idx), opt.batch):
            b = idx[i:i + opt.batch]
            if len(b) < 2:
                continue
            pass #if
            x, y = store.batch(train, b, augment=opt.augment)
            adam.zero_grad()
            loss = loss_train(model(x), y)
            loss.backward()
            adam.step()
            tr += loss.detach() * len(b)
            n += len(b)
        pass #for
        dl = dev_loss(model, dev, loss_dev, device, store)
        sched.step(dl)
        dt = time.time() - t0
        opt.log.append((epoch, float(tr.item()) / max(n, 1), dl, adam.param_groups[0]["lr"], dt))
        improved = dl < best
        if improved:
            best, stale = dl, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        pass #if
        if opt.verbose:
            print(f"  [{opt.tag}] epoch {epoch + 1:2d}/{opt.epochs}  train {opt.log[-1][1]:.4f}  "
                  f"dev {dl:.4f}{' *' if improved else '  '}  lr {adam.param_groups[0]['lr']:.0e}  "
                  f"{dt:4.0f}s  (stale {stale}/{opt.patience})", flush=True)
        pass #if
        if stale >= opt.patience:
            break
        pass #if
    pass #for
    if best_state is not None:
        model.load_state_dict(best_state)
    pass #if
    return model
pass #def


def dev_split(idx: np.ndarray, win: Windows, frac: float, by_record: bool, seed: int):
    """Split training indices into (train, dev). Whole records go to dev unless by_record is False."""
    rng = np.random.default_rng(seed)
    if not by_record:
        perm = rng.permutation(idx)
        k = int(len(perm) * frac)
        return perm[k:], perm[:k]
    pass #if
    g = win.groups()[idx]
    keys = np.unique(g)
    # records with positives first, so that the dev set always holds some
    pos = {k: win.y[idx][g == k].sum() for k in keys}
    withpos = [k for k in keys if pos[k] > 0]
    nopos = [k for k in keys if pos[k] == 0]
    rng.shuffle(withpos)
    rng.shuffle(nopos)
    take = max(1, int(round(len(keys) * frac)))
    dev_keys = set(withpos[:max(1, take // 2)] + nopos[:take - max(1, take // 2)])
    mask = np.isin(g, list(dev_keys))
    return idx[~mask], idx[mask]
pass #def


def run_split(
    win: Windows,
    folds,
    labels: list[str],
    cfg: Config,
    opt: Fit,
    by_record: bool,
    dev_frac: float = 0.15,
    device=None,
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Train one model per fold and score it on the fold's test windows.

    Returns the per-fold rows and the per-record detail over all test folds.
    """
    rows, detail = [], []
    device = device or pick_device()
    store = Store(win.recs, device)
    for (tr, te), lab in zip(folds, labels):
        tr_idx, dev_idx = dev_split(tr, win, dev_frac, by_record, opt.seed)
        o = Fit(**{**opt.__dict__, "log": [], "tag": lab})
        if verbose:
            t_fold = time.time()
            o.verbose = True
            print(f"[{lab}] fold {len(rows) + 1}/{len(folds)}: train {len(tr_idx)} windows "
                  f"({int(win.y[tr_idx].sum())} pos), dev {len(dev_idx)} ({int(win.y[dev_idx].sum())} pos), "
                  f"test {len(te)} ({int(win.y[te].sum())} pos)", flush=True)
        pass #if
        model = fit(win.subset(tr_idx), win.subset(dev_idx), cfg, o, device, store)
        dev_win = win.subset(dev_idx)
        thr = best_f1_threshold(dev_win.y, predict(model, dev_win, device, store))
        test = win.subset(te)
        p = predict(model, test, device, store)
        r = scores(test.y, p, thr)
        r["thr"] = thr
        rows.append(r)
        for d in per_record(test.groups(), test.databases(), test.y, p, thr):
            detail.append({"fold": lab, **d})
        pass #for
        if verbose:
            print(f"[{lab}] done in {time.time() - t_fold:.0f}s, {len(o.log)} epochs  thr {thr:.3f}  "
                  f"AUC {r['auc']:.3f}  Se {r['se']:.3f}  Sp {r['sp']:.3f}  PPV {r['ppv']:.3f}  "
                  f"F1 {r['f1']:.3f}", flush=True)
        pass #if
    pass #for
    return rows, detail
pass #def


def _main() -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate vfnn under one split protocol.")
    ap.add_argument("--split", choices=["random", "group", "lodo"], default="group")
    ap.add_argument("--dbs", nargs="+", default=["vfdb", "cudb", "mitdb"])
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--net", choices=["lite", "mid"], default="lite")
    ap.add_argument("--vt-positive", action="store_true", help="count VT as positive (LDIAED task)")
    ap.add_argument("--ahadb-8-only", action="store_true",
                    help="keep only the annotated 8-series of ahadb (forced with --vt-positive)")
    ap.add_argument("--folds", type=int, default=None, help="number of folds (random 10, group 5)")
    ap.add_argument("--only", type=int, default=None, help="run only this fold index")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--per-epoch", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--out", default=None, help="CSV path for the per-fold rows")
    ap.add_argument("--detail-out", default=None, help="CSV path for the per-record rows")
    args = ap.parse_args()

    cfg = B_MID if args.net == "mid" else B_LITE
    only8 = args.ahadb_8_only or args.vt_positive
    if args.vt_positive and not args.ahadb_8_only:
        print("--vt-positive: using the annotated ahadb 8-series only (the other ahadb records "
              "have no rhythm labels, so their VT would count as negative)")
    pass #if
    t_start = time.time()
    print(f"loading and windowing {', '.join(args.dbs)} ...", flush=True)
    recs = data.load_records(tuple(args.dbs), ahadb_8_only=only8)
    win = data.make_windows(recs, args.seconds, vt_positive=args.vt_positive)
    model = VFNet(cfg)
    print(f"{len(recs)} records, {len(win)} windows, {int(win.y.sum())} positive "
          f"({win.y.mean():.3f}); net {args.net}: {n_params(model)} params, "
          f"{macs(model, win.size) / 1e6:.1f} MMAC/window; device {pick_device()}; "
          f"setup took {time.time() - t_start:.0f}s", flush=True)
    print(f"epochs {args.epochs} x {args.per_epoch or len(win)} windows, batch {args.batch}, "
          f"patience 6, augment {not args.no_augment}, seed {args.seed}", flush=True)

    if args.split == "random":
        folds = data.random_folds(len(win), args.folds or 10, args.seed)
        labels = [f"r{i}" for i in range(len(folds))]
    elif args.split == "group":
        folds = data.group_folds(win.groups(), win.y, args.folds or 5, args.seed)
        labels = [f"g{i}" for i in range(len(folds))]
    else:
        folds = data.leave_db_out(win.databases())
        labels = sorted(set(win.databases()))
    pass #if
    if args.only is not None:
        folds, labels = [folds[args.only]], [labels[args.only]]
    pass #if

    opt = Fit(epochs=args.epochs, batch=args.batch, seed=args.seed, per_epoch=args.per_epoch,
              augment=not args.no_augment)
    rows, detail = run_split(win, folds, labels, cfg, opt, by_record=args.split != "random")
    print(f"\nall folds finished in {time.time() - t_start:.0f}s\n")
    print(table(f"{args.split} / {args.seconds:g} s / {args.net}", rows, labels))
    print("\nfalse positives on records without positive windows")
    print(false_positive_table(detail))
    if args.out:
        import csv
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["fold"] + list(rows[0].keys()))
            w.writeheader()
            for lab, r in zip(labels, rows):
                w.writerow({"fold": lab, **r})
            pass #for
        pass #with
    pass #if
    if args.detail_out and detail:
        import csv
        os.makedirs(os.path.dirname(args.detail_out) or ".", exist_ok=True)
        with open(args.detail_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(detail[0].keys()))
            w.writeheader()
            w.writerows(detail)
        pass #with
    pass #if
pass #def


if __name__ == "__main__":
    _main()
pass #if
