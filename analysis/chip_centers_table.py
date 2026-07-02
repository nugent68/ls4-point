#!/usr/bin/env python3
"""Table of TELE pointing + WCS chip-center for one chip/filter in proc/std.
Generalization of swd_i_table.py.  Columns (decimal degrees):
  image_name  TELE_RA  TELE_DEC  chip_ra  chip_dec
chip_ra/dec = WCS sky coords at chip-center pixel (1024,2048), FITS 1-indexed.

  chip_centers_table.py --chip NE_A --filt i
"""
import glob, os, sys, math, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import Angle
import astropy.units as u
from multiprocessing import Pool

ROOT = "/pscratch/sd/n/nugent/ls4/proc/std"


def _dec(v):
    if v is None: return float("nan")
    if isinstance(v, (int, float)): return float(v)
    s = str(v).strip()
    try: return float(s)
    except ValueError: return Angle(s, unit=u.deg).deg


def _ra(v, ref):
    if v is None: return float("nan")
    if isinstance(v, (int, float)): cands = [float(v), float(v) * 15.0]
    else:
        s = str(v).strip()
        try:
            x = float(s); cands = [x, x * 15.0]
        except ValueError:
            cands = []
            for unit in (u.hourangle, u.deg):
                try: cands.append(Angle(s, unit=unit).deg)
                except Exception: pass
    if not cands: return float("nan")
    if math.isnan(ref): return cands[0]
    return min(cands, key=lambda a: abs((a - ref + 180) % 360 - 180))


def one(f):
    try:
        with fits.open(f) as hd:
            i = next(k for k, h in enumerate(hd)
                     if h.data is not None and getattr(h.data, "ndim", 0) == 2)
            H = hd[i].header
            cra, cdec = WCS(H).all_pix2world(1024, 2048, 1)
            cra = float(cra) % 360.0; cdec = float(cdec)
            tdec = _dec(H.get("TELE-DEC", H.get("TELEDEC")))
            tra = _ra(H.get("TELE-RA", H.get("TELERA")), cra)
            tra = tra % 360.0 if not math.isnan(tra) else tra
        return (os.path.basename(f), tra, tdec, cra, cdec)
    except Exception as e:
        sys.stderr.write(f"ERR {f}: {e}\n"); return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", required=True)        # e.g. NE_A
    ap.add_argument("--filt", required=True)         # e.g. i
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pat = f"{ROOT}/*/*/ls4_*_{a.chip}_{a.filt}_Sci_*image.w.fits.fz"
    files = sorted(glob.glob(pat))
    print(f"# {len(files)} {a.chip} {a.filt}-band science images", file=sys.stderr)
    with Pool(64) as p:
        rows = [r for r in p.imap(one, files, chunksize=16) if r]
    out = a.out or f"/pscratch/sd/n/nugent/ls4/stack/{a.chip.lower()}_{a.filt}_centers.csv"
    with open(out, "w") as fh:
        fh.write("image_name,TELE_RA,TELE_DEC,chip_ra,chip_dec\n")
        for n, tra, tdec, cra, cdec in rows:
            fh.write(f"{n},{tra:.6f},{tdec:.6f},{cra:.6f},{cdec:.6f}\n")
    print(f"# wrote {out} ({len(rows)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
