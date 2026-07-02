#!/usr/bin/env python3
"""LS4 start-of-night pointing zero-point tool.

Measures the focal-plane-center <-> TELE pointing offset zero points (a0, b0)
directly from RAW LS4 exposures, for use at the telescope at the start of a
night (e.g. after any stow/TCS/dome intervention).

Per raw exposure (single MEF, 32 chip HDUs):
  1. extract the NE_A and SW_D chips (they bracket the FP center),
  2. overscan-subtract (+ optional superbias/flat/mask, small per-chip files),
  3. solve astrometry with a locally-built astrometry.net solve-field using
     its INTERNAL source detector (simplexy) and a minimal index set,
  4. FP center = unit-vector mean of the two chip centers (pixel 1024,2048),
  5. offset  dRA = FP_RA - TELE_RA, dDec = FP_Dec - TELE_DEC,
  6. subtract the known HA-flexure terms (June-2026 model) -> a0, b0.

The flexure model:
  dRA  = a0 + a1 sinH + a2 cosH + a3 sinH tanD + a4 cosH tanD
  dDec = b0 + b1 sinH + b2 cosH
a1..a4/b1..b2 are the GLOBAL cross-epoch fit (Apr-Jun 2026, 4610 exposures
with per-night zero points): all four engineering epochs, including
pre-stow, lie on the same flexure curve — CONFIRMED stable, only a0/b0
re-zero after engineering events.  H is computed from DATE-OBS.

Usage:
  python night_zero_point.py raw1.fits.fz [raw2 ...] [--no-calib] [--json out]
solve-field is located in this order: --solve-bin, <--astrom-dir>/bin/
solve-field, or $PATH (e.g. from `apt install astrometry.net`).  The index
config comes from --index-cfg, <--astrom-dir>/astrometry.cfg, or the system
default (/etc/astrometry.cfg with apt).  Calibration files (superbias/flat/
FP mask for the two chips) are looked up in --calib-dir, <--astrom-dir>/calib,
or <this script's dir>/calib.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from astropy.io import fits
from astropy.time import Time

# ---- raw chip geometry (same as calibrate_ls4.py) --------------------------
DATA_X = slice(6, 2054)          # 2048 data columns
DATA_Y = slice(0, 4096)          # 4096 rows
BIAS_X = slice(2054, 2074)       # 20 overscan columns
CENPIX = (1024.0, 2048.0)        # chip center (FITS 1-indexed x, y)

# ---- La Silla + June-2026 flexure model (from pointing/true_to_tele.py) ----
LAT, LON = -29.2563, -70.7397
# a0/b0 = June-2026 epoch reference; a1../b1.. = global cross-epoch flexure
A_RA = (-1.92916, 0.00078, -0.01361, 0.03608, -0.01553)    # a0..a4
A_DEC = (-0.45011, 0.00083, -0.04654)                      # b0..b2
FLAT_DATE = "20260421"


def log(msg):
    print(msg, flush=True)


def find_solve_bin(args):
    if args.solve_bin:
        return args.solve_bin
    if args.astrom_dir:
        p = os.path.join(args.astrom_dir, "bin", "solve-field")
        if os.path.exists(p):
            return p
    p = shutil.which("solve-field")           # e.g. apt install astrometry.net
    if p:
        return p
    raise SystemExit("solve-field not found: use --solve-bin, --astrom-dir, "
                     "or install astrometry.net on PATH")


def find_index_cfg(args):
    if args.index_cfg:
        return args.index_cfg
    if args.astrom_dir:
        p = os.path.join(args.astrom_dir, "astrometry.cfg")
        if os.path.exists(p):
            return p
    return None                               # fall back to system default cfg


def find_calib_dir(args):
    for d in (args.calib_dir,
              os.path.join(args.astrom_dir, "calib") if args.astrom_dir else None,
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib")):
        if d and os.path.isdir(d):
            return d
    return None


def mjd_from_dateobs(dobs, exptime=0.0):
    t = Time(dobs, scale="utc", format="isot")
    return t.mjd + 0.5 * float(exptime or 0.0) / 86400.0


def hour_angle_deg(mjd, ra_deg):
    d = mjd - 51544.5
    gmst = (18.697374558 + 24.06570982441908 * d) % 24.0 * 15.0
    lst = (gmst + LON) % 360.0
    return (lst - ra_deg + 180.0) % 360.0 - 180.0


def read_first_2d(path):
    with fits.open(path) as h:
        for hd in h:
            if hd.data is not None and getattr(hd.data, "ndim", 0) == 2:
                return hd.data.astype(np.float32)
    raise IOError(f"no 2-D image in {path}")


def extract_chips(rawpath, chips):
    """Return dict chip -> (trimmed float32 image, chip header) + exposure meta."""
    out, meta = {}, {}
    with fits.open(rawpath) as hdul:
        for hdu in hdul:
            loc = str(hdu.header.get("CCD_LOC", "")).strip()
            if loc not in chips:
                continue
            arr = hdu.data
            if arr is None:
                continue
            raw = arr.astype(np.float32)
            bias_med = np.median(raw[DATA_Y, BIAS_X], axis=1)
            img = raw[DATA_Y, DATA_X] - bias_med[:, None]
            out[loc] = (img, hdu.header.copy())
            if not meta:
                h = hdu.header
                tra = float(h["TELE-RA"])
                if tra < 24.0:              # raw headers carry RA in hours
                    tra *= 15.0
                meta = {"tele_ra": tra % 360.0,
                        "tele_dec": float(h["TELE-DEC"]),
                        "dateobs": h.get("DATE-OBS"),
                        "exptime": float(h.get("EXPTIME", 0.0))}
    missing = set(chips) - set(out)
    if missing:
        raise IOError(f"{os.path.basename(rawpath)}: chips not found: {missing}")
    return out, meta


def apply_calib(img, chip, calib_dir):
    """superbias + flat + FP mask; masked/bad pixels -> image median."""
    bias_f = os.path.join(calib_dir, f"superbias_{chip}.fits.fz")
    flat_f = os.path.join(calib_dir, f"LS4Cam_flat_{FLAT_DATE}_{chip}.fits")
    mask_f = os.path.join(calib_dir, f"FP_mask_{chip}.fits.fz")
    bad = np.zeros(img.shape, dtype=bool)
    if os.path.exists(bias_f):
        img = img - read_first_2d(bias_f)
    if os.path.exists(flat_f):
        flat = read_first_2d(flat_f)
        badflat = ~np.isfinite(flat) | (flat <= 0)
        img = img / np.where(badflat, 1.0, flat)
        bad |= badflat
    if os.path.exists(mask_f):
        bad |= (read_first_2d(mask_f) != 0)
    if bad.any():
        img = img.copy()
        img[bad] = np.median(img[~bad])
    return img


def solve_chip(img, chip, meta, args, workdir):
    """Write temp FITS, run solve-field (internal detector), return (WCS, dt)."""
    from astropy.wcs import WCS
    fitspath = os.path.join(workdir, f"{chip}.fits")
    wcspath = os.path.join(workdir, f"{chip}.wcs")
    fits.PrimaryHDU(img.astype(np.float32)).writeto(fitspath, overwrite=True)

    sf = find_solve_bin(args)
    cfg = find_index_cfg(args)
    cmd = [sf] + (["--config", cfg] if cfg else []) + ["--fits-image",
           "--ra", f"{meta['tele_ra']:.4f}", "--dec", f"{meta['tele_dec']:.4f}",
           "--radius", str(args.radius),
           "--scale-units", "arcsecperpix",
           "--scale-low", "0.95", "--scale-high", "1.05",
           "--downsample", str(args.downsample),
           "--cpulimit", str(args.cpulimit),
           "--no-plots", "--overwrite", "--dir", workdir,
           "--new-fits", "none", "--wcs", wcspath,
           "--corr", "none", "--rdls", "none", "--match", "none",
           "--solved", "none", "--index-xyls", "none",
           fitspath]
    if args.use_se:
        cmd += ["--use-source-extractor"]
        if args.se_path:
            cmd += ["--source-extractor-path", args.se_path]
        if args.se_config:
            cmd += ["--source-extractor-config", args.se_config]
    env = dict(os.environ)
    if args.astrom_dir and os.path.isdir(os.path.join(args.astrom_dir, "lib")):
        env["LD_LIBRARY_PATH"] = (os.path.join(args.astrom_dir, "lib") + ":"
                                  + env.get("LD_LIBRARY_PATH", ""))
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    if not os.path.exists(wcspath):
        tail = "\n".join((r.stdout + r.stderr).splitlines()[-12:])
        raise RuntimeError(f"{chip}: solve-field FAILED ({dt:.0f}s)\n{tail}")
    return WCS(fits.getheader(wcspath)), dt


def fp_center(centers):
    """Unit-vector mean of [(ra, dec), ...] in deg."""
    v = np.zeros(3)
    for ra, dec in centers:
        r, d = math.radians(ra), math.radians(dec)
        v += [math.cos(d) * math.cos(r), math.cos(d) * math.sin(r), math.sin(d)]
    v /= np.linalg.norm(v)
    return math.degrees(math.atan2(v[1], v[0])) % 360.0, math.degrees(math.asin(v[2]))


def flexure_terms(ha_deg, dec_deg):
    """The HA-dependent parts (WITHOUT a0/b0)."""
    H = math.radians(ha_deg)
    sH, cH, tD = math.sin(H), math.cos(H), math.tan(math.radians(dec_deg))
    fra = A_RA[1] * sH + A_RA[2] * cH + A_RA[3] * sH * tD + A_RA[4] * cH * tD
    fdec = A_DEC[1] * sH + A_DEC[2] * cH
    return fra, fdec


def process_raw(rawpath, args):
    name = os.path.basename(rawpath)
    chips, meta = extract_chips(rawpath, args.chips)
    workdir = args.workdir or tempfile.mkdtemp(prefix="nzp_")
    os.makedirs(workdir, exist_ok=True)
    calib_dir = find_calib_dir(args)
    centers, times = [], {}
    try:
        for chip, (img, hdr) in sorted(chips.items()):
            if not args.no_calib and calib_dir:
                img = apply_calib(img, chip, calib_dir)
            w, dt = solve_chip(img, chip, meta, args, workdir)
            ra, dec = w.all_pix2world(CENPIX[0], CENPIX[1], 1)
            centers.append((float(ra) % 360.0, float(dec)))
            times[chip] = dt
    finally:
        if not args.keep and not args.workdir:
            shutil.rmtree(workdir, ignore_errors=True)

    fra, fdec = fp_center(centers)
    dra = (fra - meta["tele_ra"] + 180.0) % 360.0 - 180.0
    ddec = fdec - meta["tele_dec"]
    mjd = mjd_from_dateobs(meta["dateobs"], meta["exptime"])
    ha = hour_angle_deg(mjd, meta["tele_ra"])
    flx_ra, flx_dec = flexure_terms(ha, meta["tele_dec"])
    a0, b0 = dra - flx_ra, ddec - flx_dec
    rec = dict(raw=name, mjd=round(mjd, 6), tele_ra=meta["tele_ra"],
               tele_dec=meta["tele_dec"], ha=round(ha, 3),
               chips={c: {"ra": centers[i][0], "dec": centers[i][1],
                          "t_solve": round(times[c], 1)}
                      for i, c in enumerate(sorted(chips))},
               dRA=round(dra, 5), dDec=round(ddec, 5),
               a0=round(a0, 5), b0=round(b0, 5))
    log(f"  {name}: HA={ha:+7.2f}  dRA={dra:+.4f} dDec={ddec:+.4f}"
        f"  ->  a0={a0:+.4f}  b0={b0:+.4f}   "
        f"(solve {' '.join(f'{c}:{times[c]:.0f}s' for c in sorted(times))})")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("raws", nargs="+", help="raw LS4 exposure(s) (.fits.fz)")
    ap.add_argument("--astrom-dir", default=None,
                    help="self-built install dir (bin/solve-field, "
                         "astrometry.cfg, lib/, calib/); omit for a system "
                         "install (apt astrometry.net)")
    ap.add_argument("--solve-bin", default=None,
                    help="explicit solve-field path (overrides --astrom-dir/PATH)")
    ap.add_argument("--calib-dir", default=None,
                    help="dir with superbias/flat/FP_mask files "
                         "(default: <astrom-dir>/calib or <script dir>/calib)")
    ap.add_argument("--chips", nargs=2, default=["NE_A", "SW_D"])
    ap.add_argument("--index-cfg", default=None,
                    help="index config (default: <astrom-dir>/astrometry.cfg "
                         "or the system default, e.g. /etc/astrometry.cfg)")
    ap.add_argument("--radius", type=float, default=5.0)
    ap.add_argument("--cpulimit", type=int, default=30)
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--no-calib", action="store_true",
                    help="skip superbias/flat/mask (overscan-only)")
    ap.add_argument("--use-se", action="store_true",
                    help="use source-extractor instead of internal simplexy")
    ap.add_argument("--se-path", default=None)
    ap.add_argument("--se-config", default=None)
    ap.add_argument("--workdir", default=None, help="keep intermediates here")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    log(f"[night_zero_point] chips={args.chips}  astrom={args.astrom_dir}"
        f"  cfg={args.index_cfg or 'default'}  calib={not args.no_calib}"
        f"  detector={'source-extractor' if args.use_se else 'internal'}")
    recs, fails = [], []
    for raw in args.raws:
        try:
            recs.append(process_raw(raw, args))
        except Exception as e:
            fails.append(os.path.basename(raw))
            log(f"  {os.path.basename(raw)}: ERROR {e}")

    if recs:
        a0s = np.array([r["a0"] for r in recs]); b0s = np.array([r["b0"] for r in recs])
        log(f"\n  ZERO POINTS over {len(recs)} exposure(s):")
        log(f"    a0 = {a0s.mean():+.4f} deg  (sigma {a0s.std()*3600:.1f}\")")
        log(f"    b0 = {b0s.mean():+.4f} deg  (sigma {b0s.std()*3600:.1f}\")")
        log(f"    request:  TELE_RA = true_RA - ({a0s.mean():+.4f} + flexure_RA(H,Dec))")
        log(f"              TELE_DEC = true_Dec - ({b0s.mean():+.4f} + flexure_Dec(H))")
    if fails:
        log(f"  FAILED: {fails}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"exposures": recs, "failed": fails,
                       "a0_mean": float(a0s.mean()) if recs else None,
                       "b0_mean": float(b0s.mean()) if recs else None,
                       "flexure_A_RA": A_RA, "flexure_A_DEC": A_DEC}, f, indent=1)
        log(f"  wrote {args.json_out}")
    sys.exit(1 if fails and not recs else 0)


if __name__ == "__main__":
    main()
