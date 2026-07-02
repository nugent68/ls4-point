#!/usr/bin/env python3
"""Focal-plane-center pointing offset and true->TELE converter.

FP center = cos-Dec-corrected (spherical) mean of the NE_A and SW_D i-band
chip centers.  delta = FP_center - TELE (coordinate degrees).  Restricted to
the post-shift era (nights >= 20260502).

Plots:  delta_Dec vs RA   and   delta_RA vs Dec   (colored by night).
Fit:    delta_RA, delta_Dec as bilinear functions of true (FP) RA, Dec, so
        TELE = true - delta(true)  ->  converter true_to_tele(ra, dec).
"""
import csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

CUT = "20260502"


def load(fn):
    d = {}
    for r in csv.DictReader(open(fn)):
        p = r["image_name"].split("_")
        d[(p[1], p[2])] = (float(r["TELE_RA"]), float(r["TELE_DEC"]),
                           float(r["chip_ra"]), float(r["chip_dec"]))
    return d


ne, sw = load("ne_a_i_centers.csv"), load("swd_i_centers.csv")
keys = sorted(k for k in (set(ne) & set(sw)) if k[0] >= CUT)
night = np.array([k[0] for k in keys])
na = np.array([ne[k] for k in keys]); sd = np.array([sw[k] for k in keys])
tra, tdec = na[:, 0], na[:, 1]


def unit(ra, dec):
    r, d = np.radians(ra), np.radians(dec)
    return np.array([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])


v = (unit(na[:, 2], na[:, 3]) + unit(sd[:, 2], sd[:, 3])) / 2
v /= np.linalg.norm(v, axis=0)
fra = np.degrees(np.arctan2(v[1], v[0])) % 360.0      # FP-center RA (true)
fdec = np.degrees(np.arcsin(v[2]))                    # FP-center Dec (true)
dRA = (fra - tra + 180) % 360 - 180                   # FP - TELE, wrap-safe
dDec = fdec - tdec

# ---- bilinear fit: delta = c0 + c1*ra + c2*dec  (independent var = TRUE/FP pos) ----
A = np.c_[np.ones_like(fra), fra, fdec]
cRA, *_ = np.linalg.lstsq(A, dRA, rcond=None)
cDec, *_ = np.linalg.lstsq(A, dDec, rcond=None)
predRA, predDec = A @ cRA, A @ cDec
rmsRA = np.std(dRA - predRA); rmsDec = np.std(dDec - predDec)

# constant-only baseline for comparison
rms0RA, rms0Dec = dRA.std(), dDec.std()

print(f"N = {len(keys)} matched exposures, nights {min(night)}..{max(night)}")
print("delta_RA  = %.5f + %.6f*RA + %.6f*Dec    (deg)" % tuple(cRA))
print("delta_Dec = %.5f + %.6f*RA + %.6f*Dec    (deg)" % tuple(cDec))
print(f"RA  offset: const-only RMS {rms0RA*3600:6.1f}\"  ->  bilinear RMS {rmsRA*3600:6.1f}\"")
print(f"Dec offset: const-only RMS {rms0Dec*3600:6.1f}\"  ->  bilinear RMS {rmsDec*3600:6.1f}\"")


def true_to_tele(ra, dec):
    """Desired true on-sky FP-center (deg) -> requested TELE-RA, TELE-DEC (deg)."""
    dra = cRA[0] + cRA[1] * ra + cRA[2] * dec
    ddec = cDec[0] + cDec[1] * ra + cDec[2] * dec
    return (ra - dra) % 360.0, dec - ddec


# closed-loop check: predict TELE from true, compare to actual TELE
pt_ra, pt_dec = true_to_tele(fra, fdec)
res_ra = ((pt_ra - tra + 180) % 360 - 180) * 3600
res_dec = (pt_dec - tdec) * 3600
print(f"closed-loop TELE residual: RA {res_ra.std():.1f}\"  Dec {res_dec.std():.1f}\"  "
      f"(max |RA| {np.abs(res_ra).max():.1f}\", |Dec| {np.abs(res_dec).max():.1f}\")")

# ---------- plots ----------
nights = sorted(set(night)); idx = {n: i for i, n in enumerate(nights)}; ni = np.array([idx[n] for n in night])
Nn = len(nights); cmap = plt.cm.turbo; norm = BoundaryNorm(np.arange(-0.5, Nn + 0.5, 1), cmap.N)
sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])

fig, ax = plt.subplots(1, 2, figsize=(15, 6.4), constrained_layout=True)
# delta_Dec vs RA
ax[0].scatter(fra, dDec, c=ni, cmap=cmap, norm=norm, s=14, alpha=0.7, edgecolors="none")
xs = np.linspace(fra.min(), fra.max(), 100)
ax[0].plot(xs, cDec[0] + cDec[1] * xs + cDec[2] * np.median(fdec), "k-", lw=1.5,
           label=f"fit @ Dec={np.median(fdec):.0f}°  (RMS {rmsDec*3600:.0f}\")")
ax[0].set_xlabel("true (FP-center) RA (deg)"); ax[0].set_ylabel("ΔDec = FP_Dec − TELE_DEC (deg)")
ax[0].set_title("Dec offset vs RA"); ax[0].legend(); ax[0].grid(alpha=0.25); ax[0].invert_xaxis()
# delta_RA vs Dec
ax[1].scatter(fdec, dRA, c=ni, cmap=cmap, norm=norm, s=14, alpha=0.7, edgecolors="none")
xs = np.linspace(fdec.min(), fdec.max(), 100)
ax[1].plot(xs, cRA[0] + cRA[1] * np.median(fra) + cRA[2] * xs, "k-", lw=1.5,
           label=f"fit @ RA={np.median(fra):.0f}°  (RMS {rmsRA*3600:.0f}\")")
ax[1].set_xlabel("true (FP-center) Dec (deg)"); ax[1].set_ylabel("ΔRA = FP_RA − TELE_RA (deg)")
ax[1].set_title("RA offset vs Dec"); ax[1].legend(); ax[1].grid(alpha=0.25)

cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
ti = np.linspace(0, Nn - 1, min(Nn, 14)).round().astype(int)
cb.set_ticks(ti); cb.set_ticklabels([nights[i] for i in ti]); cb.set_label("observing night")
fig.suptitle(f"LS4 focal-plane-center pointing offset (NE_A+SW_D mean), post-shift ≥{CUT} — {len(keys)} exp",
             fontsize=13)
fig.savefig("/Users/nugent/claude/ls4/pointing/fp_pointing_offset.png", dpi=140)
print("wrote fp_pointing_offset.png")
