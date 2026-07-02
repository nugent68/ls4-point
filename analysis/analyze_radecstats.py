#!/usr/bin/env python3
"""Repeat the FP-center pointing analysis using radecstats.csv (larger list,
processing suffix PIYORP, nights 20260501..20260626).

Grabs NE_A + SW_D, matches by exposure, FP center = cos-Dec-corrected mean of
the two chip centers, delta = FP - exposure(TELE).  Robustly drops WCS-failure
outliers.  Makes delta_Dec-vs-RA and delta_RA-vs-Dec plots, per-night drift,
and fits a true->TELE converter on the post-shift (>=20260502) era.
"""
import csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

CUT = "20260502"
GATE = 0.75          # deg: |delta - median| gate to reject bad-WCS frames

rows = list(csv.DictReader(open("radecstats.csv")))
ne, sw = {}, {}
for r in rows:
    c = r["chip"]
    if c not in ("NE_A", "SW_D"):
        continue
    p = r["image"].split("/")            # ls4/<night>/<exp>/<name>
    k = (p[1], p[2])
    rec = (float(r["exposure_ra"]), float(r["exposure_dec"]),
           float(r["wcs_image_ctr_ra"]), float(r["wcs_image_ctr_dec"]), float(r["mjd"]))
    (ne if c == "NE_A" else sw)[k] = rec

keys = sorted(set(ne) & set(sw))
na = np.array([ne[k] for k in keys]); sd = np.array([sw[k] for k in keys])
night = np.array([k[0] for k in keys]); mjd = na[:, 4]
tra, tdec = na[:, 0], na[:, 1]


def unit(ra, dec):
    r, d = np.radians(ra), np.radians(dec)
    return np.array([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])


v = (unit(na[:, 2], na[:, 3]) + unit(sd[:, 2], sd[:, 3])) / 2
v /= np.linalg.norm(v, axis=0)
fra = np.degrees(np.arctan2(v[1], v[0])) % 360.0
fdec = np.degrees(np.arcsin(v[2]))
dRA = (fra - tra + 180) % 360 - 180
dDec = fdec - tdec

# robust outlier rejection (bad WCS on a chip -> FP center far off)
good = (np.abs(dRA - np.median(dRA)) < GATE) & (np.abs(dDec - np.median(dDec)) < GATE)
print(f"matched {len(keys)}, dropped {(~good).sum()} WCS outliers (>|{GATE}| deg), kept {good.sum()}")
for a in (fra, fdec, dRA, dDec, tra, tdec, night, mjd):
    pass
fra, fdec, dRA, dDec, tra, tdec, night, mjd = (x[good] for x in (fra, fdec, dRA, dDec, tra, tdec, night, mjd))

post = night >= CUT
print(f"pre <{CUT}: N={(~post).sum()}   post: N={post.sum()}")
print(f"post medians: dRA {np.median(dRA[post]):+.3f}  dDec {np.median(dDec[post]):+.3f}")

# within vs between night (post)
ns = sorted(set(night[post]))
wR = np.median([dRA[post][night[post] == n].std() for n in ns]) * 3600
wD = np.median([dDec[post][night[post] == n].std() for n in ns]) * 3600
bR = np.std([np.median(dRA[post][night[post] == n]) for n in ns]) * 3600
bD = np.std([np.median(dDec[post][night[post] == n]) for n in ns]) * 3600
print(f"within-night std : dRA {wR:.0f}\"  dDec {wD:.0f}\"")
print(f"between-night std: dRA {bR:.0f}\"  dDec {bD:.0f}\"")

# ---- bilinear converter fit on post-shift ----
fp_ra, fp_dec = fra[post], fdec[post]
A = np.c_[np.ones_like(fp_ra), fp_ra, fp_dec]
cRA = np.linalg.lstsq(A, dRA[post], rcond=None)[0]
cDec = np.linalg.lstsq(A, dDec[post], rcond=None)[0]
rmsRA = np.std(dRA[post] - A @ cRA) * 3600
rmsDec = np.std(dDec[post] - A @ cDec) * 3600
print("delta_RA  = %.5f + %.6f*RA + %.6f*Dec" % tuple(cRA))
print("delta_Dec = %.5f + %.6f*RA + %.6f*Dec" % tuple(cDec))
print(f"bilinear residual RMS: RA {rmsRA:.0f}\"  Dec {rmsDec:.0f}\"  "
      f"(const-only RA {dRA[post].std()*3600:.0f}\" Dec {dDec[post].std()*3600:.0f}\")")

# ---------- the two requested plots (post-shift) ----------
nn = sorted(set(night[post])); idx = {n: i for i, n in enumerate(nn)}
ni = np.array([idx[n] for n in night[post]]); Nn = len(nn)
cmap = plt.cm.turbo; norm = BoundaryNorm(np.arange(-0.5, Nn + 0.5, 1), cmap.N)
sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
fig, ax = plt.subplots(1, 2, figsize=(15, 6.4), constrained_layout=True)
ax[0].scatter(fp_ra, dDec[post], c=ni, cmap=cmap, norm=norm, s=12, alpha=0.7, edgecolors="none")
ax[0].axhline(np.median(dDec[post]), color="k", ls=":", lw=1, label=f"median {np.median(dDec[post]):+.3f}")
ax[0].set_xlabel("true (FP-center) RA (deg)"); ax[0].set_ylabel("ΔDec = FP_Dec − TELE (deg)")
ax[0].set_title("Dec offset vs RA"); ax[0].legend(); ax[0].grid(alpha=0.25); ax[0].invert_xaxis()
ax[1].scatter(fp_dec, dRA[post], c=ni, cmap=cmap, norm=norm, s=12, alpha=0.7, edgecolors="none")
ax[1].axhline(np.median(dRA[post]), color="k", ls=":", lw=1, label=f"median {np.median(dRA[post]):+.3f}")
ax[1].set_xlabel("true (FP-center) Dec (deg)"); ax[1].set_ylabel("ΔRA = FP_RA − TELE (deg)")
ax[1].set_title("RA offset vs Dec"); ax[1].legend(); ax[1].grid(alpha=0.25)
cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
ti = np.linspace(0, Nn - 1, min(Nn, 14)).round().astype(int)
cb.set_ticks(ti); cb.set_ticklabels([nn[i] for i in ti]); cb.set_label("observing night")
fig.suptitle(f"radecstats.csv FP-center offset (NE_A+SW_D), post-shift ≥{CUT} — {post.sum()} exp",
             fontsize=13)
fig.savefig("fp_pointing_offset_rds.png", dpi=140); print("wrote fp_pointing_offset_rds.png")

# ---------- per-night drift (use mjd order) ----------
fig2, axd = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
x = np.arange(len(nn))
axd[0].errorbar(x, [np.median(dRA[post][night[post]==n]) for n in nn],
                yerr=[dRA[post][night[post]==n].std() for n in nn], fmt='o-', color="tab:blue")
axd[0].set_ylabel("ΔRA (deg)"); axd[0].set_title("radecstats: per-night FP-center offset (post-shift)")
axd[1].errorbar(x, [np.median(dDec[post][night[post]==n]) for n in nn],
                yerr=[dDec[post][night[post]==n].std() for n in nn], fmt='s-', color="tab:red")
axd[1].set_ylabel("ΔDec (deg)"); axd[1].set_xticks(x); axd[1].set_xticklabels(nn, rotation=90, fontsize=7)
for a in axd: a.grid(alpha=0.3)
fig2.savefig("fp_offset_per_night_rds.png", dpi=140); print("wrote fp_offset_per_night_rds.png")

# save fitted coefficients for the converter
np.savez("fp_fit_rds.npz", cRA=cRA, cDec=cDec)
print("saved fp_fit_rds.npz")
