#!/usr/bin/env python3
"""Physical pointing analysis: MJD -> LST(La Silla) -> hour angle + altitude,
then fit the FP-center offset (and its rotation) against a standard equatorial
pointing model.  Tests whether HA/alt explains the night-to-night drift that a
pure (RA,Dec) fit could not.

Uses radecstats.csv, chips NE_A + SW_D, post-shift era (>=20260502).
"""
import csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

# La Silla (ESO Schmidt)
LAT = -29.2563; LON = -70.7397
CUT = "20260502"; GATE = 0.75

rows = list(csv.DictReader(open("radecstats.csv")))
ne, sw = {}, {}
for r in rows:
    c = r["chip"]
    if c not in ("NE_A", "SW_D"):
        continue
    p = r["image"].split("/"); k = (p[1], p[2])
    rec = (float(r["exposure_ra"]), float(r["exposure_dec"]),
           float(r["wcs_image_ctr_ra"]), float(r["wcs_image_ctr_dec"]), float(r["mjd"]))
    (ne if c == "NE_A" else sw)[k] = rec

keys = sorted(k for k in (set(ne) & set(sw)) if k[0] >= CUT)
na = np.array([ne[k] for k in keys]); sd = np.array([sw[k] for k in keys])
night = np.array([k[0] for k in keys])
tra, tdec, mjd = na[:, 0], na[:, 1], na[:, 4]


def unit(ra, dec):
    r, d = np.radians(ra), np.radians(dec)
    return np.array([np.cos(d)*np.cos(r), np.cos(d)*np.sin(r), np.sin(d)])


v = (unit(na[:, 2], na[:, 3]) + unit(sd[:, 2], sd[:, 3])) / 2
v /= np.linalg.norm(v, axis=0)
fra = np.degrees(np.arctan2(v[1], v[0])) % 360.0
fdec = np.degrees(np.arcsin(v[2]))
dRA = (fra - tra + 180) % 360 - 180
dDec = fdec - tdec
good = (np.abs(dRA - np.median(dRA)) < GATE) & (np.abs(dDec - np.median(dDec)) < GATE)
tra, tdec, mjd, dRA, dDec, night = (x[good] for x in (tra, tdec, mjd, dRA, dDec, night))
print(f"post-shift, cleaned: {good.sum()} exposures, {len(set(night))} nights")

# ---- MJD -> GMST -> LST -> HA  (UT1~UTC; good to ~arcsec) ----
d = mjd - 51544.5                                  # days since J2000 (UT)
gmst = (18.697374558 + 24.06570982441908 * d) % 24.0 * 15.0   # deg
lst = (gmst + LON) % 360.0
HA = (lst - tra + 180) % 360 - 180                 # deg, [-180,180]
phi = np.radians(LAT); H = np.radians(HA); dec = np.radians(tdec)
alt = np.degrees(np.arcsin(np.sin(phi)*np.sin(dec) + np.cos(phi)*np.cos(dec)*np.cos(H)))
q = np.degrees(np.arctan2(np.sin(H), np.tan(phi)*np.cos(dec) - np.sin(dec)*np.cos(H)))  # parallactic
print(f"HA range {HA.min():.0f}..{HA.max():.0f} deg   alt {alt.min():.0f}..{alt.max():.0f} deg")

# ---- correlations of the (small) offset variation with geometry ----
def corr(a, b): return np.corrcoef(a, b)[0, 1]
print("\ncorrelations:")
for nm, x in [("HA", HA), ("sinHA", np.sin(H)), ("cosHA", np.cos(H)),
              ("alt", alt), ("cot(alt)", 1/np.tan(np.radians(alt))), ("q", q), ("Dec", tdec)]:
    print(f"  dRA vs {nm:8s} {corr(dRA,x):+.2f}    dDec vs {nm:8s} {corr(dDec,x):+.2f}")

# ---- equatorial pointing-model fit ----
secd, tand, sind, cosd = 1/np.cos(dec), np.tan(dec), np.sin(dec), np.cos(dec)
sH, cH = np.sin(H), np.cos(H)
# dRA (~ -dHA):  IH(const), CH sec, NP tan, ME sinH tan, MA cosH tan, TF cosphi sinH sec
Ara = np.c_[np.ones_like(dRA), secd, tand, sH*tand, cH*tand, np.cos(phi)*sH*secd]
# dDec: ID(const), ME cosH, MA sinH, TF tube-flexure
Adec = np.c_[np.ones_like(dDec), cH, sH, (np.cos(phi)*cH*sind - np.sin(phi)*cosd)]
cra = np.linalg.lstsq(Ara, dRA, rcond=None)[0]
cdc = np.linalg.lstsq(Adec, dDec, rcond=None)[0]
resRA = dRA - Ara@cra; resDec = dDec - Adec@cdc
print(f"\npointing-model fit:")
print(f"  dRA  RMS  {dRA.std()*3600:.0f}\" (const-only) -> {resRA.std()*3600:.0f}\" (HA/Dec model)")
print(f"  dDec RMS  {dDec.std()*3600:.0f}\" (const-only) -> {resDec.std()*3600:.0f}\" (HA/Dec model)")

# ---- does it absorb the night-to-night drift? ----
def between_night_std(x):
    return np.std([np.median(x[night == n]) for n in sorted(set(night))]) * 3600
print(f"\nbetween-night drift of medians:")
print(f"  dRA : raw {between_night_std(dRA):.0f}\"  ->  residual {between_night_std(resRA):.0f}\"")
print(f"  dDec: raw {between_night_std(dDec):.0f}\"  ->  residual {between_night_std(resDec):.0f}\"")

# ---- plots: offset vs HA and vs alt, colored by night ----
nn = sorted(set(night)); idx = {n: i for i, n in enumerate(nn)}; ni = np.array([idx[n] for n in night])
cmap = plt.cm.turbo; norm = BoundaryNorm(np.arange(-0.5, len(nn)+0.5, 1), cmap.N)
sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
fig, ax = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
for a, x, dy, xl, yl in [(ax[0,0], HA, dRA, "hour angle (deg)", "ΔRA (deg)"),
                          (ax[0,1], HA, dDec, "hour angle (deg)", "ΔDec (deg)"),
                          (ax[1,0], alt, dRA, "altitude (deg)", "ΔRA (deg)"),
                          (ax[1,1], alt, dDec, "altitude (deg)", "ΔDec (deg)")]:
    a.scatter(x, dy, c=ni, cmap=cmap, norm=norm, s=12, alpha=0.7, edgecolors="none")
    a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=0.25)
cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
ti = np.linspace(0, len(nn)-1, min(len(nn),14)).round().astype(int)
cb.set_ticks(ti); cb.set_ticklabels([nn[i] for i in ti]); cb.set_label("observing night")
fig.suptitle("FP-center offset vs hour angle & altitude (La Silla), colored by night", fontsize=13)
fig.savefig("fp_pointing_vs_HA_alt.png", dpi=140); print("\nwrote fp_pointing_vs_HA_alt.png")
