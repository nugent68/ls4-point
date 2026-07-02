#!/usr/bin/env python3
"""Re-examine the FP-center pointing offset using only NORMAL-observing nights
(>= 20260514, i.e. after the May-13 TCS/dome realignment; the Apr29->May2 stow
experiment and the May-13 event are one-off interventions).

Recomputes per-night drift, HA/altitude relationships, the drift-vs-flexure
decomposition, and refits the true->TELE converter.  radecstats.csv, NE_A+SW_D.
"""
import csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

LAT, LON = -29.2563, -70.7397
CUT = "20260514"          # post May-13 realignment
GATE = 0.75

rows = list(csv.DictReader(open("radecstats.csv"))); ne, sw = {}, {}
for r in rows:
    c = r["chip"]
    if c not in ("NE_A", "SW_D"):
        continue
    p = r["image"].split("/"); k = (p[1], p[2])
    (ne if c == "NE_A" else sw)[k] = (float(r["exposure_ra"]), float(r["exposure_dec"]),
        float(r["wcs_image_ctr_ra"]), float(r["wcs_image_ctr_dec"]), float(r["mjd"]))

keys = sorted(k for k in (set(ne) & set(sw)) if k[0] >= CUT)
na = np.array([ne[k] for k in keys]); sd = np.array([sw[k] for k in keys])
night = np.array([k[0] for k in keys]); tra, tdec, mjd = na[:, 0], na[:, 1], na[:, 4]


def unit(ra, dec):
    r, d = np.radians(ra), np.radians(dec)
    return np.array([np.cos(d)*np.cos(r), np.cos(d)*np.sin(r), np.sin(d)])


v = (unit(na[:, 2], na[:, 3]) + unit(sd[:, 2], sd[:, 3])) / 2; v /= np.linalg.norm(v, axis=0)
fra = np.degrees(np.arctan2(v[1], v[0])) % 360.0; fdec = np.degrees(np.arcsin(v[2]))
dRA = (fra - tra + 180) % 360 - 180; dDec = fdec - tdec
g = (np.abs(dRA - np.median(dRA)) < GATE) & (np.abs(dDec - np.median(dDec)) < GATE)
tra, tdec, mjd, fra, fdec, dRA, dDec, night = (x[g] for x in (tra, tdec, mjd, fra, fdec, dRA, dDec, night))

d = mjd - 51544.5; gmst = (18.697374558 + 24.06570982441908*d) % 24 * 15; lst = (gmst + LON) % 360
HA = (lst - tra + 180) % 360 - 180; phi = np.radians(LAT); H = np.radians(HA); dec = np.radians(tdec)
alt = np.degrees(np.arcsin(np.sin(phi)*np.sin(dec) + np.cos(phi)*np.cos(dec)*np.cos(H)))

nn = sorted(set(night))
print(f"POST May-13: {g.sum()} exposures, {len(nn)} nights ({nn[0]}..{nn[-1]})")
print(f"medians: dRA {np.median(dRA):+.4f}  dDec {np.median(dDec):+.4f}")


def dm(x):
    o = x.copy()
    for n in set(night): o[night == n] -= np.median(x[night == n])
    return o


def bn(x): return np.std([np.median(x[night == n]) for n in nn]) * 3600
def wn(x): return dm(x).std() * 3600
print(f"total std : dRA {dRA.std()*3600:.0f}\"  dDec {dDec.std()*3600:.0f}\"")
print(f"between-night: dRA {bn(dRA):.0f}\"  dDec {bn(dDec):.0f}\"   (was 159\"/81\" over all post-shift)")
print(f"within-night : dRA {wn(dRA):.0f}\"  dDec {wn(dDec):.0f}\"")
print("\ncorrelations (post May-13):")
for nm, x in [("HA", HA), ("cosHA", np.cos(H)), ("alt", alt), ("cot(alt)", 1/np.tan(np.radians(alt)))]:
    print(f"  dRA vs {nm:8s} {np.corrcoef(dRA,x)[0,1]:+.2f}   dDec vs {nm:8s} {np.corrcoef(dDec,x)[0,1]:+.2f}")

# converter fit (bilinear in true position)
A = np.c_[np.ones_like(fra), fra, fdec]
cRA = np.linalg.lstsq(A, dRA, rcond=None)[0]; cDec = np.linalg.lstsq(A, dDec, rcond=None)[0]
print("\nconverter (post May-13):")
print("  delta_RA  = %.5f + %.6f*RA + %.6f*Dec" % tuple(cRA))
print("  delta_Dec = %.5f + %.6f*RA + %.6f*Dec" % tuple(cDec))
print("  bilinear RMS: RA %.0f\"  Dec %.0f\"" % (np.std(dRA-A@cRA)*3600, np.std(dDec-A@cDec)*3600))

# ---------- plots ----------
idx = {n: i for i, n in enumerate(nn)}; ni = np.array([idx[n] for n in night])
cmap = plt.cm.turbo; norm = BoundaryNorm(np.arange(-0.5, len(nn)+0.5, 1), cmap.N)
sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])

# per-night drift
fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
x = np.arange(len(nn))
ax[0].errorbar(x, [np.median(dRA[night==n]) for n in nn], yerr=[dRA[night==n].std() for n in nn], fmt='o-', color='tab:blue')
ax[0].set_ylabel("ΔRA (deg)"); ax[0].set_title(f"Post May-13 per-night FP-center offset ({g.sum()} exp)")
ax[1].errorbar(x, [np.median(dDec[night==n]) for n in nn], yerr=[dDec[night==n].std() for n in nn], fmt='s-', color='tab:red')
ax[1].set_ylabel("ΔDec (deg)"); ax[1].set_xticks(x); ax[1].set_xticklabels(nn, rotation=90, fontsize=7)
for a in ax: a.grid(alpha=0.3)
fig.savefig("fp_offset_per_night_postmay13.png", dpi=140); print("\nwrote fp_offset_per_night_postmay13.png")

# offset vs HA & alt
fig2, ax2 = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
for a, xx, yy, xl, yl in [(ax2[0,0], HA, dRA, "hour angle (deg)", "ΔRA (deg)"),
                           (ax2[0,1], HA, dDec, "hour angle (deg)", "ΔDec (deg)"),
                           (ax2[1,0], alt, dRA, "altitude (deg)", "ΔRA (deg)"),
                           (ax2[1,1], alt, dDec, "altitude (deg)", "ΔDec (deg)")]:
    a.scatter(xx, yy, c=ni, cmap=cmap, norm=norm, s=12, alpha=0.7, edgecolors="none")
    a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=0.25)
cb = fig2.colorbar(sm, ax=ax2, fraction=0.02, pad=0.01)
ti = np.linspace(0, len(nn)-1, min(len(nn),12)).round().astype(int)
cb.set_ticks(ti); cb.set_ticklabels([nn[i] for i in ti]); cb.set_label("night")
fig2.suptitle("Post May-13: FP-center offset vs hour angle & altitude", fontsize=13)
fig2.savefig("fp_pointing_vs_HA_alt_postmay13.png", dpi=140); print("wrote fp_pointing_vs_HA_alt_postmay13.png")
