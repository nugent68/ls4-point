#!/usr/bin/env python3
"""True on-sky offsets of the SW_D chip center from the telescope boresight,
computed with exact spherical geometry (astropy spherical_offsets_to /
separation / position_angle), not the flat cos(Dec) approximation.

  East/North = boresight.spherical_offsets_to(chip)   # true tangent-plane
  sep        = boresight.separation(chip)             # great-circle
  PA         = boresight.position_angle(chip)         # E of N
Colored by night.
"""
import csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from astropy.coordinates import SkyCoord
import astropy.units as u

CSV = "/Users/nugent/claude/ls4/pointing/swd_i_centers.csv"
rows = list(csv.DictReader(open(CSV)))
night = np.array([r["image_name"][4:12] for r in rows])
nights = sorted(set(night)); idx = {n: i for i, n in enumerate(nights)}
ni = np.array([idx[n] for n in night]); N = len(nights)
tra = np.array([float(r["TELE_RA"]) for r in rows])
tdec = np.array([float(r["TELE_DEC"]) for r in rows])
cra = np.array([float(r["chip_ra"]) for r in rows])
cdec = np.array([float(r["chip_dec"]) for r in rows])

bore = SkyCoord(tra * u.deg, tdec * u.deg)
chip = SkyCoord(cra * u.deg, cdec * u.deg)
east_a, north_a = bore.spherical_offsets_to(chip)
east = east_a.deg; north = north_a.deg
sep = bore.separation(chip).deg
pa = bore.position_angle(chip).deg

print(f"East  : median {np.median(east):+.4f}  std {east.std():.4f} deg")
print(f"North : median {np.median(north):+.4f}  std {north.std():.4f} deg")
print(f"|sep| : median {np.median(sep):.4f}  std {sep.std():.4f} deg")
print(f"PA    : median {np.median(pa):.2f}  std {pa.std():.2f} deg (E of N)")

cmap = plt.cm.turbo
norm = BoundaryNorm(np.arange(-0.5, N + 0.5, 1), cmap.N)
sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])


def sca(a, x, y):
    return a.scatter(x, y, c=ni, cmap=cmap, norm=norm, s=14, alpha=0.7, edgecolors="none")


fig, ax = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

sca(ax[0, 0], tdec, east)
ax[0, 0].axhline(np.median(east), color="k", ls=":", lw=1, label=f"median {np.median(east):+.3f}")
ax[0, 0].set_xlabel("TELE_DEC (deg)"); ax[0, 0].set_ylabel("East offset (deg)")
ax[0, 0].set_title("True East on-sky offset vs Dec"); ax[0, 0].legend(loc="upper right"); ax[0, 0].grid(alpha=0.25)

sca(ax[0, 1], tdec, north)
ax[0, 1].axhline(np.median(north), color="k", ls=":", lw=1, label=f"median {np.median(north):+.3f}")
ax[0, 1].set_xlabel("TELE_DEC (deg)"); ax[0, 1].set_ylabel("North offset (deg)")
ax[0, 1].set_title("True North on-sky offset vs Dec"); ax[0, 1].legend(loc="upper right"); ax[0, 1].grid(alpha=0.25)

sca(ax[1, 0], east, north)
th = np.linspace(0, 2 * np.pi, 200); R = np.median(sep)
ax[1, 0].plot(R * np.cos(th), R * np.sin(th), "k--", lw=1, alpha=0.6, label=f"|sep| median {R:.3f}°")
ax[1, 0].axhline(0, color="grey", lw=0.5); ax[1, 0].axvline(0, color="grey", lw=0.5)
ax[1, 0].set_xlabel("East offset (deg)"); ax[1, 0].set_ylabel("North offset (deg)")
ax[1, 0].set_title("On-sky offset vector (chip − boresight)")
ax[1, 0].set_aspect("equal", adjustable="box"); ax[1, 0].legend(loc="upper right"); ax[1, 0].grid(alpha=0.25)

sca(ax[1, 1], tdec, sep)
ax[1, 1].axhline(np.median(sep), color="k", ls=":", lw=1, label=f"median {np.median(sep):.3f}")
ax[1, 1].set_xlabel("TELE_DEC (deg)"); ax[1, 1].set_ylabel("great-circle separation (deg)")
ax[1, 1].set_title("Total on-sky separation vs Dec"); ax[1, 1].legend(loc="upper right"); ax[1, 1].grid(alpha=0.25)

cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
ti = np.linspace(0, N - 1, min(N, 14)).round().astype(int)
cb.set_ticks(ti); cb.set_ticklabels([nights[i] for i in ti])
cb.set_label("observing night (YYYYMMDD)")
fig.suptitle(f"LS4 SW_D i-band: true on-sky offset of chip center from boresight — {len(rows)} frames",
             fontsize=14)
out = "/Users/nugent/claude/ls4/pointing/onsky_offsets.png"
fig.savefig(out, dpi=140); print("wrote", out)
