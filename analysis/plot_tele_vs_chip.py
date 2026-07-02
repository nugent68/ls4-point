#!/usr/bin/env python3
"""Compare telescope boresight vs SW_D chip-center coords.
Top row : TELE_RA vs chip_ra, TELE_DEC vs chip_dec (direct, with 1:1 line).
Bottom  : residual chip-tele vs tele  (where the offset is actually visible).
Colored by night."""
import csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

CSV = "/Users/nugent/claude/ls4/pointing/swd_i_centers.csv"
rows = list(csv.DictReader(open(CSV)))
night = np.array([r["image_name"][4:12] for r in rows])
nights = sorted(set(night)); idx = {n: i for i, n in enumerate(nights)}
ni = np.array([idx[n] for n in night]); N = len(nights)
tra = np.array([float(r["TELE_RA"]) for r in rows])
tdec = np.array([float(r["TELE_DEC"]) for r in rows])
cra = np.array([float(r["chip_ra"]) for r in rows])
cdec = np.array([float(r["chip_dec"]) for r in rows])
dra, ddec = cra - tra, cdec - tdec

cmap = plt.cm.turbo
norm = BoundaryNorm(np.arange(-0.5, N + 0.5, 1), cmap.N)
sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])

print(f"dRA  (chip-tele): median {np.median(dra):+.4f}  mean {dra.mean():+.4f}  std {dra.std():.4f} deg")
print(f"dDec (chip-tele): median {np.median(ddec):+.4f}  mean {ddec.mean():+.4f}  std {ddec.std():.4f} deg")

fig, ax = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

def sca(a, x, y, c=ni):
    return a.scatter(x, y, c=c, cmap=cmap, norm=norm, s=14, alpha=0.7, edgecolors="none")

# --- direct comparisons with 1:1 line ---
sca(ax[0, 0], tra, cra)
lo, hi = min(tra.min(), cra.min()), max(tra.max(), cra.max())
ax[0, 0].plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="1:1")
ax[0, 0].set_xlabel("TELE_RA (deg)"); ax[0, 0].set_ylabel("chip_ra (deg)")
ax[0, 0].set_title("RA: telescope vs chip center"); ax[0, 0].legend(loc="upper left")

sca(ax[0, 1], tdec, cdec)
lo, hi = min(tdec.min(), cdec.min()), max(tdec.max(), cdec.max())
ax[0, 1].plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="1:1")
ax[0, 1].set_xlabel("TELE_DEC (deg)"); ax[0, 1].set_ylabel("chip_dec (deg)")
ax[0, 1].set_title("Dec: telescope vs chip center"); ax[0, 1].legend(loc="upper left")

# --- residuals (chip - tele): the actual offset ---
sca(ax[1, 0], tra, dra)
ax[1, 0].axhline(np.median(dra), color="k", ls=":", lw=1,
                 label=f"median {np.median(dra):+.3f}")
ax[1, 0].set_xlabel("TELE_RA (deg)"); ax[1, 0].set_ylabel("chip_ra - TELE_RA (deg)")
ax[1, 0].set_title("RA offset vs pointing"); ax[1, 0].legend(loc="upper right"); ax[1, 0].grid(alpha=0.25)

sca(ax[1, 1], tdec, ddec)
ax[1, 1].axhline(np.median(ddec), color="k", ls=":", lw=1,
                 label=f"median {np.median(ddec):+.3f}")
ax[1, 1].set_xlabel("TELE_DEC (deg)"); ax[1, 1].set_ylabel("chip_dec - TELE_DEC (deg)")
ax[1, 1].set_title("Dec offset vs pointing"); ax[1, 1].legend(loc="upper right"); ax[1, 1].grid(alpha=0.25)

cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
ti = np.linspace(0, N - 1, min(N, 14)).round().astype(int)
cb.set_ticks(ti); cb.set_ticklabels([nights[i] for i in ti])
cb.set_label("observing night (YYYYMMDD)")
fig.suptitle(f"LS4 SW_D i-band: telescope boresight vs chip center — {len(rows)} frames",
             fontsize=14)
out = "/Users/nugent/claude/ls4/pointing/tele_vs_chip.png"
fig.savefig(out, dpi=140); print("wrote", out)
