#!/usr/bin/env python3
"""RA vs Dec scatter of LS4 SW_D i-band pointings, colored by night.
Two panels: telescope boresight (TELE_RA/DEC) and chip center (chip_ra/dec).
RA axis inverted (astronomical convention). Colorbar ticks = night dates."""
import csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

CSV = "/Users/nugent/claude/ls4/pointing/swd_i_centers.csv"
rows = list(csv.DictReader(open(CSV)))
night = np.array([r["image_name"][4:12] for r in rows])
nights = sorted(set(night))
idx = {n: i for i, n in enumerate(nights)}
ni = np.array([idx[n] for n in night])
N = len(nights)

cmap = plt.cm.turbo
norm = BoundaryNorm(np.arange(-0.5, N + 0.5, 1), cmap.N)

cols = {"TELE": ("TELE_RA", "TELE_DEC", "telescope boresight"),
        "chip": ("chip_ra", "chip_dec", "SW_D chip center (px 1024,2048)")}

fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), constrained_layout=True)
for ax, key in zip(axes, ("TELE", "chip")):
    rk, dk, title = cols[key]
    ra = np.array([float(r[rk]) for r in rows])
    dec = np.array([float(r[dk]) for r in rows])
    ax.scatter(ra, dec, c=ni, cmap=cmap, norm=norm, s=14, alpha=0.7,
               edgecolors="none")
    ax.set_xlabel(f"{rk}  (deg)"); ax.set_ylabel(f"{dk}  (deg)")
    ax.set_title(title, fontsize=11)
    ax.invert_xaxis()                      # RA increases to the left
    ax.grid(alpha=0.25)
    ax.margins(0.03)                       # no equal-aspect: 220deg RA vs 100deg Dec

sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01)
tick_i = np.linspace(0, N - 1, min(N, 14)).round().astype(int)
cb.set_ticks(tick_i)
cb.set_ticklabels([nights[i] for i in tick_i])
cb.set_label("observing night (YYYYMMDD)")
fig.suptitle(f"LS4 SW_D i-band pointings — {len(rows)} frames, {N} nights",
             fontsize=13)
out = "/Users/nugent/claude/ls4/pointing/pointing_radec_by_night.png"
fig.savefig(out, dpi=140)
print("wrote", out)

# --- second figure: zoom on the densest region (the (158,-28) survey field) ---
ra = np.array([float(r["chip_ra"]) for r in rows])
dec = np.array([float(r["chip_dec"]) for r in rows])
m = (ra > 150) & (ra < 167) & (dec > -33) & (dec < -22)
if m.sum() > 20:
    fig2, ax = plt.subplots(figsize=(8.4, 7.2), constrained_layout=True)
    ax.scatter(ra[m], dec[m], c=ni[m], cmap=cmap, norm=norm, s=24, alpha=0.8,
               edgecolors="none")
    ax.set_xlabel("chip_ra (deg)"); ax.set_ylabel("chip_dec (deg)")
    ax.invert_xaxis(); ax.grid(alpha=0.25); ax.set_aspect("equal", adjustable="datalim")
    cb2 = fig2.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cb2.set_ticks(tick_i); cb2.set_ticklabels([nights[i] for i in tick_i])
    cb2.set_label("observing night (YYYYMMDD)")
    ax.set_title(f"Zoom: main survey field (chip center) — {int(m.sum())} frames",
                 fontsize=12)
    out2 = "/Users/nugent/claude/ls4/pointing/pointing_radec_by_night_zoom.png"
    fig2.savefig(out2, dpi=140); print("wrote", out2)
else:
    print("zoom: too few points in 150<RA<167, -33<dec<-22 (", int(m.sum()), ")")
