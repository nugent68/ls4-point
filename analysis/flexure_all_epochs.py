#!/usr/bin/env python3
"""Cross-epoch flexure analysis + operational confirmation.

Questions:
 1. Are the HA-flexure coefficients (a1..a4, b1, b2) STABLE across the four
    engineering epochs?  (Fit each epoch with per-night zero points so the
    secular drift cannot contaminate the flexure terms.)
 2. If we solve only the offset (a0/b0) at the start of a night — as
    night_zero_point.py does — and apply a shared flexure model, how well do
    we predict the pointing for the rest of the night?

Data: radecstats.csv (PIYORP, 20260501-0626) + STDALN tables
(ne_a_i_centers.csv / swd_i_centers.csv) for the April pre-stow epoch.
"""
import csv, math, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAT, LON = -29.2563, -70.7397
GATE = 0.75

EPOCHS = [("E1 Apr10-29 (pre-stow)", "20260410", "20260429"),
          ("E2 May02-11", "20260502", "20260511"),
          ("E3 May14-27", "20260514", "20260527"),
          ("E4 Jun04-19", "20260604", "20260619")]


def unitv(ra, dec):
    r, d = np.radians(ra), np.radians(dec)
    return np.array([np.cos(d)*np.cos(r), np.cos(d)*np.sin(r), np.sin(d)])


def fp_offsets(tra, tdec, ra1, de1, ra2, de2):
    v = (unitv(ra1, de1) + unitv(ra2, de2)) / 2
    v /= np.linalg.norm(v, axis=0)
    fra = np.degrees(np.arctan2(v[1], v[0])) % 360
    fdec = np.degrees(np.arcsin(v[2]))
    return (fra - tra + 180) % 360 - 180, fdec - tdec


def mjd_from_name(night, hhmmss):
    """Night dir (= UT calendar date) + UT HHMMSS -> MJD."""
    y, m, d = int(night[:4]), int(night[4:6]), int(night[6:8])
    a = (14 - m)//12; yy = y + 4800 - a; mm = m + 12*a - 3
    jdn = d + (153*mm+2)//5 + 365*yy + yy//4 - yy//100 + yy//400 - 32045
    mjd0 = jdn - 2400001  # MJD at 0h UT of this calendar date
    h, mi, s = int(hhmmss[:2]), int(hhmmss[2:4]), int(hhmmss[4:6])
    return mjd0 + (h*3600 + mi*60 + s)/86400.0


def hour_angle(mjd, ra):
    d = mjd - 51544.5
    gmst = (18.697374558 + 24.06570982441908*d) % 24 * 15
    return ((gmst + LON) % 360 - ra + 180) % 360 - 180


# ---------- load radecstats (May-June) ----------
ne, sw = {}, {}
for r in csv.DictReader(open("radecstats.csv")):
    if r["chip"] not in ("NE_A", "SW_D"):
        continue
    p = r["image"].split("/"); k = (p[1], p[2])
    rec = (float(r["exposure_ra"]), float(r["exposure_dec"]),
           float(r["wcs_image_ctr_ra"]), float(r["wcs_image_ctr_dec"]), float(r["mjd"]))
    (ne if r["chip"] == "NE_A" else sw)[k] = rec
keys = sorted(set(ne) & set(sw))
na = np.array([ne[k] for k in keys]); sd = np.array([sw[k] for k in keys])
night_r = np.array([k[0] for k in keys])
dRA_r, dDec_r = fp_offsets(na[:, 0], na[:, 1], na[:, 2], na[:, 3], sd[:, 2], sd[:, 3])
data_r = dict(night=night_r, tra=na[:, 0], tdec=na[:, 1], mjd=na[:, 4],
              dRA=dRA_r, dDec=dDec_r)

# ---------- load STDALN tables (April only; also used for MJD-rule check) ----
def load_stdaln(fn):
    d = {}
    for r in csv.DictReader(open(fn)):
        p = r["image_name"].split("_")          # ls4 night hhmmss CHIP...
        d[(p[1], p[2])] = (float(r["TELE_RA"]), float(r["TELE_DEC"]),
                           float(r["chip_ra"]), float(r["chip_dec"]))
    return d


sa, ss = load_stdaln("ne_a_i_centers.csv"), load_stdaln("swd_i_centers.csv")
keys_s = sorted(k for k in (set(sa) & set(ss)) if k[0] < "20260501")
aa = np.array([sa[k] for k in keys_s]); bb = np.array([ss[k] for k in keys_s])
night_s = np.array([k[0] for k in keys_s])
mjd_s = np.array([mjd_from_name(k[0], k[1]) for k in keys_s])
dRA_s, dDec_s = fp_offsets(aa[:, 0], aa[:, 1], aa[:, 2], aa[:, 3], bb[:, 2], bb[:, 3])
data_s = dict(night=night_s, tra=aa[:, 0], tdec=aa[:, 1], mjd=mjd_s,
              dRA=dRA_s, dDec=dDec_s)

# validate the name->MJD rule against radecstats where both datasets overlap
kk = [k for k in (set(sa) & set(ss)) if k in ne]
if kk:
    err = [abs(mjd_from_name(k[0], k[1]) - ne[k][4])*86400 for k in kk[:500]]
    print(f"MJD-from-name check vs radecstats: median |err| {np.median(err):.1f}s "
          f"max {np.max(err):.1f}s over {len(err)} exposures")

# ---------- combine + clean ----------
night = np.concatenate([data_s["night"], data_r["night"]])
tra = np.concatenate([data_s["tra"], data_r["tra"]])
tdec = np.concatenate([data_s["tdec"], data_r["tdec"]])
mjd = np.concatenate([data_s["mjd"], data_r["mjd"]])
dRA = np.concatenate([data_s["dRA"], data_r["dRA"]])
dDec = np.concatenate([data_s["dDec"], data_r["dDec"]])
# per-era robust gate (era medians differ hugely)
good = np.zeros(len(night), bool)
for _, n0, n1 in [("all-april", "20260101", "20260430"), ("post", "20260501", "20261231")]:
    m = (night >= n0) & (night <= n1)
    if m.sum():
        g = (np.abs(dRA[m] - np.median(dRA[m])) < GATE) & \
            (np.abs(dDec[m] - np.median(dDec[m])) < GATE)
        idx = np.where(m)[0]; good[idx[g]] = True
night, tra, tdec, mjd, dRA, dDec = (x[good] for x in (night, tra, tdec, mjd, dRA, dDec))
pnight = np.floor(mjd - 0.5).astype(int)   # physical observing night (noon-shifted)
HA = hour_angle(mjd, tra)
H = np.radians(HA); dec = np.radians(tdec)
sH, cH, tD = np.sin(H), np.cos(H), np.tan(dec)
FLEX_RA = np.c_[sH, cH, sH*tD, cH*tD]
FLEX_DE = np.c_[sH, cH]
print(f"combined: {len(night)} exposures, {len(set(night))} nights "
      f"({min(night)}..{max(night)}), dropped {int((~good).sum())} outliers")


def fit_nightzp(mask, X, y, clip=4.0, niter=3):
    """LSQ with per-night zero points + shared columns X; returns
    (coeffs, errs, resid, sigma_robust)."""
    nts = sorted(set(pnight[mask]))
    ni = {n: i for i, n in enumerate(nts)}
    D = np.zeros((mask.sum(), len(nts)))
    for j, n in enumerate(pnight[mask]):
        D[j, ni[n]] = 1.0
    A = np.hstack([D, X[mask]])
    yy = y[mask]; w = np.ones(len(yy), bool)
    for _ in range(niter):
        c, *_ = np.linalg.lstsq(A[w], yy[w], rcond=None)
        r = yy - A @ c
        s = 1.4826*np.median(np.abs(r[w] - np.median(r[w])))
        w = np.abs(r) < clip*max(s, 1e-6)
    cov = np.linalg.pinv(A[w].T @ A[w]) * s**2
    return c[len(nts):], np.sqrt(np.abs(np.diag(cov)))[len(nts):], r, s


# ---------- 1. per-epoch flexure coefficients ----------
print("\n=== per-epoch flexure fits (per-night ZPs; coeff ± err, deg) ===")
print(f"{'epoch':26s} {'N':>5} | a1(sinH)      a2(cosH)      a3(sHtanD)    a4(cHtanD)   | b1(sinH)      b2(cosH)")
tab = {}
for lab, n0, n1 in EPOCHS:
    m = (night >= n0) & (night <= n1)
    if m.sum() < 50:
        print(f"{lab:26s} {m.sum():5d} | too few"); continue
    cR, eR, _, sR = fit_nightzp(m, FLEX_RA, dRA)
    cD, eD, _, sD = fit_nightzp(m, FLEX_DE, dDec)
    tab[lab] = (cR, eR, cD, eD, sR, sD, int(m.sum()))
    fR = " ".join(f"{c:+.4f}±{e:.4f}" for c, e in zip(cR, eR))
    fD = " ".join(f"{c:+.4f}±{e:.4f}" for c, e in zip(cD, eD))
    print(f"{lab:26s} {m.sum():5d} | {fR} | {fD}")

# ---------- 2. global shared-flexure fit ----------
allm = np.ones(len(night), bool)
cRg, eRg, rRg, sRg = fit_nightzp(allm, FLEX_RA, dRA)
cDg, eDg, rDg, sDg = fit_nightzp(allm, FLEX_DE, dDec)
print("\n=== GLOBAL shared flexure (per-night ZPs, all epochs) ===")
print("  dRA :  " + " ".join(f"{c:+.5f}±{e:.5f}" for c, e in zip(cRg, eRg)))
print("  dDec:  " + " ".join(f"{c:+.5f}±{e:.5f}" for c, e in zip(cDg, eDg)))
print(f"  robust per-exposure resid: RA {sRg*3600:.1f}\"  Dec {sDg*3600:.1f}\"")
print("  (June-only fit was: a1..a4 = -0.02062 -0.01702 +0.03026 -0.00848 ; "
      "b1,b2 = -0.00165 -0.04661)")

# ---------- 3. OPERATIONAL SIMULATION ----------
# per night: a0/b0 from the first NCAL exposures (flexure-corrected with the
# GLOBAL coefficients), then predict every later exposure of that night.
NCAL = 3
resid_ra, resid_de, res_night = [], [], []
for n in sorted(set(pnight)):
    m = pnight == n
    if m.sum() < NCAL + 3:
        continue
    idx = np.where(m)[0][np.argsort(mjd[m])]
    flex_ra = FLEX_RA[idx] @ cRg; flex_de = FLEX_DE[idx] @ cDg
    a0 = np.median(dRA[idx[:NCAL]] - flex_ra[:NCAL])
    b0 = np.median(dDec[idx[:NCAL]] - flex_de[:NCAL])
    pr = dRA[idx[NCAL:]] - (a0 + flex_ra[NCAL:])
    pd = dDec[idx[NCAL:]] - (b0 + flex_de[NCAL:])
    resid_ra.append(pr * np.cos(dec[idx[NCAL:]]))   # on-sky
    resid_de.append(pd)
    res_night.append((n, len(pr), np.median(np.abs(pr*np.cos(dec[idx[NCAL:]])))*3600,
                      np.median(np.abs(pd))*3600))
rr = np.concatenate(resid_ra)*3600; rd = np.concatenate(resid_de)*3600
print(f"\n=== OPERATIONAL SIM: a0/b0 from first {NCAL} exposures/night, "
      f"global flexure applied to the rest ===")
print(f"  {len(rr)} predicted exposures over {len(res_night)} nights")
for p in (50, 68, 90, 95):
    print(f"  |resid| p{p}: RA(on-sky) {np.percentile(np.abs(rr),p):6.1f}\"   "
          f"Dec {np.percentile(np.abs(rd),p):6.1f}\"")
worst = sorted(res_night, key=lambda t: -max(t[2], t[3]))[:5]
print("  worst nights (median |resid| RA\"/Dec\"): "
      + ", ".join(f"{n}({r:.0f}/{d:.0f})" for n, c, r, d in worst))

# same sim but WITHOUT flexure (constant-only start-of-night offset)
resid_ra0, resid_de0 = [], []
for n in sorted(set(pnight)):
    m = pnight == n
    if m.sum() < NCAL + 3:
        continue
    idx = np.where(m)[0][np.argsort(mjd[m])]
    a0 = np.median(dRA[idx[:NCAL]]); b0 = np.median(dDec[idx[:NCAL]])
    resid_ra0.append((dRA[idx[NCAL:]] - a0) * np.cos(dec[idx[NCAL:]]))
    resid_de0.append(dDec[idx[NCAL:]] - b0)
rr0 = np.concatenate(resid_ra0)*3600; rd0 = np.concatenate(resid_de0)*3600
print(f"  [no-flexure baseline] p68: RA {np.percentile(np.abs(rr0),68):.1f}\" "
      f" Dec {np.percentile(np.abs(rd0),68):.1f}\"   "
      f"p95: RA {np.percentile(np.abs(rr0),95):.1f}\"  Dec {np.percentile(np.abs(rd0),95):.1f}\"")

# ---------- plots ----------
fig, ax = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
cols = dict(zip([e[0] for e in EPOCHS], ["tab:purple", "tab:blue", "tab:green", "tab:red"]))
# detrended (per-night ZP removed) vs HA + global model
def detrend(y, Xg, cg):
    out = y.copy()
    for n in set(pnight):
        m = pnight == n
        out[m] -= np.median(y[m] - Xg[m] @ cg)
    return out
dtR = detrend(dRA, FLEX_RA, cRg); dtD = detrend(dDec, FLEX_DE, cDg)
hgrid = np.linspace(-65, 65, 200); Hg = np.radians(hgrid)
for a, dt, cg, isra in ((ax[0, 0], dtR, cRg, True), (ax[0, 1], dtD, cDg, False)):
    for lab, n0, n1 in EPOCHS:
        m = (night >= n0) & (night <= n1)
        a.scatter(HA[m], dt[m]*3600, s=6, alpha=0.4, color=cols[lab], label=lab, edgecolors="none")
    md = np.median(tdec)
    if isra:
        mdl = (np.sin(Hg)*cg[0] + np.cos(Hg)*cg[1]
               + np.sin(Hg)*math.tan(math.radians(md))*cg[2]
               + np.cos(Hg)*math.tan(math.radians(md))*cg[3])
        a.set_ylabel('night-detrended ΔRA (")')
    else:
        mdl = np.sin(Hg)*cg[0] + np.cos(Hg)*cg[1]
        a.set_ylabel('night-detrended ΔDec (")')
    a.plot(hgrid, mdl*3600, "k-", lw=2, label=f"global model (Dec={md:.0f}°)")
    a.set_xlabel("hour angle (deg)"); a.grid(alpha=0.25); a.legend(fontsize=7)
    a.set_ylim(-250, 250)
ax[0, 0].set_title("flexure signal, all epochs (per-night ZP removed)")
ax[0, 1].set_title("same for Dec")
# operational residual histograms
ax[1, 0].hist(rr, bins=np.linspace(-120, 120, 81), alpha=0.6, label=f"with flexure p68={np.percentile(np.abs(rr),68):.0f}\"")
ax[1, 0].hist(rr0, bins=np.linspace(-120, 120, 81), histtype="step", color="k", label=f"const-only p68={np.percentile(np.abs(rr0),68):.0f}\"")
ax[1, 0].set_xlabel('RA prediction residual on-sky (")'); ax[1, 0].legend(fontsize=8)
ax[1, 1].hist(rd, bins=np.linspace(-120, 120, 81), alpha=0.6, color="tab:red", label=f"with flexure p68={np.percentile(np.abs(rd),68):.0f}\"")
ax[1, 1].hist(rd0, bins=np.linspace(-120, 120, 81), histtype="step", color="k", label=f"const-only p68={np.percentile(np.abs(rd0),68):.0f}\"")
ax[1, 1].set_xlabel('Dec prediction residual (")'); ax[1, 1].legend(fontsize=8)
ax[1, 0].set_title(f"operational sim: solve a0/b0 from first {NCAL} exposures, predict rest of night")
for a in ax[1]: a.grid(alpha=0.25)
fig.suptitle("LS4 flexure across ALL epochs + start-of-night operational confirmation", fontsize=13)
fig.savefig("flexure_all_epochs.png", dpi=140)
print("\nwrote flexure_all_epochs.png")
print("\nGLOBAL coefficients for implementation (deg):")
print(f"  A_RA  = (a0_nightly, {cRg[0]:+.5f}, {cRg[1]:+.5f}, {cRg[2]:+.5f}, {cRg[3]:+.5f})")
print(f"  A_DEC = (b0_nightly, {cDg[0]:+.5f}, {cDg[1]:+.5f})")
