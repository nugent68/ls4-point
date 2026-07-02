#!/usr/bin/env python3
"""Convert a desired TRUE on-sky position (where you want the LS4 focal-plane
CENTER to land) into the TELE-RA / TELE-DEC to request.

Calibration: FP center = cos-Dec-corrected mean of the NE_A and SW_D i-band
chip centers (the chips bracketing the FP center), from radecstats.csv.
  delta_RA  = FP_RA  - TELE_RA
  delta_Dec = FP_Dec - TELE_DEC

The offset is CONSTANT PER ENGINEERING EPOCH (it re-zeroes after stow/TCS/dome
interventions: 2026-04-30..05-01 stow experiment, ~05-13 TCS/dome realign, and
one more step in the 05-28..06-04 gap), plus a repeatable HA flexure.

CURRENT EPOCH: June 4+ 2026 (523 exposures, 8 nights, 20260604..20260619):
  dRA  = a0 + a1 sinH + a2 cosH + a3 sinH tanD + a4 cosH tanD
  dDec = b0 + b1 sinH + b2 cosH
H = hour angle at La Silla, D = Dec.  cot(alt) terms tested — add nothing.
Scatter (robust 68%): ~8.6" on-sky RA, ~11.7" Dec with the HA terms;
~24"/30" with the constant only.  REFIT a0/b0 after any engineering event.

CLI:  true_to_tele.py <true_RA_deg> <true_Dec_deg> [MJD]
      (with MJD -> full HA model; without -> constant-offset epoch values)
"""
import sys
import math

LAT, LON = -29.2563, -70.7397          # La Silla

# ---- flexure terms: GLOBAL cross-epoch fit (Apr 10 - Jun 19 2026, 4610 exp,
# per-night zero points; see pointing/flexure_all_epochs.py).  The flexure is
# a property of the telescope: all four engineering epochs (incl. pre-stow)
# lie on the same curve.  Only a0/b0 re-zero after engineering events.
# a0/b0 below = June-2026 epoch, refit against these flexure terms.
A_JUNE_RA = (-1.92916, 0.00078, -0.01361, 0.03608, -0.01553)
A_JUNE_DEC = (-0.45011, 0.00083, -0.04654)
CONST_JUNE_RA, CONST_JUNE_DEC = -1.9472, -0.4839    # epoch medians (no HA term)
# previous June-only fit (single epoch ZP, superseded):
#   (-1.91220, -0.02062, -0.01702, 0.03026, -0.00848) / (-0.44861, -0.00165, -0.04661)


def _ha_deg(mjd, tele_ra):
    d = mjd - 51544.5
    gmst = (18.697374558 + 24.06570982441908 * d) % 24.0 * 15.0
    lst = (gmst + LON) % 360.0
    return (lst - tele_ra + 180.0) % 360.0 - 180.0


def delta_june(ha_deg, dec_deg):
    """(dRA, dDec) in deg for the June-4+ epoch at hour angle / Dec."""
    H = math.radians(ha_deg); tD = math.tan(math.radians(dec_deg))
    sH, cH = math.sin(H), math.cos(H)
    a, b = A_JUNE_RA, A_JUNE_DEC
    dra = a[0] + a[1]*sH + a[2]*cH + a[3]*sH*tD + a[4]*cH*tD
    ddec = b[0] + b[1]*sH + b[2]*cH
    return dra, ddec


def true_to_tele(ra, dec, mjd=None):
    """Desired true FP-center (deg) -> TELE-RA/DEC to request.
    With mjd: full HA-flexure model (~9"/12" robust).
    Without : constant epoch offset (~24"/30" robust)."""
    if mjd is None:
        return (ra - CONST_JUNE_RA) % 360.0, dec - CONST_JUNE_DEC
    tra, tdec = ra, dec
    for _ in range(2):                       # fixed-point on H (dRA ~ -1.9 deg)
        dra, ddec = delta_june(_ha_deg(mjd, tra), tdec)
        tra, tdec = (ra - dra) % 360.0, dec - ddec
    return tra, tdec


def tele_to_true(tele_ra, tele_dec, mjd):
    """Where the FP center actually lands for a commanded TELE pointing."""
    dra, ddec = delta_june(_ha_deg(mjd, tele_ra), tele_dec)
    return (tele_ra + dra) % 360.0, tele_dec + ddec


# back-compat aliases
true_to_tele_june = true_to_tele
tele_to_true_june = tele_to_true


def true_to_tele_constant(ra, dec):
    """Constant-offset form, June 4+ epoch."""
    return true_to_tele(ra, dec, mjd=None)


# ---- legacy (whole post-shift era >=20260502; superseded — kept for record) —
# bilinear fit, RMS 149"(RA)/73"(Dec): mixes three engineering epochs.
A_RA_LEGACY = (-1.95664, 0.000361, 0.000750)     # const, *RA, *Dec
A_DEC_LEGACY = (-0.48109, -0.000165, -0.000495)


if __name__ == "__main__":
    ra, dec = float(sys.argv[1]), float(sys.argv[2])
    mjd = float(sys.argv[3]) if len(sys.argv) > 3 else None
    tra, tdec = true_to_tele(ra, dec, mjd)
    print(f"true (FP-center)  RA={ra:.5f}  Dec={dec:.5f}"
          + (f"  MJD={mjd:.5f}" if mjd else ""))
    if mjd is None:
        print(f"request TELE-RA={tra:.5f}  TELE-DEC={tdec:.5f}   "
              f"(constant offset; ~24\"/30\" 1-sigma. Pass MJD for HA model)")
    else:
        print(f"request TELE-RA={tra:.5f}  TELE-DEC={tdec:.5f}   "
              f"(HA model; ~9\"/12\" 1-sigma)")
        cra, cdec = true_to_tele(ra, dec)
        print(f"(constant-offset would give: {cra:.5f}, {cdec:.5f})")
