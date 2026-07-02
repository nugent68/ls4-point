# ls4-point — LS4 start-of-night pointing zero-point tool

Measures the LS4 focal-plane-center ↔ telescope pointing offset zero points
(**a₀/b₀**) directly from **raw** LS4 exposures, so the pointing model can be
re-zeroed at the telescope at the start of a night (required after any
stow-position / TCS / dome-alignment intervention — these re-zero the offset;
the hour-angle flexure terms carry over).

Per raw exposure (single MEF, 32 chip HDUs) the tool:
1. extracts the **NE_A** and **SW_D** chips (the two i-band chips that
   bracket the focal-plane center),
2. overscan-subtracts (optional superbias/flat/mask in `calib/`),
3. solves each chip with astrometry.net `solve-field` (internal source
   detector — no source-extractor, no netpbm needed),
4. focal-plane center = unit-vector mean of the two chip centers
   (chip pixel 1024, 2048),
5. offset = FP center − TELE pointing; subtracts the HA-flexure model →
   **a₀, b₀** per exposure and the mean over exposures.

Feed it 2–3 exposures from the night (different HAs help) and update the two
constants in `true_to_tele.py` (`A_JUNE_RA[0]`, `A_JUNE_DEC[0]`).

```
python night_zero_point.py raw1.fits.fz raw2.fits.fz --no-calib --json out.json
```

Typical runtime ~5 s per chip; expected accuracy ~10–20″ per exposure
(validated against 3 epochs of survey data — see `validation/`).

---

## Installation on a generic Linux box

Python needs: `numpy`, `astropy` (e.g. `pip install numpy astropy` or
`apt install python3-numpy python3-astropy`).

### Option A — package manager (simplest)

```bash
sudo apt install astrometry.net          # Debian/Ubuntu
# (Fedora/EL: dnf install astrometry.net)
```

This puts `solve-field` on `$PATH`; the tool finds it automatically (no
`--astrom-dir` needed). Index files still must be installed by hand — see
**Index files** below; either drop them into the directory named in
`/etc/astrometry.cfg` (usually `/usr/share/astrometry`) or pass
`--index-cfg your.cfg`.

### Option B — build from source (what we run at NERSC; no root needed)

Prereqs: `gcc`, `make`, `pkg-config`, `zlib` headers (`zlib1g-dev`).

```bash
AP=/opt/ls4-astrometry            # install prefix — anywhere you like
mkdir -p $AP/src && cd $AP/src

# 1. cfitsio
curl -LO https://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio_latest.tar.gz
tar xzf cfitsio_latest.tar.gz && cd cfitsio-*/
./configure --prefix=$AP && make -j8 && make install
cd $AP/src

# 2. astrometry.net (release tarball)
curl -LO https://github.com/dstndstn/astrometry.net/releases/download/0.97/astrometry.net-0.97.tar.gz
tar xzf astrometry.net-0.97.tar.gz && cd astrometry.net-0.97
export PKG_CONFIG_PATH=$AP/lib/pkgconfig:$PKG_CONFIG_PATH
make && make install INSTALL_DIR=$AP
```

Notes:
- **netpbm is NOT required** — the tool passes `--fits-image` so solve-field
  never touches the pnm path.
- **source-extractor is NOT required** — the internal simplexy detector is
  used (and validated to be sufficient).
- cairo/plotting libs are optional; build warnings about them are harmless.
- Runtime linking: the tool sets `LD_LIBRARY_PATH=$AP/lib` automatically when
  you pass `--astrom-dir $AP`.

Then run with `--astrom-dir $AP` (expects `bin/solve-field`, `astrometry.cfg`,
`lib/`, and optionally `calib/` under it).

## Index files (not in this repo — too big)

Use the Gaia-EDR3 **5200-series** skymarks. On NERSC they live in
`/global/cfs/cdirs/cosmo/work/users/dstn/index-5200/` (also downloadable from
http://data.astrometry.net/5200/). For the LS4 chip field of view (34′×68′)
only the large scales are needed — all of the following sets were validated
to give **identical** solutions on LS4 raws; pick by available disk:

| set | files | size | note |
|---|---|---|---|
| recommended | `index-5204-*.fits  index-5205-*.fits  index-5206-*.fits` | 4.8 GB | best margin |
| tight | `index-5205-*.fits  index-5206-*.fits` | 2.1 GB | validated |
| minimum | `index-5206-*.fits` | 0.7 GB | validated, least margin |

```bash
# from NERSC:
rsync -av 'perlmutter.nersc.gov:/global/cfs/cdirs/cosmo/work/users/dstn/index-5200/index-520[456]-*.fits' $AP/indices/
```

Do **not** bother with the 5000 series: it is Gaia-DR2, 62 GB, and only
contains small-scale skymarks unsuited to this field of view.

Config file (`$AP/astrometry.cfg`) — note `autoindex`, required for the
0.97 release (`indexset` is newer):

```
add_path /opt/ls4-astrometry/indices
autoindex
inparallel
```

## Calibration files (`calib/`)

`superbias_<CHIP>.fits.fz`, `LS4Cam_flat_20260421_<CHIP>.fits`,
`FP_mask_<CHIP>.fits.fz` for NE_A and SW_D. **Optional**: they change the
solved chip centers by ≤0.25″ (negligible vs the ~10″ model scatter) and
roughly double the runtime, so `--no-calib` (overscan-only) is the
recommended operating mode. They are included for completeness/experiments.

## The pointing model (`true_to_tele.py`)

Flexure terms are a **global cross-epoch fit** (Apr 10 – Jun 19 2026, 4610
exposures, per-night zero points; `analysis/flexure_all_epochs.py`). All four
engineering epochs — including pre-stow — lie on the same flexure curve
(`validation/flexure_all_epochs.png`), confirming that **only a₀/b₀ re-zero
after engineering events**:

```
a1..a4 = +0.00078  -0.01361  +0.03608  -0.01553   (dRA terms, deg)
b1,b2  = +0.00083  -0.04654                        (dDec terms, deg)
```

**Operational confirmation:** solving a₀/b₀ from just the first 3 exposures
of each night and applying this shared flexure model predicts the pointing
for the rest of the night to **8″ (RA, on-sky) / 13″ (Dec) at p68** and
20″/27″ at p95 — versus 39″/28″ (p68) with a constant offset only.

```
dRA  = a0 + a1·sinH + a2·cosH + a3·sinH·tanδ + a4·cosH·tanδ
dDec = b0 + b1·sinH + b2·cosH
TELE_RA = true_RA − dRA ,  TELE_DEC = true_Dec − dDec
```

`true_to_tele(ra, dec, mjd)` converts a desired true FP-center position into
the TELE-RA/DEC to request; `tele_to_true()` is the inverse. Only **a₀/b₀**
change after engineering events; refit them with `night_zero_point.py`.

## Repo layout

```
night_zero_point.py   the start-of-night tool (standalone)
true_to_tele.py       pointing model + converters (a0/b0 live here)
calib/                per-chip superbias / flat / FP mask (optional)
analysis/             scripts that derived the model (offset tables, HA/alt
                      flexure fits, per-night drift)
validation/           solver A/B results on 3 raws spanning all 3
                      engineering epochs (a0 reproduced to 1.8″)
```

## Validation summary (2026-06-29/07-02, NERSC)

Three raw exposures spanning all three engineering epochs:

| raw | epoch | measured ΔRA/ΔDec | survey per-night value |
|---|---|---|---|
| 20260410 | pre-stow | −1.089 / +0.243 | −1.10 / +0.25 |
| 20260503 | May era | −1.988 / −0.462 | −1.988 / −0.462 (exact) |
| 20260604 | June | a₀/b₀ = −1.9117 / −0.4528 | model −1.9122 / −0.4486 |

Index sets 9.8 GB → 0.7 GB: identical centers (0.00″). Calib vs no-calib:
≤0.24″. Internal detector solves 6/6 chips, ~5 s each.
