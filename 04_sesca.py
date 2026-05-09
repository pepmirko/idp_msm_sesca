"""
04_sesca.py
===========
Step 4 — SESCA CD forward modelling + D-SPR ensemble validation.

Inputs  : analysis/sesca/macro_N_cd.dat   (SESCA output, one per macrostate)
          data/abeta42_cd_exp.dat          (experimental CD spectrum)
          analysis/msm.pyemma
Outputs : analysis/plots/cd_comparison.png
          analysis/dspr_value.txt

Run SESCA externally before this script:
    for i in 0 1 2 3 4; do
        python /path/to/sesca_main.py \
            -pdb analysis/representatives/macro_${i}.pdb \
            -basis DB5 \
            -output analysis/sesca/macro_${i}_cd.dat
    done
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pyemma
import os
import sys

os.makedirs("analysis/plots", exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────

EXP_DATA_PATH = "data/abeta42_cd_exp.dat"   # two-column: wavelength, signal

# ── Load MSM populations ──────────────────────────────────────────────────────

print("=" * 60)
print("STEP 4 — SESCA CD forward modelling")
print("=" * 60)
print()

msm    = pyemma.load("analysis/msm.pyemma")
N_MACRO = msm.n_metastable

populations = np.array([
    msm.pi[msm.metastable_assignments == i].sum()
    for i in range(N_MACRO)
])
populations /= populations.sum()

print(f"Macrostate populations:")
for i, p in enumerate(populations):
    print(f"  Macro {i}: {p:.4f}")

# ── Load SESCA spectra ────────────────────────────────────────────────────────

print(f"\nLoading SESCA spectra for {N_MACRO} macrostates...")

wavelengths = None
spectra     = []
missing     = []

for i in range(N_MACRO):
    path = f"analysis/sesca/macro_{i}_cd.dat"
    if not os.path.exists(path):
        missing.append(i)
        continue
    data = np.loadtxt(path, comments="#")
    if wavelengths is None:
        wavelengths = data[:, 0]
    spectra.append((i, data[:, 1]))
    print(f"  Loaded macro_{i}_cd.dat  ({len(data)} points)")

if missing:
    print(f"\n⚠ Missing SESCA output for macrostates: {missing}")
    print("  Run SESCA on the corresponding PDBs first (see script header).")
    sys.exit(1)

spectra_array = np.array([s for _, s in spectra])     # [N_MACRO, n_wl]
indices       = [i for i, _ in spectra]

# ── Population-weighted ensemble spectrum ─────────────────────────────────────

cd_ensemble = np.sum(populations[indices, None] * spectra_array, axis=0)

# ── Load experimental spectrum ────────────────────────────────────────────────

if not os.path.exists(EXP_DATA_PATH):
    print(f"\n⚠ Experimental data not found at {EXP_DATA_PATH}")
    print("  Place your experimental CD file there (2 columns: λ, signal)")
    print("  Proceeding with ensemble spectrum only — D-SPR will be skipped.")
    exp_available = False
else:
    cd_exp = np.loadtxt(EXP_DATA_PATH, comments="#")
    wl_exp = cd_exp[:, 0]
    signal_exp = cd_exp[:, 1]
    exp_available = True
    print(f"\nExperimental data: {len(cd_exp)} points, "
          f"λ = [{wl_exp.min():.0f}, {wl_exp.max():.0f}] nm")

# ── D-SPR metric ──────────────────────────────────────────────────────────────

def d_spr(pred, exp):
    """
    Normalised spectral RMSD (D-SPR).
    D-SPR = RMSD(pred, exp) / RMS(exp)
    0 = perfect match; < 0.25 = good.
    """
    residuals = pred - exp
    rmsd  = np.sqrt(np.mean(residuals ** 2))
    scale = np.sqrt(np.mean(exp ** 2))
    return rmsd / (scale + 1e-12)

dspr_value = None

if exp_available:
    # Interpolate ensemble onto experimental wavelength grid
    f_interp = interp1d(wavelengths, cd_ensemble, kind="cubic",
                        fill_value="extrapolate", bounds_error=False)
    cd_pred  = f_interp(wl_exp)

    dspr_value = d_spr(cd_pred, signal_exp)

    quality = (
        "Excellent" if dspr_value < 0.15 else
        "Good"      if dspr_value < 0.25 else
        "Moderate"  if dspr_value < 0.40 else
        "Poor"
    )
    print(f"\nD-SPR = {dspr_value:.4f}  ({quality})")

    with open("analysis/dspr_value.txt", "w") as fh:
        fh.write(f"D-SPR = {dspr_value:.6f}\n")
        fh.write(f"Quality: {quality}\n")
        fh.write(f"N_MACRO = {N_MACRO}\n")
        for i, p in enumerate(populations):
            fh.write(f"  Macro {i}: pop = {p:.4f}\n")

# ── Plot ──────────────────────────────────────────────────────────────────────

COLORS_MACRO = plt.cm.tab10(np.linspace(0, 0.9, N_MACRO))

fig, ax = plt.subplots(figsize=(9, 6))

# Individual macrostate spectra (thin, transparent)
for idx, (i, sp) in enumerate(zip(indices, spectra_array)):
    f_i = interp1d(wavelengths, sp, kind="cubic",
                   fill_value="extrapolate", bounds_error=False)
    wl_plot = wl_exp if exp_available else wavelengths
    ax.plot(wl_plot, f_i(wl_plot), lw=1.2, alpha=0.4,
            color=COLORS_MACRO[idx], label=f"Macro {i} ({populations[i]:.2f})")

# Ensemble prediction
wl_plot = wl_exp if exp_available else wavelengths
f_ens   = interp1d(wavelengths, cd_ensemble, kind="cubic",
                   fill_value="extrapolate", bounds_error=False)
label_ens = (f"Ensemble (D-SPR = {dspr_value:.3f})" if dspr_value is not None
             else "Ensemble prediction")
ax.plot(wl_plot, f_ens(wl_plot), lw=2.5, color="crimson",
        ls="--", label=label_ens, zorder=5)

# Experimental
if exp_available:
    ax.plot(wl_exp, signal_exp, lw=2.5, color="black",
            label="Experimental (lit.)", zorder=6)

ax.axhline(0, color="grey", lw=0.6)
ax.set_xlabel("Wavelength (nm)", fontsize=12)
ax.set_ylabel("ΔΕ  (M⁻¹ cm⁻¹)  or  [θ]", fontsize=12)
ax.set_title("SESCA CD Forward Modelling — Aβ(1-42)", fontsize=13)
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig("analysis/plots/cd_comparison.png", dpi=150)
print("Saved: analysis/plots/cd_comparison.png")

# ── Secondary structure summary from SESCA (if available) ─────────────────────
# SESCA -basis DB5 outputs secondary structure fractions in the header.
# Parse and report ensemble-averaged fractions.

print()
print("Tip: SESCA output headers contain secondary structure fractions.")
print("     Combine with MSM populations for ensemble-averaged SS content.")
print()
print("Pipeline complete. See analysis/ for all outputs.")
