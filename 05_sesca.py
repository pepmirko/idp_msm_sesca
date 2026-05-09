"""
05_sesca.py
===========
Step 5 — SESCA CD forward modelling + D-SPR ensemble validation.

Prerequisites
-------------
Run SESCA externally on each representative PDB (output of 04_representatives.py):

    mkdir -p analysis/sesca
    for i in $(seq 0 $((N_MACRO-1))); do
        python /path/to/sesca_main.py \
            -pdb analysis/representatives/macro_${i}.pdb \
            -basis DB5 \
            -output analysis/sesca/macro_${i}_cd.dat
    done

Inputs  : analysis/sesca/macro_N_cd.dat
          data/*_cd_exp.dat  (path set in config.py → EXP_CD)
          analysis/msm.pyemma
Outputs : analysis/plots/cd_comparison.png
          analysis/dspr_value.txt
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pyemma
import os
import sys
from config import PATHS, N_MACRO, EXP_CD

os.makedirs(PATHS["plots"], exist_ok=True)
os.makedirs(PATHS["sesca_output"], exist_ok=True)

# ── Load MSM populations ───────────────────────────────────────────────────────

print("=" * 60)
print("STEP 5 — SESCA CD forward modelling")
print("=" * 60)

msm = pyemma.load(PATHS["msm_model"])

populations = np.array([
    msm.pi[msm.metastable_assignments == i].sum()
    for i in range(N_MACRO)
])
populations /= populations.sum()

print("Macrostate populations:")
for i, p in enumerate(populations):
    print(f"  Macro {i}: {p:.4f}")

# ── Load SESCA spectra ─────────────────────────────────────────────────────────

print(f"\nLoading SESCA spectra...")

wavelengths, spectra, missing = None, [], []

for i in range(N_MACRO):
    path = f"{PATHS['sesca_output']}/macro_{i}_cd.dat"
    if not os.path.exists(path):
        missing.append(i)
        continue
    data = np.loadtxt(path, comments="#")
    if wavelengths is None:
        wavelengths = data[:, 0]
    spectra.append((i, data[:, 1]))
    print(f"  macro_{i}_cd.dat — {len(data)} points")

if missing:
    print(f"\n✗ Missing SESCA output for macrostates: {missing}")
    print("  Run SESCA on the corresponding PDBs first (see script header).")
    sys.exit(1)

spectra_idx   = [i for i, _ in spectra]
spectra_array = np.array([s for _, s in spectra])   # [N_MACRO, n_wl]

# ── Ensemble spectrum ──────────────────────────────────────────────────────────

cd_ensemble = np.sum(populations[spectra_idx, None] * spectra_array, axis=0)

# ── Experimental data + D-SPR ──────────────────────────────────────────────────

def d_spr(pred, exp):
    """Normalised spectral RMSD. 0 = perfect; < 0.25 = good."""
    residuals = pred - exp
    return np.sqrt(np.mean(residuals**2)) / (np.sqrt(np.mean(exp**2)) + 1e-12)

exp_available = os.path.exists(EXP_CD)
dspr_value    = None

if not exp_available:
    print(f"\n⚠ Experimental data not found at {EXP_CD}")
    print("  Plotting ensemble spectrum only — D-SPR skipped.")
else:
    cd_exp    = np.loadtxt(EXP_CD, comments="#")
    wl_exp    = cd_exp[:, 0]
    signal_exp = cd_exp[:, 1]

    f_interp  = interp1d(wavelengths, cd_ensemble, kind="cubic",
                         fill_value="extrapolate", bounds_error=False)
    cd_pred   = f_interp(wl_exp)
    dspr_value = d_spr(cd_pred, signal_exp)

    quality = ("Excellent" if dspr_value < 0.15 else
               "Good"      if dspr_value < 0.25 else
               "Moderate"  if dspr_value < 0.40 else "Poor")

    print(f"\nD-SPR = {dspr_value:.4f}  ({quality})")

    with open(PATHS["dspr"], "w") as fh:
        fh.write(f"D-SPR    = {dspr_value:.6f}\n")
        fh.write(f"Quality  = {quality}\n")
        fh.write(f"N_MACRO  = {N_MACRO}\n")
        for i, p in enumerate(populations):
            fh.write(f"  Macro {i}: pop = {p:.4f}\n")
    print(f"Saved: {PATHS['dspr']}")

# ── Plot ───────────────────────────────────────────────────────────────────────

COLORS_MACRO = plt.cm.tab10(np.linspace(0, 0.9, N_MACRO))

fig, ax = plt.subplots(figsize=(9, 6))

wl_plot = wl_exp if exp_available else wavelengths

for idx, (i, sp) in enumerate(zip(spectra_idx, spectra_array)):
    f_i = interp1d(wavelengths, sp, kind="cubic",
                   fill_value="extrapolate", bounds_error=False)
    ax.plot(wl_plot, f_i(wl_plot), lw=1.2, alpha=0.4,
            color=COLORS_MACRO[idx], label=f"Macro {i} (pop={populations[i]:.2f})")

f_ens     = interp1d(wavelengths, cd_ensemble, kind="cubic",
                     fill_value="extrapolate", bounds_error=False)
ens_label = (f"Ensemble (D-SPR={dspr_value:.3f})" if dspr_value is not None
             else "Ensemble prediction")
ax.plot(wl_plot, f_ens(wl_plot), lw=2.5, color="crimson",
        ls="--", label=ens_label, zorder=5)

if exp_available:
    ax.plot(wl_exp, signal_exp, lw=2.5, color="black",
            label="Experimental", zorder=6)

ax.axhline(0, color="grey", lw=0.6)
ax.set_xlabel("Wavelength (nm)", fontsize=12)
ax.set_ylabel("ΔΕ  (M⁻¹ cm⁻¹)  or  [θ]", fontsize=12)
ax.set_title("SESCA CD Forward Modelling", fontsize=13)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{PATHS['plots']}/cd_comparison.png", dpi=150)

print(f"Saved: {PATHS['plots']}/cd_comparison.png")
print()
print("Pipeline complete. All outputs in analysis/")
