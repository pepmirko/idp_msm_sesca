"""
02_tica.py
==========
Step 2 — Cα pairwise distance featurisation + TICA lagtime scan.

Inputs  : trajs/*_aligned.xtc
Outputs : analysis/plots/its_tica.png
          analysis/tica.pyemma   (serialised TICA model)
          analysis/tica_output/  (numpy arrays, one per trajectory)

Edit LAG_TICA_FINAL after inspecting the ITS plot.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyemma
import os

os.makedirs("analysis/plots", exist_ok=True)
os.makedirs("analysis/tica_output", exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────

REF_TOP = "trajs/1IYT_dry.pdb"

SYSTEMS = [
    ("1IYT",       "trajs/1IYT_aligned.xtc"),
    ("1Z0Q",       "trajs/1Z0Q_aligned.xtc"),
    ("1Z0Q_mod20", "trajs/1Z0Q_mod20_aligned.xtc"),
]

# Lagtimes to scan (frames). Adjust if your saving stride differs.
# At 100 ps/frame: lag=50 → 5 ns, lag=200 → 20 ns
LAGTIMES_SCAN = [5, 10, 20, 50, 100, 200, 500]

# ── SET THIS after inspecting its_tica.png ────────────────────────────────────
LAG_TICA_FINAL = 100     # frames — EDIT based on ITS plateau
N_TICA_DIMS    = 6       # number of TICA components to retain

# ── Featurise: pairwise Cα distances ─────────────────────────────────────────

print("=" * 60)
print("STEP 2 — Featurisation + TICA")
print("=" * 60)
print()
print("Featurising with pairwise Cα distances (excluded_neighbors=2)...")

feat = pyemma.coordinates.featurizer(REF_TOP)
feat.add_distances_ca(excluded_neighbors=2, periodic=False)
print(f"  N features: {feat.n_features}")

traj_files = [t[1] for t in SYSTEMS]
reader = pyemma.coordinates.source(traj_files, features=feat)

# ── ITS scan ──────────────────────────────────────────────────────────────────

print(f"\nScanning TICA lagtimes: {LAGTIMES_SCAN} frames")
print(f"{'lag':>6}  {'ITS1':>10}  {'ITS2':>10}  {'ITS3':>10}  {'ITS4':>10}")
print("-" * 50)

its_records = []

for lag in LAGTIMES_SCAN:
    tica_tmp = pyemma.coordinates.tica(reader, lag=lag, dim=N_TICA_DIMS,
                                       kinetic_map=True)
    its = tica_tmp.timescales[:5]
    its_records.append(its)
    vals = "  ".join(f"{v:>10.1f}" for v in its[:4])
    print(f"{lag:>6}  {vals}")

its_records = np.array(its_records)   # [n_lags, 5]

# ── Plot ITS ──────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
colors_its = plt.cm.plasma(np.linspace(0.1, 0.9, 5))

for k in range(5):
    ax.plot(LAGTIMES_SCAN, its_records[:, k], marker='o',
            color=colors_its[k], label=f"ITS {k+1}")

ax.axvline(LAG_TICA_FINAL, color="red", ls="--", lw=1.5,
           label=f"Current lag = {LAG_TICA_FINAL}")
ax.set_xlabel("TICA lagtime (frames)")
ax.set_ylabel("Implied timescale (frames)")
ax.set_title("ITS vs TICA lagtime — Aβ(1-42)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("analysis/plots/its_tica.png", dpi=150)
print(f"\nSaved: analysis/plots/its_tica.png")
print(f"→ Inspect the plot and set LAG_TICA_FINAL at the plateau.")
print(f"  Current value: {LAG_TICA_FINAL} frames\n")

# ── Final TICA ────────────────────────────────────────────────────────────────

print(f"Running final TICA with lag={LAG_TICA_FINAL}, dim={N_TICA_DIMS}...")

tica = pyemma.coordinates.tica(reader, lag=LAG_TICA_FINAL,
                                dim=N_TICA_DIMS, kinetic_map=True)

cumvar = np.cumsum(tica.cumvar)
print(f"  Cumulative variance explained:")
for i, cv in enumerate(cumvar):
    print(f"    IC{i+1}: {cv:.1%}")

tica_output = tica.get_output()   # list of [n_frames, n_dims] arrays

# Save each trajectory's TICA projection
for i, (name, _) in enumerate(SYSTEMS):
    out_path = f"analysis/tica_output/{name}.npy"
    np.save(out_path, tica_output[i])
    print(f"  Saved: {out_path}")

# Serialise TICA model
tica.save("analysis/tica.pyemma", overwrite=True)
print("\nSaved: analysis/tica.pyemma")
print()
print("Next: run  python scripts/03_msm.py")
