"""
03_msm.py
=========
Step 3 — k-means clustering, Bayesian MSM, Chapman-Kolmogorov test, PCCA+.

Inputs  : analysis/tica_output/*.npy   (from 02_tica.py)
Outputs : analysis/plots/its_msm.png
          analysis/plots/ck_test.png
          analysis/msm.pyemma
          analysis/dtrajs.npy          (discrete trajectories)
          analysis/macrostate_info.txt
"""

import numpy as np
import matplotlib.pyplot as plt
import pyemma
from pyemma.msm import bayesian_markov_model, timescales_msm
import os

os.makedirs("analysis/plots", exist_ok=True)

# ── Configuration — EDIT THESE after inspecting ITS plots ─────────────────────

SYSTEMS = ["1IYT", "1Z0Q", "1Z0Q_mod20"]

LAG_TICA   = 100    # must match what was used in 02_tica.py
LAG_MSM    = 20     # frames — set to plateau in its_msm.png
N_MACRO    = 5      # macrostates for PCCA+ — reduce to 3 if CK test fails

DT_TRAJ    = "0.1 ns"   # time step between saved frames — adjust to your stride

# ── Load TICA output ──────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 3 — Clustering + MSM")
print("=" * 60)
print()
print("Loading TICA projections...")

tica_output = []
for name in SYSTEMS:
    arr = np.load(f"analysis/tica_output/{name}.npy")
    tica_output.append(arr)
    print(f"  {name}: {arr.shape}")

n_frames_total = sum(a.shape[0] for a in tica_output)

# ── k-means clustering ────────────────────────────────────────────────────────

n_clusters = max(100, int(np.sqrt(n_frames_total / 10)))
print(f"\nClustering: k-means with k={n_clusters} (heuristic: √(N/{10}))")

clustering = pyemma.coordinates.cluster_kmeans(
    tica_output,
    k=n_clusters,
    max_iter=100,
    stride=10,
    fixed_seed=42,
)

dtrajs = clustering.dtrajs   # list of integer arrays

# Save discrete trajectories for later steps
np.save("analysis/dtrajs.npy", np.array(dtrajs, dtype=object))
print(f"  Saved: analysis/dtrajs.npy")

# ── MSM implied timescales scan ───────────────────────────────────────────────

print(f"\nScanning MSM lagtimes for implied timescales...")

lags_msm = [1, 2, 5, 10, 20, 50, 100, 200]
its_msm  = timescales_msm(dtrajs, lags=lags_msm, nits=5, errors="bayes")

fig, ax = plt.subplots(figsize=(8, 5))
pyemma.plots.plot_implied_timescales(its_msm, ax=ax, units="frames",
                                      dt=float(DT_TRAJ.split()[0]))
ax.axvline(LAG_MSM, color="red", ls="--", lw=1.5,
           label=f"Chosen lag = {LAG_MSM}")
ax.set_title("MSM Implied Timescales — Aβ(1-42)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("analysis/plots/its_msm.png", dpi=150)
print(f"  Saved: analysis/plots/its_msm.png")
print(f"  → Verify plateau at lag={LAG_MSM}. Edit LAG_MSM if needed.\n")

# ── Bayesian MSM ──────────────────────────────────────────────────────────────

print(f"Estimating Bayesian MSM (lag={LAG_MSM}, 200 samples)...")

msm = bayesian_markov_model(
    dtrajs,
    lag=LAG_MSM,
    dt_traj=DT_TRAJ,
    conf=0.95,
    n_samples=200,
)

print(f"  Active states:        {msm.n_states}")
print(f"  Active count fraction: {msm.active_count_fraction:.3f}",
      "✓" if msm.active_count_fraction > 0.90 else "⚠ consider reducing lag or n_clusters")

top5_its = msm.timescales[:5]
print(f"  Top-5 timescales (in trajectory units): {np.round(top5_its, 1)}")

# Serialise
msm.save("analysis/msm.pyemma", overwrite=True)
print("  Saved: analysis/msm.pyemma")

# ── Chapman-Kolmogorov test ───────────────────────────────────────────────────

print(f"\nRunning Chapman-Kolmogorov test (n_sets={N_MACRO}, mlags=10)...")

ck = msm.cktest(N_MACRO, mlags=10)

fig_ck, _ = plt.subplots(figsize=(12, 3 * ((N_MACRO + 1) // 2)))
pyemma.plots.plot_cktest(ck, figsize=(12, 3 * ((N_MACRO + 1) // 2)))
plt.suptitle(f"Chapman-Kolmogorov test — {N_MACRO} sets", y=1.01)
plt.tight_layout()
plt.savefig("analysis/plots/ck_test.png", dpi=150, bbox_inches="tight")
print("  Saved: analysis/plots/ck_test.png")
print("  → Dashed (predicted) and solid (re-estimated) lines should overlap.")

# ── PCCA+ macrostates ─────────────────────────────────────────────────────────

print(f"\nRunning PCCA+ with {N_MACRO} macrostates...")
msm.pcca(N_MACRO)

populations = np.array([
    msm.pi[msm.metastable_assignments == i].sum()
    for i in range(N_MACRO)
])
populations /= populations.sum()

print()
print(f"  {'Macrostate':<12} {'Population':>12} {'N microstates':>15}")
print("  " + "-" * 42)
for i in range(N_MACRO):
    n_micro = np.sum(msm.metastable_assignments == i)
    print(f"  {i:<12} {populations[i]:>12.4f} {n_micro:>15}")

# Save summary
lines = [
    f"LAG_TICA   = {LAG_TICA}",
    f"LAG_MSM    = {LAG_MSM}",
    f"N_CLUSTERS = {n_clusters}",
    f"N_MACRO    = {N_MACRO}",
    f"active_count_fraction = {msm.active_count_fraction:.4f}",
    "",
    f"{'Macrostate':<12} {'Population':>12} {'N microstates':>15}",
    "-" * 42,
]
for i in range(N_MACRO):
    n_micro = np.sum(msm.metastable_assignments == i)
    lines.append(f"{i:<12} {populations[i]:>12.4f} {n_micro:>15}")

with open("analysis/macrostate_info.txt", "w") as f:
    f.write("\n".join(lines))

print()
print("Saved: analysis/macrostate_info.txt")
print()
print("Next: run  python scripts/05_representatives.py")
print("      then python scripts/04_sesca.py")
