"""
01_convergence.py
=================
Step 1 — Joint PCA on backbone torsions + cosine content per trajectory.

Inputs  : trajs/*_aligned.xtc
Outputs : analysis/plots/pca_joint.png
          analysis/plots/cosine_content.txt
"""

import numpy as np
import matplotlib.pyplot as plt
import pyemma
import os

os.makedirs("analysis/plots", exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────

REF_TOP = "trajs/1IYT_dry.pdb"   # topology used for featuriser (all must match)

SYSTEMS = [
    ("1IYT",       "trajs/1IYT_aligned.xtc"),
    ("1Z0Q",       "trajs/1Z0Q_aligned.xtc"),
    ("1Z0Q_mod20", "trajs/1Z0Q_mod20_aligned.xtc"),
]

COLORS = ["#E63946", "#457B9D", "#2A9D8F"]
N_PCA_DIMS = 10

# ── Featurise ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1 — Convergence check")
print("=" * 60)
print()
print("Featurising with backbone torsions (cos/sin φ,ψ)...")

feat = pyemma.coordinates.featurizer(REF_TOP)
feat.add_backbone_torsions(cossin=True)
print(f"  N features: {feat.n_features}")

traj_files = [t[1] for t in SYSTEMS]
reader = pyemma.coordinates.source(traj_files, features=feat)

# ── Joint PCA ─────────────────────────────────────────────────────────────────

print("\nRunning joint PCA...")
pca = pyemma.coordinates.pca(reader, dim=N_PCA_DIMS)
pca_output = pca.get_output()    # list of arrays [n_frames, n_dims]

cumvar = np.cumsum(pca.eigenvalues / pca.eigenvalues.sum())
print(f"  Variance explained by PC1-3: {cumvar[2]:.1%}")

# ── Cosine content ────────────────────────────────────────────────────────────

def cosine_content(traj_pc, dim=0):
    """
    Cosine content for PC `dim`.
    Value < 0.5 indicates acceptable convergence (Becker 1997).
    """
    T = len(traj_pc)
    t = np.arange(1, T + 1)
    integral = np.trapz(traj_pc[:, dim] * np.cos(np.pi * t / T))
    norm = np.sqrt(2.0 / T) * integral
    return (norm / (np.std(traj_pc[:, dim]) + 1e-12)) ** 2

print()
print(f"{'System':<18} {'PC1':>8} {'PC2':>8} {'PC3':>8}")
print("-" * 46)

lines = [f"{'System':<18} {'PC1':>8} {'PC2':>8} {'PC3':>8}\n" + "-" * 46]

for i, (name, _) in enumerate(SYSTEMS):
    cc1 = cosine_content(pca_output[i], 0)
    cc2 = cosine_content(pca_output[i], 1)
    cc3 = cosine_content(pca_output[i], 2)

    flags = [("✓" if cc < 0.5 else "⚠") for cc in [cc1, cc2, cc3]]
    row = f"{name:<18} {cc1:>6.3f}{flags[0]} {cc2:>6.3f}{flags[1]} {cc3:>6.3f}{flags[2]}"
    print(row)
    lines.append(row)

with open("analysis/plots/cosine_content.txt", "w") as f:
    f.write("\n".join(lines))

# ── Plot PCA 2D landscape ─────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: all trajectories overlaid
ax = axes[0]
for i, (name, _) in enumerate(SYSTEMS):
    ax.scatter(pca_output[i][:, 0], pca_output[i][:, 1],
               s=1, alpha=0.25, color=COLORS[i], label=name, rasterized=True)
ax.set_xlabel(f"PC1 ({pca.eigenvalues[0]/pca.eigenvalues.sum():.1%})")
ax.set_ylabel(f"PC2 ({pca.eigenvalues[1]/pca.eigenvalues.sum():.1%})")
ax.set_title("Joint PCA — Aβ(1-42)")
ax.legend(markerscale=6, fontsize=9)

# Right: coloured by time (concatenated)
ax2 = axes[1]
for i, (name, _) in enumerate(SYSTEMS):
    n = len(pca_output[i])
    time_colour = np.arange(n) / n
    sc = ax2.scatter(pca_output[i][:, 0], pca_output[i][:, 1],
                     c=time_colour, cmap="viridis", s=1, alpha=0.3,
                     rasterized=True, vmin=0, vmax=1)

fig.colorbar(sc, ax=ax2, label="Normalised time")
ax2.set_xlabel(f"PC1")
ax2.set_ylabel(f"PC2")
ax2.set_title("PCA coloured by time (convergence)")

plt.tight_layout()
plt.savefig("analysis/plots/pca_joint.png", dpi=150)
print()
print("Saved: analysis/plots/pca_joint.png")
print("Saved: analysis/plots/cosine_content.txt")
print()
print("Next: run  python scripts/02_tica.py")
