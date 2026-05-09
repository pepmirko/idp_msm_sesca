"""
05_representatives.py
=====================
Step 5 — Extract centroid PDB for each PCCA+ macrostate.

Inputs  : analysis/msm.pyemma
          analysis/tica_output/*.npy
          analysis/dtrajs.npy
          trajs/*_aligned.xtc
Outputs : analysis/representatives/macro_N.pdb
          analysis/representatives/summary.txt
"""

import numpy as np
import MDAnalysis as mda
import pyemma
import os

os.makedirs("analysis/representatives", exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────

SYSTEMS = [
    ("1IYT",       "trajs/1IYT_dry.pdb",       "trajs/1IYT_aligned.xtc"),
    ("1Z0Q",       "trajs/1Z0Q_dry.pdb",        "trajs/1Z0Q_aligned.xtc"),
    ("1Z0Q_mod20", "trajs/1Z0Q_mod20_dry.pdb",  "trajs/1Z0Q_mod20_aligned.xtc"),
]

# ── Load models ───────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 5 — Representative structures")
print("=" * 60)
print()

msm = pyemma.load("analysis/msm.pyemma")
N_MACRO = msm.n_metastable

tica_output = []
for name, _, _ in SYSTEMS:
    tica_output.append(np.load(f"analysis/tica_output/{name}.npy"))

dtrajs_raw = np.load("analysis/dtrajs.npy", allow_pickle=True)
dtrajs = list(dtrajs_raw)

# Recompute populations
populations = np.array([
    msm.pi[msm.metastable_assignments == i].sum()
    for i in range(N_MACRO)
])
populations /= populations.sum()

# ── Find centroid frames ──────────────────────────────────────────────────────

def get_representative_frames(tica_output, dtrajs, msm, n_macro):
    """
    For each macrostate, collect all member frames in TICA space,
    compute the centroid, and return the frame closest to it.
    Returns: dict {macro_i: (traj_idx, frame_idx)}
    """
    reps = {}
    for macro_i in range(n_macro):
        micro_states = set(np.where(msm.metastable_assignments == macro_i)[0])

        frames_macro  = []
        tica_macro    = []

        for traj_i, dtraj in enumerate(dtrajs):
            for frame_j, state in enumerate(dtraj):
                if state in micro_states:
                    frames_macro.append((traj_i, frame_j))
                    tica_macro.append(tica_output[traj_i][frame_j])

        tica_macro = np.array(tica_macro)
        centroid   = tica_macro.mean(axis=0)
        closest    = np.argmin(np.linalg.norm(tica_macro - centroid, axis=1))
        reps[macro_i] = frames_macro[closest]

    return reps

print("Finding centroid frames in TICA space...")
reps = get_representative_frames(tica_output, dtrajs, msm, N_MACRO)

# ── Write PDBs ────────────────────────────────────────────────────────────────

print()
print(f"{'Macro':<8} {'Pop.':<8} {'Trajectory':<16} {'Frame':<8}")
print("-" * 44)

summary_lines = [f"{'Macro':<8} {'Pop.':<8} {'Trajectory':<16} {'Frame':<8}", "-" * 44]

for macro_i in range(N_MACRO):
    traj_i, frame_j = reps[macro_i]
    name, top_path, traj_path = SYSTEMS[traj_i]

    u = mda.Universe(top_path, traj_path)
    u.trajectory[frame_j]
    protein = u.select_atoms("protein")
    out_pdb = f"analysis/representatives/macro_{macro_i}.pdb"
    protein.write(out_pdb)

    row = f"{macro_i:<8} {populations[macro_i]:.4f}  {name:<16} {frame_j:<8}"
    print(row)
    summary_lines.append(row)

with open("analysis/representatives/summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

print()
print("PDB files written to analysis/representatives/")
print("Saved: analysis/representatives/summary.txt")
print()
print("Next: run SESCA on each PDB, then  python scripts/04_sesca.py")
print()
print("Example SESCA call:")
print("  python /path/to/sesca_main.py \\")
print("      -pdb analysis/representatives/macro_0.pdb \\")
print("      -basis DB5 \\")
print("      -output analysis/sesca/macro_0_cd.dat")
