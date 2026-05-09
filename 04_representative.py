"""
04_representatives.py
=====================
Step 4 — Extract centroid PDB for each PCCA+ macrostate.

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
from config import SYSTEMS, SYSTEM_NAMES, PATHS, N_MACRO

os.makedirs(PATHS["representatives"], exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 4 — Representative structures")
print("=" * 60)

msm = pyemma.load(PATHS["msm_model"])

tica_output = [np.load(f"{PATHS['tica_output_dir']}/{name}.npy")
               for name in SYSTEM_NAMES]

dtrajs = list(np.load(PATHS["dtrajs"], allow_pickle=True))

populations = np.array([
    msm.pi[msm.metastable_assignments == i].sum()
    for i in range(N_MACRO)
])
populations /= populations.sum()

# ── Find centroid frames ───────────────────────────────────────────────────────

def get_representatives(tica_output, dtrajs, msm, n_macro):
    reps = {}
    for macro_i in range(n_macro):
        micro_states = set(np.where(msm.metastable_assignments == macro_i)[0])
        frames, tica_pts = [], []
        for traj_i, dtraj in enumerate(dtrajs):
            for frame_j, state in enumerate(dtraj):
                if state in micro_states:
                    frames.append((traj_i, frame_j))
                    tica_pts.append(tica_output[traj_i][frame_j])
        tica_pts = np.array(tica_pts)
        centroid = tica_pts.mean(axis=0)
        closest  = np.argmin(np.linalg.norm(tica_pts - centroid, axis=1))
        reps[macro_i] = frames[closest]
    return reps

print("Computing centroids in TICA space...")
reps = get_representatives(tica_output, dtrajs, msm, N_MACRO)

# ── Write PDBs ─────────────────────────────────────────────────────────────────

system_list = list(SYSTEMS.items())   # [(name, (top, traj)), ...]
aligned_trajs = [f"trajs/{name}_aligned.xtc" for name in SYSTEM_NAMES]

print()
print(f"{'Macro':<8} {'Pop.':<8} {'Trajectory':<18} {'Frame':<8}")
print("-" * 46)
lines = [f"{'Macro':<8} {'Pop.':<8} {'Trajectory':<18} {'Frame':<8}", "-" * 46]

for macro_i in range(N_MACRO):
    traj_i, frame_j = reps[macro_i]
    name, (top_path, _) = system_list[traj_i]

    u = mda.Universe(top_path, aligned_trajs[traj_i])
    u.trajectory[frame_j]
    out_pdb = f"{PATHS['representatives']}/macro_{macro_i}.pdb"
    u.select_atoms("protein").write(out_pdb)

    row = f"{macro_i:<8} {populations[macro_i]:.4f}  {name:<18} {frame_j:<8}"
    print(row)
    lines.append(row)

with open(f"{PATHS['representatives']}/summary.txt", "w") as f:
    f.write("\n".join(lines))

print(f"\nPDBs written to {PATHS['representatives']}/")
print(f"Saved: {PATHS['representatives']}/summary.txt")
print()
print("Run SESCA on each PDB, then: python scripts/05_sesca.py")
print()
print("Example SESCA call:")
for i in range(N_MACRO):
    print(f"  python /path/to/sesca_main.py "
          f"-pdb {PATHS['representatives']}/macro_{i}.pdb "
          f"-basis DB5 "
          f"-output {PATHS['sesca_output']}/macro_{i}_cd.dat")
