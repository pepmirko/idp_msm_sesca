"""
00_preprocess.py
================
Step 0 — PBC fix, Cα alignment, topology consistency check.

Inputs  : trajs/*_dry.xtc + trajs/*_dry.pdb
Outputs : trajs/*_aligned.xtc
"""

import MDAnalysis as mda
from MDAnalysis.analysis import align
import sys

# ── Configuration ────────────────────────────────────────────────────────────

SYSTEMS = {
    "1IYT":       ("trajs/1IYT_dry.pdb",        "trajs/1IYT_dry.xtc"),
    "1Z0Q":       ("trajs/1Z0Q_dry.pdb",         "trajs/1Z0Q_dry.xtc"),
    "1Z0Q_mod20": ("trajs/1Z0Q_mod20_dry.pdb",   "trajs/1Z0Q_mod20_dry.xtc"),
}

ALIGN_SELECTION = "backbone"  # used for both alignment and topology check

# ── Topology check ────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 0 — Topology check")
print("=" * 60)

atom_counts = {}
frame_counts = {}

for name, (top, traj) in SYSTEMS.items():
    u = mda.Universe(top, traj)
    atom_counts[name] = u.atoms.n_atoms
    frame_counts[name] = u.trajectory.n_frames
    print(f"  {name:<16}: {u.atoms.n_atoms:>6} atoms | {u.trajectory.n_frames:>6} frames")

counts = list(atom_counts.values())
if len(set(counts)) != 1:
    print("\n✗ TOPOLOGY MISMATCH — atom counts differ across systems.")
    print("  Re-generate trajectories with gmx trjconv using the same .tpr")
    print("  or re-strip topologies to a common atom set before proceeding.")
    sys.exit(1)
else:
    print("\n✓ Topology check passed — all systems have identical atom counts.\n")

# ── Alignment ─────────────────────────────────────────────────────────────────

print("Aligning trajectories on Cα to frame 0 of each respective topology...")
print()

for name, (top, traj) in SYSTEMS.items():
    out_path = f"trajs/{name}_aligned.xtc"
    u   = mda.Universe(top, traj)
    ref = mda.Universe(top)           # static reference = frame 0

    aligner = align.AlignTraj(
        u, ref,
        select=ALIGN_SELECTION,
        in_memory=False,
        filename=out_path,
        verbose=False,
    )
    aligner.run()
    print(f"  ✓ {name:<16} → {out_path}")

print()
print("Preprocessing complete. Aligned files are in trajs/")
print("Next: run  python scripts/01_convergence.py")
