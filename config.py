"""
config.py
=========
Central configuration for the idp_msm_sesca pipeline.

Edit this file to switch between systems or tune MSM parameters.
All scripts import from here — you never need to edit the scripts themselves.

Usage
-----
    from config import SYSTEMS, REF_TOP, TRAJ_FILES, LAG_TICA, LAG_MSM, N_MACRO, ...
"""

# ── Active system ──────────────────────────────────────────────────────────────
# Switch between pre-defined system blocks below, or define your own.

ACTIVE_SYSTEM = "abeta42"   # options: "abeta42" | "tau" | "custom"

# ── System definitions ─────────────────────────────────────────────────────────

_SYSTEMS = {

    "abeta42": {
        "systems": {
            "1IYT":       ("trajs/1IYT_dry.pdb",       "trajs/1IYT_dry.xtc"),
            "1Z0Q":       ("trajs/1Z0Q_dry.pdb",        "trajs/1Z0Q_dry.xtc"),
            "1Z0Q_mod20": ("trajs/1Z0Q_mod20_dry.pdb",  "trajs/1Z0Q_mod20_dry.xtc"),
        },
        "ref_top":       "trajs/1IYT_dry.pdb",
        "exp_cd_path":   "data/abeta42_cd_exp.dat",
        "lag_tica":      100,    # frames — tune from its_tica.png
        "lag_msm":       20,     # frames — tune from its_msm.png
        "n_macro":       5,      # PCCA+ macrostates
        "dt_traj":       "0.1 ns",
    },

    "tau": {
        "systems": {
            "tau_ext":   ("trajs/tau_ext.pdb",   "trajs/tau_ext.xtc"),
            "tau_helix": ("trajs/tau_helix.pdb", "trajs/tau_helix.xtc"),
            "tau_nmr":   ("trajs/tau_nmr.pdb",   "trajs/tau_nmr.xtc"),
        },
        "ref_top":       "trajs/tau_ext.pdb",
        "exp_cd_path":   "data/tau_cd_exp.dat",
        "lag_tica":      200,    # frames — tune from its_tica.png
        "lag_msm":       50,     # frames — tune from its_msm.png
        "n_macro":       5,
        "dt_traj":       "0.1 ns",
    },

    # ── Add your own system here ───────────────────────────────────────────────
    "custom": {
        "systems": {
            "sys_a": ("trajs/sys_a.pdb", "trajs/sys_a.xtc"),
            "sys_b": ("trajs/sys_b.pdb", "trajs/sys_b.xtc"),
        },
        "ref_top":       "trajs/sys_a.pdb",
        "exp_cd_path":   "data/custom_cd_exp.dat",
        "lag_tica":      100,
        "lag_msm":       20,
        "n_macro":       5,
        "dt_traj":       "0.1 ns",
    },
}

# ── Export active configuration ────────────────────────────────────────────────

_cfg = _SYSTEMS[ACTIVE_SYSTEM]

SYSTEMS    = _cfg["systems"]                    # dict {name: (top, traj)}
REF_TOP    = _cfg["ref_top"]                    # topology for featuriser
EXP_CD     = _cfg["exp_cd_path"]               # experimental CD file
LAG_TICA   = _cfg["lag_tica"]                   # TICA lagtime (frames)
LAG_MSM    = _cfg["lag_msm"]                    # MSM lagtime (frames)
N_MACRO    = _cfg["n_macro"]                    # number of PCCA+ macrostates
DT_TRAJ    = _cfg["dt_traj"]                    # time step between saved frames

# Derived convenience lists
SYSTEM_NAMES = list(SYSTEMS.keys())
TRAJ_FILES   = [v[1] for v in SYSTEMS.values()]
TOP_FILES    = [v[0] for v in SYSTEMS.values()]

# ── TICA scan parameters ───────────────────────────────────────────────────────

LAGTIMES_SCAN_TICA = [5, 10, 20, 50, 100, 200, 500]   # for its_tica.png
LAGTIMES_SCAN_MSM  = [1, 2, 5, 10, 20, 50, 100, 200]  # for its_msm.png
N_TICA_DIMS        = 6     # TICA components to retain

# ── Clustering ─────────────────────────────────────────────────────────────────

import numpy as np

def n_clusters_heuristic(n_frames_total: int, divisor: int = 10) -> int:
    """max(100, √(N_frames / divisor))"""
    return max(100, int(np.sqrt(n_frames_total / divisor)))

# ── Paths ──────────────────────────────────────────────────────────────────────

PATHS = {
    "tica_model":       "analysis/tica.pyemma",
    "tica_output_dir":  "analysis/tica_output",
    "msm_model":        "analysis/msm.pyemma",
    "dtrajs":           "analysis/dtrajs.npy",
    "macrostate_info":  "analysis/macrostate_info.txt",
    "representatives":  "analysis/representatives",
    "sesca_output":     "analysis/sesca",
    "plots":            "analysis/plots",
    "dspr":             "analysis/dspr_value.txt",
}

# ── Sanity check on import ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Active system : {ACTIVE_SYSTEM}")
    print(f"Trajectories  : {SYSTEM_NAMES}")
    print(f"LAG_TICA      : {LAG_TICA} frames")
    print(f"LAG_MSM       : {LAG_MSM} frames")
    print(f"N_MACRO       : {N_MACRO}")
    print(f"Exp. CD       : {EXP_CD}")
