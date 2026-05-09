# IDP MSM SESCA Pipeline

**by Mirko Caruso (UniCT) & Giuseppe Stefano Basile (SNS)**

End-to-end pipeline for building a Markov State Model from multiple MD trajectories of an intrinsically disordered peptide (IDP) and validating the conformational ensemble against experimental CD spectra via SESCA forward modelling.

> **Force field:** a99SB-disp + TIP4P-D  
> **Software:** GROMACS, PyEMMA, MDAnalysis, SESCA-2025

**Validated on:**
- Aβ(1-42) — 3 NMR starting structures (1IYT, 1Z0Q mod1, 1Z0Q mod20), ~750 ns/replica
- Aβ(1-40) — REST2 conformational ensemble (16 replicas, 300–450 K, 500 ns/replica)
- Tau(272-314) — microtubule-binding repeat region
- CB01, CB02, CB03, VW19 — see https://github.com/pepmirko/pdb_BScthesis

The pipeline is **system-agnostic**: swap the `SYSTEMS` dict in each script to target any IDP.

---

## Table of Contents

1. [Scientific rationale](#scientific-rationale)
2. [Repository structure](#repository-structure)
3. [Installation](#installation)
4. [Input data](#input-data)
5. [Pipeline overview](#pipeline-overview)
6. [Step-by-step guide](#step-by-step-guide)
   - [Step 0 — Preprocessing](#step-0--preprocessing)
   - [Step 1 — Convergence check](#step-1--convergence-check)
   - [Step 2 — Featurisation + TICA](#step-2--featurisation--tica)
   - [Step 3 — Clustering + MSM](#step-3--clustering--msm)
   - [Step 4 — SESCA CD forward modelling](#step-4--sesca-cd-forward-modelling)
   - [Step 5 — Representative structures](#step-5--representative-structures)
7. [Adapting to a new system](#adapting-to-a-new-system)
8. [Outputs](#outputs)
9. [Troubleshooting](#troubleshooting)
10. [References](#references)
11. [Acknowledgements](#acknowledgements)

---

## Scientific rationale

IDPs such as Aβ(1-42) and Tau(272-314) do not adopt a single folded structure: their biological activity and aggregation propensity are governed by a **conformational ensemble**. Classical MD started from a single structure undersamples this space. This pipeline addresses the problem by:

- Starting from **multiple diverse starting structures** (NMR bundles, homology models, random coil) to maximise coverage of conformational space
- Building a **Markov State Model** (MSM) on the combined trajectory space to extract thermodynamic populations and kinetic connectivity
- Validating the ensemble via **SESCA CD forward modelling**: predicted CD spectra are weighted by MSM populations and compared to experimental data using the D-SPR metric

This approach mirrors the ensemble-first strategy validated on α-helical model peptides (CB02, CB03, VW19) and extends it to the disordered/aggregating regime. For enhanced sampling, Replica Exchange with Solute Tempering 2 (REST2) can be used upstream (see Input data section), with the multi-replica trajectory set fed directly into Step 0.

---

## Repository structure

```
idp_msm_sesca/
│
├── README.md
├── environment.yml            # conda environment
├── .gitignore
│
├── scripts/
│   ├── 00_preprocess.py       # PBC fix, alignment, topology check
│   ├── 01_convergence.py      # Joint PCA + cosine content
│   ├── 02_tica.py             # Featurisation (Cα distances) + TICA lagtime scan
│   ├── 03_msm.py              # k-means clustering + Bayesian MSM + PCCA+
│   ├── 04_sesca.py            # CD forward modelling + D-SPR
│   └── 05_representatives.py  # Centroid extraction + PDB output
│
├── data/
│   ├── README_data.md         # instructions for placing input files
│   ├── abeta42_cd_exp.dat     # experimental CD — Aβ(1-42)
│   ├── abeta40_cd_exp.dat     # experimental CD — Aβ(1-40)
│   └── tau_cd_exp.dat         # experimental CD — Tau(272-314)
│
└── analysis/
    ├── plots/                 # all figures
    ├── sesca/                 # per-macrostate SESCA output
    └── representatives/       # representative PDB structures
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/idp_msm_sesca.git
cd idp_msm_sesca
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate idp_msm_sesca
```

### 3. Install SESCA-2025

SESCA is not available via pip/conda and must be installed separately:

```bash
# Download from: https://www.mrc-lmb.cam.ac.uk/sesca/
cd /path/to/sesca
python setup.py install
```

Verify:
```bash
python -c "import sesca; print('SESCA ok')"
```

---

## Input data

Place trajectory files in `trajs/` (not tracked by git):

```
trajs/
├── SYS_A.pdb  +  SYS_A.xtc
├── SYS_B.pdb  +  SYS_B.xtc
└── ...
```

**Example — Aβ(1-42):**
```
trajs/
├── 1IYT_dry.pdb       +  1IYT_dry.xtc
├── 1Z0Q_dry.pdb       +  1Z0Q_dry.xtc
└── 1Z0Q_mod20_dry.pdb +  1Z0Q_mod20_dry.xtc
```

**Example — Aβ(1-40) / Aβ(1-42) via REST2 (LEONARDO BOOSTER):**
```
trajs/
├── rep_00.pdb  +  rep_00.xtc    # T = 300 K (target replica)
├── rep_01.pdb  +  rep_01.xtc
├── ...
└── rep_15.pdb  +  rep_15.xtc    # T = 450 K
```
Only the target-temperature replica (rep_00, T = 300 K) is fed into the MSM; higher-temperature replicas are used solely to drive enhanced sampling.

**Example — Tau(272-314):**
```
trajs/
├── tau_ext.pdb   +  tau_ext.xtc      # extended starting structure
├── tau_helix.pdb +  tau_helix.xtc    # helical seed
└── tau_nmr.pdb   +  tau_nmr.xtc      # NMR-derived seed (if available)
```

> **Critical:** all replicas / systems must have **identical atom counts and ordering**. `00_preprocess.py` will flag any mismatch.

The experimental CD file (`data/*.dat`) should be two columns:
```
# wavelength(nm)   delta_epsilon_or_theta
200.0   -3.21
201.0   -4.05
```

Suitable references:
- Aβ(1-40)/(1-42): Barrow & Zagorski (1991); Jarvet et al. (2007)
- Tau(272-314): Mukrasch et al. (2009); von Bergen et al. (2001)

---

## Pipeline overview

```
trajs/*.xtc  (plain MD or REST2 target replica)
     │
     ▼
[00] Preprocessing ──────── PBC unwrap, Cα alignment, topology check
     │
     ▼
[01] Convergence check ───── Joint PCA, cosine content (threshold < 0.5)
     │
     ▼
[02] Featurisation + TICA ── Cα pairwise distances, ITS scan → optimal lagtime
     │
     ▼
[03] Clustering + MSM ────── k-means (√N heuristic), Bayesian MSM,
     │                       Chapman-Kolmogorov test, PCCA+ macrostates
     ▼
[04] SESCA CD ────────────── Per-macrostate CD prediction,
     │                       population-weighted ensemble spectrum, D-SPR
     ▼
[05] Representatives ──────── Centroid PDB for each macrostate
```

---

## Step-by-step guide

### Step 0 — Preprocessing

**Script:** `scripts/00_preprocess.py`

Checks topology consistency, unwraps PBC, aligns all trajectories on Cα to a common reference frame. Outputs `trajs/*_aligned.xtc`.

```bash
python scripts/00_preprocess.py
```

Expected output:
```
1IYT_dry:       4200 atoms, 7500 frames  ✓
1Z0Q_dry:       4200 atoms, 7500 frames  ✓
1Z0Q_mod20_dry: 4200 atoms, 7500 frames  ✓
✓ Topology check passed
```

> ⚠️ If atom counts differ, re-process topologies in GROMACS (`gmx trjconv -pbc mol`) using the same `.tpr`.

---

### Step 1 — Convergence check

**Script:** `scripts/01_convergence.py`

Joint PCA on backbone torsions (φ/ψ) + cosine content per trajectory.

```bash
python scripts/01_convergence.py
```

| Cosine content | Interpretation |
|---|---|
| < 0.1 | Excellent convergence |
| 0.1 – 0.5 | Acceptable |
| > 0.5 | ⚠️ Poor — report as limitation |

**Output:** `analysis/plots/pca_joint.png`, `analysis/plots/cosine_content.txt`

---

### Step 2 — Featurisation + TICA

**Script:** `scripts/02_tica.py`

Featurises as pairwise Cα distances (`excluded_neighbors=2`), scans TICA lagtimes, runs final TICA with `kinetic_map=True`.

```bash
python scripts/02_tica.py
```

Inspect `analysis/plots/its_tica.png` and set `LAG_TICA_FINAL` to the smallest lagtime at which the top ITS plateau. Typical values:

| System | Suggested lagtime range |
|---|---|
| Aβ(1-40)/(1-42) at 100 ps/frame | 50–200 frames (5–20 ns) |
| Tau(272-314) at 100 ps/frame | 50–500 frames (5–50 ns) |

**Output:** `analysis/plots/its_tica.png`, `analysis/tica.pyemma`

---

### Step 3 — Clustering + MSM

**Script:** `scripts/03_msm.py`

k-means → ITS scan → Bayesian MSM (200 samples) → Chapman-Kolmogorov test → PCCA+.

```bash
python scripts/03_msm.py
```

**Parameters to set** (top of script):
```python
LAG_TICA = 100   # from Step 2
LAG_MSM  = 20    # from ITS MSM plateau
N_MACRO  = 5     # reduce to 3 if CK test fails
```

CK test: dashed (predicted) and solid (re-estimated) lines must overlap within error bars.

**Output:** `analysis/plots/its_msm.png`, `analysis/plots/ck_test.png`, `analysis/msm.pyemma`

---

### Step 4 — SESCA CD forward modelling

**Script:** `scripts/04_sesca.py`

Run SESCA externally on each representative PDB (produced by Step 5), then:

```bash
for i in 0 1 2 3 4; do
    python /path/to/sesca_main.py \
        -pdb analysis/representatives/macro_${i}.pdb \
        -basis DB5 \
        -output analysis/sesca/macro_${i}_cd.dat
done

python scripts/04_sesca.py
```

**D-SPR interpretation:**

| D-SPR | Quality |
|---|---|
| < 0.15 | Excellent |
| 0.15 – 0.25 | Good |
| 0.25 – 0.40 | Moderate |
| > 0.40 | Poor — revisit macrostates or force field |

**Output:** `analysis/plots/cd_comparison.png`, `analysis/dspr_value.txt`

---

### Step 5 — Representative structures

**Script:** `scripts/05_representatives.py`

Finds the frame closest to the TICA-space centroid for each macrostate and writes one PDB per macrostate.

```bash
python scripts/05_representatives.py
```

**Output:** `analysis/representatives/macro_N.pdb`, `analysis/representatives/summary.txt`

---

## Adapting to a new system

All system-specific configuration lives in the `SYSTEMS` dict at the top of each script. To run on a new IDP:

**1.** Edit `SYSTEMS` in every script:
```python
# Tau(272-314) example
SYSTEMS = {
    "tau_ext":   ("trajs/tau_ext.pdb",   "trajs/tau_ext.xtc"),
    "tau_helix": ("trajs/tau_helix.pdb", "trajs/tau_helix.xtc"),
    "tau_nmr":   ("trajs/tau_nmr.pdb",   "trajs/tau_nmr.xtc"),
}
```

**2.** Update `EXP_DATA_PATH` in `04_sesca.py`:
```python
EXP_DATA_PATH = "data/tau_cd_exp.dat"
```

**3.** Re-tune `LAG_TICA_FINAL`, `LAG_MSM`, `N_MACRO` from the ITS plots — do not reuse values from a different system.

**4.** Run from Step 0.

Nothing else needs to change.

---

## Outputs

| File | Description |
|---|---|
| `analysis/plots/pca_joint.png` | 2D PCA landscape coloured by starting structure |
| `analysis/plots/its_tica.png` | ITS vs TICA lagtime |
| `analysis/plots/its_msm.png` | ITS vs MSM lagtime |
| `analysis/plots/ck_test.png` | Chapman-Kolmogorov test |
| `analysis/plots/cd_comparison.png` | SESCA ensemble vs experimental CD |
| `analysis/dspr_value.txt` | D-SPR metric + quality label |
| `analysis/representatives/macro_N.pdb` | Representative structure per macrostate |
| `analysis/msm.pyemma` | Serialised Bayesian MSM |
| `analysis/tica.pyemma` | Serialised TICA model |

---

## Troubleshooting

**`ValueError: number of atoms mismatch`**  
→ Topologies not identical. Re-generate with `gmx trjconv -pbc mol` using the same `.tpr`.

**`MSM has too few active states`**  
→ Increase `N_CLUSTERS` or decrease `LAG_MSM`. Check `active_count_fraction > 0.90`.

**CK test fails visually**  
→ Reduce `N_MACRO` to 3, or increase `LAG_MSM`.

**SESCA gives flat/zero spectrum**  
→ PDB must contain full backbone atoms (N, Cα, C, O) as ATOM records. No HETATM, no missing residues.

**Cosine content > 0.5**  
→ Simulations not converged. Report as limitation; consider REST2 or replica-exchange to improve sampling.

**REST2 replica swap rate < 20%**  
→ Temperature ladder spacing too wide. Re-generate with `gmx_mpi mdrun -replex` and inspect `replica_temp.xvg`.

---

## References

1. Bowman, G. R., Pande, V. S. & Noé, F. (2014). *An Introduction to Markov State Models*. Springer.
2. Husic, B. E. & Pande, V. S. (2018). Markov State Models: From an Art to a Science. *JACS*, 140, 2386–2396.
3. Pérez-Hernández, G. et al. (2013). Identification of slow molecular order parameters for Markov model construction. *J. Chem. Phys.*, 139, 015102.
4. Scherer, M. K. et al. (2015). PyEMMA 2. *JCTC*, 11, 5525–5542.
5. Sieradzan, A. K. et al. (SESCA-2025). Predicting protein CD spectra from structure ensembles. *NAR Web Server Issue*.
6. Barrow, C. J. & Zagorski, M. G. (1991). Solution structures of beta peptide. *Science*, 253, 179–182.
7. Mukrasch, M. D. et al. (2009). Structural polymorphism of Tau. *PLOS Biology*, 7, e34.
8. von Bergen, M. et al. (2001). Tau aggregation is driven by a transition from random coil to beta-sheet structure. *Biochim. Biophys. Acta*, 1739, 158–166.
9. Wang, L.-P. et al. (2017). Building Force Fields: An Automatic, Systematic, and Reproducible Approach. *J. Phys. Chem. Lett.*, 8, 3707–3713. *(a99SB-disp)*
10. Piana, S. et al. (2015). Water dispersion interactions strongly influence simulated structural properties of disordered protein states. *J. Phys. Chem. B*, 119, 5113–5123. *(TIP4P-D)*

---

## Acknowledgements

We sincerely thank **CINECA and the Italian SuperComputing Resource Allocation (ISCRA) program** for making this work computationally feasible. The scale of enhanced-sampling simulations required by this pipeline — 16-replica REST2 runs at 500 ns/replica for multiple Aβ variants — would not have been achievable without access to leadership-class HPC infrastructure. The availability of LEONARDO BOOSTER has been decisive in enabling a rigorous, ensemble-first characterisation of the Aβ(1-40)/(1-42) conformational landscape.

