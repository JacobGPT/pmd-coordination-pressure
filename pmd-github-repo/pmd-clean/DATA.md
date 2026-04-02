# Data Access and Reproduction Guide

## Datasets

This project uses two publicly available fMRI datasets from OpenNeuro. No data are included in this repository.

### Dataset 1: UCLA Consortium for Neuropsychiatric Phenomics

- **OpenNeuro ID:** [ds000030](https://openneuro.org/datasets/ds000030)
- **Used for:** Behavioral prediction (Sections 4.1–4.6 in paper)
- **Subjects:** 16 healthy controls
- **Tasks:** Stop-Signal, BART, SCAP, Task Switching
- **Preprocessing:** fMRIPrep, MNI152 standard space

**Download:**
```bash
# Install the OpenNeuro CLI
npm install -g openneuro-cli

# Download preprocessed derivatives
openneuro download ds000030 /path/to/ds000030
```

### Dataset 2: Michigan Human Anesthesia fMRI

- **OpenNeuro ID:** [ds006623](https://openneuro.org/datasets/ds006623)
- **Used for:** Consciousness state transitions (Section 5 in paper)
- **Subjects:** 11 with complete data (26 total)
- **Conditions:** Baseline, Light Sedation, Deep Sedation, Recovery
- **Preprocessing:** fMRIPrep, MNI152NLin2009cAsym, 4mm resolution

**Download:**
```bash
openneuro download ds006623 /path/to/ds006623
```

**Note:** The full ds006623 download is large (~120K files). If the bulk download fails, use the included utility scripts to download subject-by-subject. See `analyses/exploratory/` for download helpers.

## Expected Directory Layout

```
/path/to/data/
├── ds000030/
│   └── derivatives/
│       └── fmriprep/
│           ├── sub-10159/
│           ├── sub-10171/
│           └── ...
└── ds006623/
    └── derivatives/
        └── fmriprep/
            ├── sub-02/
            ├── sub-03/
            └── ...
```

## Parcellation

All analyses use the **Yeo 7-network atlas** (Yeo et al., 2011), which is included in the `nilearn` package and does not need to be downloaded separately.

## Which Scripts Produce Which Results

| Paper Section | Result | Script |
|---------------|--------|--------|
| 4.1 Task discrimination | STOP > GO, cross-task | `behavioral/01_level3_pipeline.py` → `behavioral/02_group_analysis.py` |
| 4.2 RT prediction | Trial-level r = +0.093 | `behavioral/03_rt_prediction.py` |
| 4.2 Quintile analysis | Monotonic slope | `behavioral/04_model_comparison.py` |
| 4.3 Cross-subject | Permutation p < 0.001 | `behavioral/05_cross_subject.py` |
| 4.5 Variance decomposition | ΔR² = +0.017, 16/16 | `behavioral/06_variance_decomposition.py` |
| 4.6 Eigenvector structure | Split-half r = 0.987 | `behavioral/07_eigenvector_structure.py` |
| 5.1 Propofol LOR | 2.69×, p = 0.005 | `consciousness/01_propofol_analysis.py` |
| 6 Math predictions | Eigenvector, transport, Morse | `behavioral/08_math_predictions.py` |
| 7.1–7.3 Vulnerability | Dissociations, ablations | `robustness/01_vulnerability_analysis.py` → `robustness/02_multiplicative_necessity.py` |
| 7.2 Parameter robustness | 9/10 configs | `robustness/03_parameter_robustness.py` |

## Intermediate Outputs

Scripts write intermediate results (CSVs, JSON summaries) to the current working directory. These are not tracked in git. To reproduce from scratch, run the scripts in the numbered order within each subdirectory.
