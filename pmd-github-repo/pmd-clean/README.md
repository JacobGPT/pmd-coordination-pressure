# Pressure Makes Diamonds: Coordination Pressure as a Consciousness Metric

**Preprint:* https://zenodo.org/records/19393842

**Author:** Jacob Beach (Independent Research)

## Overview

This repository contains the analysis code for:

> Beach, J. (2026). Coordination Pressure Between Brain Networks Predicts Behavioral Slowing and Tracks Consciousness State Transitions Under Propofol Sedation.

We introduce coordination pressure (Φ\*\_PD), a metric derived from statistical mechanics that quantifies the difficulty of reconciling conflicting policies between independently optimized brain networks. The metric is validated on two independent fMRI datasets:

- **Behavioral prediction** (UCLA CNP, ds000030, N=16): Φ\*\_PD predicts trial-level reaction time (p=0.014), generalizes across subjects (permutation p<0.001), and adds beyond entropy and variance (p=0.001, 16/16 subjects positive).
- **Consciousness state transitions** (Michigan Human Anesthesia, ds006623, N=11): Φ\*\_PD drops 2.69× at loss of responsiveness under propofol (p=0.005, d=1.12).

## Repository Structure

```
├── README.md
├── requirements.txt
├── LICENSE
├── analyses/
│   ├── behavioral/          # Primary behavioral results (Dataset 1)
│   │   ├── 01_level3_pipeline.py        # Core Level 3 decoded-policy pipeline
│   │   ├── 02_group_analysis.py         # Group-level task comparisons
│   │   ├── 03_rt_prediction.py          # Trial-level RT prediction
│   │   ├── 04_model_comparison.py       # Model comparison & quintile analysis
│   │   ├── 05_cross_subject.py          # Cross-subject generalization (z-scored)
│   │   ├── 06_variance_decomposition.py # ΔR² beyond 5 baseline predictors
│   │   ├── 07_eigenvector_structure.py  # Split-half eigenvector independence
│   │   └── 08_math_predictions.py       # Mathematical structure tests
│   │
│   ├── consciousness/      # Consciousness state transitions (Dataset 2)
│   │   ├── 01_propofol_analysis.py      # Pre-LOR vs Post-LOR (headline result)
│   │   ├── 02_state_prediction.py       # Multi-state consciousness prediction
│   │   └── 03_level3_consciousness.py   # Level 3 consciousness pipeline
│   │
│   ├── robustness/          # Vulnerability analyses & parameter robustness
│   │   ├── 01_vulnerability_analysis.py # Difficulty dissociations
│   │   ├── 02_multiplicative_necessity.py # C alone vs I×C×G
│   │   ├── 03_parameter_robustness.py   # 10-configuration robustness battery
│   │   └── 04_followup_analyses.py      # Additional robustness checks
│   │
│   └── exploratory/         # Exploratory analyses (reported but not primary)
│       ├── anesthesia_effective_conflict.py
│       ├── anesthesia_info_theoretic.py
│       ├── psychedelic_effective_conflict.py
│       ├── psychedelic_info_theoretic.py
│       ├── hcp_single_subject.py
│       ├── single_subject_deep_dive.py
│       ├── sleep_analysis.py
│       ├── task_switching.py
│       ├── level2a_vs_level3.py
│       ├── level2b_proxy.py
│       └── dynamic_level2a.py
```

## Datasets

All data are publicly available on OpenNeuro. No data are included in this repository.

| Dataset | OpenNeuro ID | Description | N |
|---------|-------------|-------------|---|
| UCLA CNP | [ds000030](https://openneuro.org/datasets/ds000030) | Stop-signal, BART, SCAP tasks | 16 |
| Michigan Anesthesia | [ds006623](https://openneuro.org/datasets/ds006623) | Propofol sedation with graded LOR | 11 |

## Requirements

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+. See `requirements.txt` for package versions.

## Reproducing Results

### Behavioral prediction (Table 1, Sections 4.1–4.6 in paper)

```bash
# Download ds000030 from OpenNeuro, then:
python analyses/behavioral/01_level3_pipeline.py --data-dir /path/to/ds000030
python analyses/behavioral/02_group_analysis.py
python analyses/behavioral/03_rt_prediction.py
python analyses/behavioral/05_cross_subject.py
python analyses/behavioral/06_variance_decomposition.py
python analyses/behavioral/07_eigenvector_structure.py
```

### Consciousness state transitions (Table 2, Section 5 in paper)

```bash
# Download ds006623 from OpenNeuro, then:
python analyses/consciousness/01_propofol_analysis.py --data-dir /path/to/ds006623
```

### Vulnerability analyses (Section 7 in paper)

```bash
python analyses/robustness/01_vulnerability_analysis.py
python analyses/robustness/02_multiplicative_necessity.py
python analyses/robustness/03_parameter_robustness.py
```

## Key Results

| Test | Result | Statistic |
|------|--------|-----------|
| STOP > GO (within-task) | 9/10 correct | p=0.014, d=0.96 |
| StopSignal > SCAP | 10/10 correct | p=0.0001, d=2.22 |
| Φ\*\_PD predicts RT | 12/16 positive | p=0.014 |
| Φ\*\_PD > entropy+variance | 16/16 positive | p=0.001 |
| Cross-subject generalization | 100% splits positive | perm p<0.001 |
| Pre-LOR > Post-LOR | 9/11 correct, 2.69× | p=0.005, d=1.12 |
| Eigenvector alignment | r=0.987 (split-half) | p≈0 |
| Morse landscape dominance | Ratio=89.1 | Single basin |

## The Metric

Coordination pressure is defined as:

```
Φ*_PD = (2 / N(N-1)) · Σ_{i<j} I_ij · C_ij · G_ij
```

where:
- **I\_ij**: Integration strength (transfer entropy between networks i and j)
- **C\_ij**: Conflict (Jensen–Shannon divergence between decoded policy vectors)
- **G\_ij**: Organization gate (filters for temporally stable, bidirectionally coupled conflict)

The multiplicative I × C × G structure is not decorative — C alone misclassifies fragmentation states (d=−2.29, wrong direction), while the full metric classifies correctly (d=+6.61).

## Companion Papers

- **Paper 2:** "The Universal Coordination Law" — mathematical extensions
- **Paper 3:** "The Self-Referential Closure" — the hard problem
- **Mathematical Atlas:** Complete derivation reference

## Citation

```bibtex
@article{beach2026coordination,
  title={Coordination Pressure Between Brain Networks Predicts Behavioral
         Slowing and Tracks Consciousness State Transitions Under Propofol
         Sedation},
  author={Beach, Jacob},
  year={2026},
  journal={zenodo},
  doi={https://doi.org/10.5281/zenodo.19393841}
}
```

## License

MIT License. See `LICENSE`.
