#!/usr/bin/env bash
# ============================================================================
# reproduce_main_results.sh
#
# Reproduces the headline results from:
#   Beach (2026). "Coordination Pressure Between Brain Networks Predicts
#   Behavioral Slowing and Tracks Consciousness State Transitions Under
#   Propofol Sedation"
#
# Prerequisites:
#   - Python 3.10+ with packages from requirements.txt
#   - Downloaded datasets (see DATA.md)
#
# Usage:
#   ./reproduce_main_results.sh /path/to/ds000030 /path/to/ds006623
#
# ============================================================================

set -e

DS000030="${1:?Usage: $0 /path/to/ds000030 /path/to/ds006623}"
DS006623="${2:?Usage: $0 /path/to/ds000030 /path/to/ds006623}"

echo "============================================================"
echo "  PRESSURE MAKES DIAMONDS — REPRODUCE MAIN RESULTS"
echo "============================================================"
echo ""
echo "  Dataset 1 (behavioral): $DS000030"
echo "  Dataset 2 (consciousness): $DS006623"
echo ""

# --- BEHAVIORAL RESULTS (Paper Sections 4.1–4.6) ---

echo "============================================================"
echo "  STEP 1/7: Level 3 Pipeline (core metric computation)"
echo "============================================================"
python analyses/behavioral/01_level3_pipeline.py --data-dir "$DS000030"

echo ""
echo "============================================================"
echo "  STEP 2/7: Group Analysis (task discrimination, Table 1)"
echo "============================================================"
python analyses/behavioral/02_group_analysis.py

echo ""
echo "============================================================"
echo "  STEP 3/7: RT Prediction (trial-level, Section 4.2)"
echo "============================================================"
python analyses/behavioral/03_rt_prediction.py

echo ""
echo "============================================================"
echo "  STEP 4/7: Cross-Subject Generalization (Section 4.3)"
echo "============================================================"
python analyses/behavioral/05_cross_subject.py

echo ""
echo "============================================================"
echo "  STEP 5/7: Variance Decomposition (Section 4.5)"
echo "============================================================"
python analyses/behavioral/06_variance_decomposition.py

echo ""
echo "============================================================"
echo "  STEP 6/7: Eigenvector Structure (Section 4.6)"
echo "============================================================"
python analyses/behavioral/07_eigenvector_structure.py

# --- CONSCIOUSNESS RESULTS (Paper Section 5) ---

echo ""
echo "============================================================"
echo "  STEP 7/7: Propofol Consciousness Transition (Section 5)"
echo "============================================================"
python analyses/consciousness/01_propofol_analysis.py --data-dir "$DS006623"

echo ""
echo "============================================================"
echo "  COMPLETE"
echo "============================================================"
echo ""
echo "  All headline results reproduced."
echo "  For robustness analyses (Section 7), run:"
echo "    python analyses/robustness/01_vulnerability_analysis.py"
echo "    python analyses/robustness/02_multiplicative_necessity.py"
echo "    python analyses/robustness/03_parameter_robustness.py"
echo ""
echo "  For mathematical structure tests (Section 8), run:"
echo "    python analyses/behavioral/08_math_predictions.py"
echo ""
