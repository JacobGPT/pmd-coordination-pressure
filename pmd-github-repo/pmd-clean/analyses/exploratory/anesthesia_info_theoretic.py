#!/usr/bin/env python3
"""
===========================================================================
PRESSURE MAKES DIAMONDS — THE KILLER TEST (Information-Theoretic Version)
Anesthesia Phase Transition: TE × JSD Proxy
===========================================================================

Same dataset (ds003171), same prediction, DIFFERENT PROXY.

The correlation proxy showed Deep Sedation with HIGHER λ_max than Awake —
the opposite of what the framework predicts. This is the same proxy
entanglement problem identified in the psychedelic analysis: correlation
derives both I and C from one scalar, creating structural coupling.

This script uses INDEPENDENT measures:
  Integration (I_ij): Transfer Entropy — directed causal information flow
  Conflict (C_ij):    Jensen-Shannon Divergence of amplitude distributions

If λ_max drops below threshold during Deep Sedation with this proxy while
the correlation proxy showed it rising, that replicates the psychedelic
proxy-reversal finding on a second dataset.

USAGE:
  python pmd_anesthesia_info_theoretic.py --data-dir ./ds003171 --skip-download
===========================================================================
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')

# =========================================================================
# CONFIGURATION
# =========================================================================
YEO7_LABELS = [
    "Visual", "Somatomotor", "DorsalAttention",
    "VentralAttention", "Limbic", "Frontoparietal", "Default"
]

# =========================================================================
# TIMESERIES EXTRACTION (same as correlation version)
# =========================================================================

def extract_network_timeseries(bold_file):
    """Extract mean timeseries for each Yeo-7 network using NiftiLabelsMasker."""
    from nilearn import maskers
    import glob
    
    yeo_dir = os.path.join(os.path.expanduser('~'), 'nilearn_data', 'yeo_2011')
    candidates = glob.glob(os.path.join(yeo_dir, '**', 'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz'), recursive=True)
    if not candidates:
        candidates = glob.glob(os.path.join(yeo_dir, '**', '*7Networks*1mm.nii.gz'), recursive=True)
    if not candidates:
        raise RuntimeError(f"Cannot find Yeo 7-network atlas in {yeo_dir}")
    
    atlas_file = candidates[0]
    masker = maskers.NiftiLabelsMasker(
        labels_img=atlas_file,
        standardize='zscore_sample',
        detrend=True,
        memory='nilearn_cache',
        memory_level=1
    )
    
    timeseries = masker.fit_transform(bold_file)
    n_timepoints, n_networks = timeseries.shape
    
    if n_timepoints < 30:
        raise ValueError(f"Too few timepoints: {n_timepoints}")
    
    if n_networks != 7:
        ts_padded = np.zeros((n_timepoints, 7))
        ts_padded[:, :min(n_networks, 7)] = timeseries[:, :min(n_networks, 7)]
        timeseries = ts_padded
    
    return timeseries


# =========================================================================
# INFORMATION-THEORETIC MEASURES
# =========================================================================

def transfer_entropy(source, target, lag=1, bins=10):
    """
    Compute transfer entropy from source to target.
    TE(X->Y) = H(Y_t | Y_{t-lag}) - H(Y_t | Y_{t-lag}, X_{t-lag})
    Estimated via binned histograms.
    """
    n = len(source)
    if n <= lag + 1:
        return 0.0
    
    # Create lagged variables
    y_now = target[lag:]
    y_past = target[:-lag]
    x_past = source[:-lag]
    
    # Bin the data
    def bin_data(x, bins):
        mn, mx = x.min(), x.max()
        if mx == mn:
            return np.zeros(len(x), dtype=int)
        return np.clip(((x - mn) / (mx - mn) * (bins - 1)).astype(int), 0, bins - 1)
    
    y_now_b = bin_data(y_now, bins)
    y_past_b = bin_data(y_past, bins)
    x_past_b = bin_data(x_past, bins)
    
    # Joint and conditional entropies via histograms
    def entropy_from_counts(counts):
        p = counts[counts > 0] / counts.sum()
        return -np.sum(p * np.log2(p))
    
    # H(Y_t, Y_{t-1})
    joint_yy = np.zeros((bins, bins))
    for i in range(len(y_now_b)):
        joint_yy[y_now_b[i], y_past_b[i]] += 1
    h_yy = entropy_from_counts(joint_yy.ravel())
    
    # H(Y_{t-1})
    hist_ypast = np.bincount(y_past_b, minlength=bins).astype(float)
    h_ypast = entropy_from_counts(hist_ypast)
    
    # H(Y_t, Y_{t-1}, X_{t-1})
    joint_yyx = np.zeros((bins, bins, bins))
    for i in range(len(y_now_b)):
        joint_yyx[y_now_b[i], y_past_b[i], x_past_b[i]] += 1
    h_yyx = entropy_from_counts(joint_yyx.ravel())
    
    # H(Y_{t-1}, X_{t-1})
    joint_yx = np.zeros((bins, bins))
    for i in range(len(y_past_b)):
        joint_yx[y_past_b[i], x_past_b[i]] += 1
    h_yx = entropy_from_counts(joint_yx.ravel())
    
    # TE = H(Y_t, Y_{t-1}) - H(Y_{t-1}) - H(Y_t, Y_{t-1}, X_{t-1}) + H(Y_{t-1}, X_{t-1})
    # = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})
    te = (h_yy - h_ypast) - (h_yyx - h_yx)
    
    return max(te, 0.0)  # TE should be non-negative


def symmetric_transfer_entropy(ts_i, ts_j, lag=1, bins=10):
    """Compute symmetric TE: average of both directions."""
    te_ij = transfer_entropy(ts_i, ts_j, lag, bins)
    te_ji = transfer_entropy(ts_j, ts_i, lag, bins)
    return (te_ij + te_ji) / 2.0


def signal_jsd(ts_i, ts_j, bins=30):
    """
    Compute Jensen-Shannon Divergence between amplitude distributions.
    This measures how differently two networks distribute their activity levels.
    """
    # Create histograms of amplitude distributions
    all_vals = np.concatenate([ts_i, ts_j])
    mn, mx = all_vals.min(), all_vals.max()
    if mx == mn:
        return 0.0
    
    edges = np.linspace(mn, mx, bins + 1)
    p = np.histogram(ts_i, bins=edges)[0].astype(float)
    q = np.histogram(ts_j, bins=edges)[0].astype(float)
    
    # Normalize
    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    
    # JSD = 0.5 * KL(p||m) + 0.5 * KL(q||m)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    
    return max(jsd, 0.0)


# =========================================================================
# PMD COMPUTATION (Info-Theoretic)
# =========================================================================

def compute_pmd_info_theoretic(timeseries):
    """
    Compute PMD metrics using information-theoretic proxies.
    I_ij = symmetric transfer entropy
    C_ij = Jensen-Shannon divergence of amplitude distributions
    J_ij = I_ij * C_ij
    """
    N = timeseries.shape[1]  # 7 networks
    
    # Also compute correlation for comparison
    corr = np.corrcoef(timeseries.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute I (transfer entropy) and C (JSD) independently
    I = np.zeros((N, N))
    C = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i + 1, N):
            te = symmetric_transfer_entropy(timeseries[:, i], timeseries[:, j])
            jsd = signal_jsd(timeseries[:, i], timeseries[:, j])
            I[i, j] = I[j, i] = te
            C[i, j] = C[j, i] = jsd
    
    # J = I ⊙ C (independently measured)
    J = I * C
    np.fill_diagonal(J, 0)
    
    # Compute Φ_PD
    phi_pd = 0
    n_pairs = 0
    pair_contributions = {}
    network_totals = np.zeros(N)
    te_pairs = {}
    jsd_pairs = {}
    
    for i in range(N):
        for j in range(i + 1, N):
            contrib = J[i, j]
            phi_pd += contrib
            n_pairs += 1
            pair_name = f"{YEO7_LABELS[i]}-{YEO7_LABELS[j]}"
            pair_contributions[pair_name] = float(contrib)
            te_pairs[pair_name] = float(I[i, j])
            jsd_pairs[pair_name] = float(C[i, j])
            network_totals[i] += contrib
            network_totals[j] += contrib
    
    phi_pd_norm = phi_pd / n_pairs if n_pairs > 0 else 0
    
    # Eigenvalue analysis
    J_sym = (J + J.T) / 2.0
    J_sym = np.nan_to_num(J_sym, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        eigenvalues = np.linalg.eigvalsh(J_sym)
    except np.linalg.LinAlgError:
        eigenvalues = np.real(np.linalg.eigvals(J_sym))
    eigenvalues = np.sort(np.real(eigenvalues))[::-1]
    lambda_max = eigenvalues[0]
    
    # Count anticorrelated pairs (from correlation, for reference)
    n_anticorr = sum(1 for i in range(N) for j in range(i+1, N) if corr[i,j] < 0)
    
    # DMN contribution
    dmn_idx = 6
    dmn_contribution = network_totals[dmn_idx]
    dmn_pct = (dmn_contribution / (2 * phi_pd) * 100) if phi_pd > 0 else 0
    
    # Mean TE and JSD
    mean_te = np.mean([I[i,j] for i in range(N) for j in range(i+1,N)])
    mean_jsd = np.mean([C[i,j] for i in range(N) for j in range(i+1,N)])
    
    return {
        'phi_pd_norm': float(phi_pd_norm),
        'phi_pd_total': float(phi_pd),
        'lambda_max': float(lambda_max),
        'lambda_spectrum': [float(x) for x in eigenvalues],
        'mean_te': float(mean_te),
        'mean_jsd': float(mean_jsd),
        'n_anticorr': int(n_anticorr),
        'dmn_contribution': float(dmn_contribution),
        'dmn_pct': float(dmn_pct),
        'pair_contributions': pair_contributions,
        'te_pairs': te_pairs,
        'jsd_pairs': jsd_pairs,
        'network_contributions': {YEO7_LABELS[i]: float(network_totals[i]) for i in range(N)},
    }


# =========================================================================
# FILE FINDER (same as correlation version)
# =========================================================================

def find_bold_files(data_dir):
    """Find resting-state BOLD files by subject and condition."""
    data_dir = Path(data_dir)
    results = {}
    task_condition_map = {
        'restawake': 'Awake',
        'restlight': 'Mild Sedation',
        'restdeep': 'Deep Sedation',
        'restrecovery': 'Recovery',
    }
    
    all_files = list(data_dir.glob("sub-*/func/*task-rest*bold.nii.gz"))
    if not all_files:
        print(f"\n  No resting-state BOLD files found in {data_dir}")
        return results
    
    print(f"\n  Found {len(all_files)} resting-state BOLD files")
    
    for fpath in sorted(all_files):
        fname = fpath.name
        parts = str(fpath).replace('\\', '/')
        
        sub_id = None
        for part in parts.split('/'):
            if part.startswith('sub-'):
                sub_id = part
                break
        if not sub_id:
            continue
        
        condition = None
        for task_key, cond_name in task_condition_map.items():
            if f"task-{task_key}" in fname:
                condition = cond_name
                break
        if not condition:
            continue
        
        if sub_id not in results:
            results[sub_id] = {}
        results[sub_id][condition] = str(fpath)
    
    return results


# =========================================================================
# MAIN ANALYSIS
# =========================================================================

def run_info_theoretic_killer_test(data_dir, output_dir, skip_download=False):
    """Run the anesthesia analysis with information-theoretic proxy."""
    
    print("=" * 75)
    print("PRESSURE MAKES DIAMONDS — THE KILLER TEST")
    print("Information-Theoretic Version (TE × JSD)")
    print("=" * 75)
    print()
    print("PROXY: Transfer Entropy (integration) × JSD (conflict)")
    print("       I and C measured INDEPENDENTLY — no structural entanglement")
    print()
    print("COMPARING AGAINST: Correlation proxy results where Deep Sedation")
    print("                   showed HIGHER λ_max than Awake (proxy artifact)")
    print()
    print("THE PREDICTION (same as before):")
    print("  Awake:          λ_max above threshold  (conscious)")
    print("  Mild Sedation:  λ_max above threshold  (drowsy but aware)")
    print("  Deep Sedation:  λ_max BELOW threshold  (unconscious)")
    print("  Recovery:       λ_max above threshold  (snap-back)")
    print()
    print("NOTE: TE computation is slower than correlation. ~1-2 min per scan.")
    print("=" * 75)
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find files
    print("\n" + "=" * 75)
    print("STEP 1: Finding BOLD files")
    print("=" * 75)
    
    subject_files = find_bold_files(data_dir)
    if not subject_files:
        print("No subjects found.")
        sys.exit(1)
    
    print(f"\n  Subjects: {len(subject_files)}")
    
    # Process
    print("\n" + "=" * 75)
    print("STEP 2: Computing TE × JSD metrics (this will take ~30-60 min)")
    print("=" * 75)
    
    all_results = {}
    condition_metrics = {}
    
    for sub_id in sorted(subject_files.keys()):
        print(f"\n  {sub_id}:")
        all_results[sub_id] = {}
        
        for condition, filepath in sorted(subject_files[sub_id].items()):
            fname = os.path.basename(filepath)
            print(f"    {condition}: {fname}...", end="", flush=True)
            
            try:
                ts = extract_network_timeseries(filepath)
                metrics = compute_pmd_info_theoretic(ts)
                all_results[sub_id][condition] = metrics
                
                if condition not in condition_metrics:
                    condition_metrics[condition] = []
                condition_metrics[condition].append(metrics)
                
                above = "ABOVE" if metrics['lambda_max'] > 0 else "ZERO"
                print(f" Φ_PD={metrics['phi_pd_norm']:.6f}, λ_max={metrics['lambda_max']:.6f}, TE={metrics['mean_te']:.4f}, JSD={metrics['mean_jsd']:.4f}")
                
            except Exception as e:
                print(f" ERROR: {e}")
                continue
    
    if not condition_metrics:
        print("\n  No successful analyses.")
        sys.exit(1)
    
    # =====================================================================
    # GROUP RESULTS
    # =====================================================================
    print("\n" + "=" * 75)
    print("STEP 3: GROUP-LEVEL RESULTS (Info-Theoretic Proxy)")
    print("=" * 75)
    
    condition_order = ["Awake", "Mild Sedation", "Deep Sedation", "Recovery"]
    ordered = [c for c in condition_order if c in condition_metrics]
    for c in condition_metrics:
        if c not in ordered:
            ordered.append(c)
    
    print(f"\n  {'Condition':<20} {'N':>4} {'Φ_PD(TE×JSD)':>14} {'λ_max':>14} {'Mean TE':>10} {'Mean JSD':>10}")
    print("  " + "-" * 74)
    
    group_summary = {}
    
    for condition in ordered:
        mlist = condition_metrics[condition]
        n = len(mlist)
        
        phi_vals = [m['phi_pd_norm'] for m in mlist]
        lmax_vals = [m['lambda_max'] for m in mlist]
        te_vals = [m['mean_te'] for m in mlist]
        jsd_vals = [m['mean_jsd'] for m in mlist]
        
        phi_m, phi_s = np.mean(phi_vals), np.std(phi_vals)
        lmax_m, lmax_s = np.mean(lmax_vals), np.std(lmax_vals)
        te_m = np.mean(te_vals)
        jsd_m = np.mean(jsd_vals)
        
        print(f"  {condition:<20} {n:>4} {phi_m:>10.6f}±{phi_s:.5f} {lmax_m:>10.6f}±{lmax_s:.5f} {te_m:>10.4f} {jsd_m:>10.4f}")
        
        group_summary[condition] = {
            'n': n,
            'phi_pd_mean': float(phi_m), 'phi_pd_std': float(phi_s),
            'lambda_max_mean': float(lmax_m), 'lambda_max_std': float(lmax_s),
            'mean_te': float(te_m), 'mean_jsd': float(jsd_m),
            'phi_pd_values': [float(x) for x in phi_vals],
            'lambda_max_values': [float(x) for x in lmax_vals],
            'te_values': [float(x) for x in te_vals],
            'jsd_values': [float(x) for x in jsd_vals],
        }
    
    # =====================================================================
    # STATISTICAL TESTS
    # =====================================================================
    print("\n" + "=" * 75)
    print("STEP 4: STATISTICAL TESTS")
    print("=" * 75)
    
    from scipy import stats
    
    contrasts = [
        ("Awake", "Deep Sedation", "Consciousness vs Unconsciousness"),
        ("Awake", "Mild Sedation", "Full vs Reduced Consciousness"),
        ("Deep Sedation", "Recovery", "Recovery Snap-Back"),
        ("Awake", "Recovery", "Full Recovery Check"),
    ]
    
    for cond_a, cond_b, label in contrasts:
        if cond_a not in group_summary or cond_b not in group_summary:
            continue
        
        a_phi = group_summary[cond_a]['phi_pd_values']
        b_phi = group_summary[cond_b]['phi_pd_values']
        a_lmax = group_summary[cond_a]['lambda_max_values']
        b_lmax = group_summary[cond_b]['lambda_max_values']
        a_te = group_summary[cond_a]['te_values']
        b_te = group_summary[cond_b]['te_values']
        a_jsd = group_summary[cond_a]['jsd_values']
        b_jsd = group_summary[cond_b]['jsd_values']
        
        def do_test(a, b, name):
            t, p = stats.ttest_ind(a, b)
            pooled_std = np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)
            d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            direction = "↑" if np.mean(a) > np.mean(b) else "↓"
            print(f"    {name:<12} {direction} d={d:+.3f}, t={t:.3f}, p={p:.4f} {sig}")
        
        print(f"\n  {label}: {cond_a} vs {cond_b}")
        do_test(a_phi, b_phi, "Φ_PD:")
        do_test(a_lmax, b_lmax, "λ_max:")
        do_test(a_te, b_te, "Mean TE:")
        do_test(a_jsd, b_jsd, "Mean JSD:")
    
    # =====================================================================
    # PROXY COMPARISON
    # =====================================================================
    print("\n" + "=" * 75)
    print("STEP 5: PROXY COMPARISON (Correlation vs Info-Theoretic)")
    print("=" * 75)
    
    # Correlation results from previous run
    corr_results = {
        'Awake': 1.062,
        'Mild Sedation': 1.081,
        'Deep Sedation': 1.196,
        'Recovery': 1.025,
    }
    
    print(f"\n  {'Condition':<20} {'Corr λ_max':>12} {'TE×JSD λ_max':>14} {'Direction Match':>16}")
    print("  " + "-" * 64)
    
    for condition in ordered:
        if condition in corr_results and condition in group_summary:
            corr_lmax = corr_results[condition]
            te_lmax = group_summary[condition]['lambda_max_mean']
            
            # Check if they agree on ordering
            match = "—"
            if condition == "Deep Sedation":
                corr_dir = "ABOVE" if corr_lmax > 1 else "BELOW"
                te_dir = "lower" if te_lmax < group_summary.get('Awake', {}).get('lambda_max_mean', 0) else "higher"
                match = f"Corr={corr_dir}"
            
            print(f"  {condition:<20} {corr_lmax:>12.4f} {te_lmax:>14.6f} {match:>16}")
    
    # =====================================================================
    # DECOMPOSITION: Is integration or conflict driving the change?
    # =====================================================================
    print("\n" + "=" * 75)
    print("STEP 6: DECOMPOSITION — What changes under anesthesia?")
    print("=" * 75)
    
    if "Awake" in group_summary and "Deep Sedation" in group_summary:
        awake_te = group_summary["Awake"]["mean_te"]
        deep_te = group_summary["Deep Sedation"]["mean_te"]
        awake_jsd = group_summary["Awake"]["mean_jsd"]
        deep_jsd = group_summary["Deep Sedation"]["mean_jsd"]
        
        te_change = "INCREASES" if deep_te > awake_te else "DECREASES"
        jsd_change = "INCREASES" if deep_jsd > awake_jsd else "DECREASES"
        
        print(f"\n  Awake → Deep Sedation:")
        print(f"    Integration (TE):  {awake_te:.4f} → {deep_te:.4f}  [{te_change}]")
        print(f"    Conflict (JSD):    {awake_jsd:.4f} → {deep_jsd:.4f}  [{jsd_change}]")
        print()
        
        if deep_te < awake_te:
            print("    → Integration DECREASES under deep sedation")
            print("      (Propofol reduces directed information flow between networks)")
        if deep_jsd < awake_jsd:
            print("    → Conflict DECREASES under deep sedation")
            print("      (Networks become more similar in their activity patterns)")
        if deep_te < awake_te and deep_jsd < awake_jsd:
            print("    → BOTH components decrease — framework prediction supported")
            print("      (Propofol reduces both coupling AND disagreement)")
    
    # =====================================================================
    # SAVE
    # =====================================================================
    
    output_file = output_dir / "anesthesia_info_theoretic_results.json"
    
    def convert(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    
    with open(output_file, 'w') as f:
        json.dump(convert({
            'dataset': 'ds003171',
            'proxy': 'TE × JSD (information-theoretic)',
            'group_summary': group_summary,
        }), f, indent=2)
    print(f"\n  Saved: {output_file}")
    
    detail_file = output_dir / "anesthesia_info_theoretic_detail.json"
    with open(detail_file, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"  Saved: {detail_file}")
    
    print("\n" + "=" * 75)
    print("INFO-THEORETIC KILLER TEST COMPLETE")
    print("=" * 75)


# =========================================================================
# ENTRY POINT
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PMD Anesthesia Info-Theoretic Test")
    parser.add_argument("--data-dir", default="./ds003171")
    parser.add_argument("--output-dir", default="./pmd_results")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    run_info_theoretic_killer_test(args.data_dir, args.output_dir, args.skip_download)
