#!/usr/bin/env python3
"""
==========================================================================
PRESSURE MAKES DIAMONDS — Single Subject Deep Dive
==========================================================================

Comprehensive analysis of one HCP subject to:
1. Validate every analysis pipeline before scaling to 1003 subjects
2. Characterize what a single conscious brain looks like under the framework
3. Preview what the full dataset analysis would test

Analyses:
  A. Per-run Φ_PD stability (do all 4 runs give similar values?)
  B. 7-network vs 5-domain comparison
  C. Static correlation-based vs information-theoretic Φ_PD
  D. Dynamic sliding-window analysis
  E. Comprehensive pair-level characterization
  F. Eigenanalysis and phase transition assessment
  G. Network hierarchy (which networks drive the most conflict?)
  H. Comparison to framework predictions

Usage:
  python pmd_single_subject_deep_dive.py --subject-dir "path/to/100307"
==========================================================================
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
from scipy import stats, signal
from scipy.linalg import eigvalsh
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings('ignore')

YEO7_LABELS = ["Visual", "Somatomotor", "DorsalAttention",
               "VentralAttention", "Limbic", "Frontoparietal", "Default"]

PMD_DOMAINS = ["Sensory", "Emotional", "Social", "Reasoning", "Memory"]
YEO7_TO_PMD = {
    "Visual": "Sensory", "Somatomotor": "Sensory",
    "DorsalAttention": "Memory", "VentralAttention": "Memory",
    "Limbic": "Emotional", "Frontoparietal": "Reasoning", "Default": "Social",
}


def extract_yeo7(fmri_file):
    """Extract Yeo-7 network time series from a NIfTI file."""
    from nilearn import datasets, maskers, image
    
    yeo = datasets.fetch_atlas_yeo_2011(n_networks=7)
    atlas_img = yeo.maps if hasattr(yeo, 'maps') else yeo.get('thick_7', yeo.get('maps'))
    
    masker = maskers.NiftiLabelsMasker(
        labels_img=atlas_img, standardize=True, detrend=True,
        low_pass=0.08, high_pass=0.01, t_r=0.72, memory='nilearn_cache', verbose=0
    )
    
    ts = masker.fit_transform(fmri_file)
    
    # Handle zero-variance
    for col in range(ts.shape[1]):
        if np.std(ts[:, col]) < 1e-10:
            ts[:, col] = np.random.randn(ts.shape[0]) * 1e-6
    
    return ts


def aggregate_to_pmd(ts_7net):
    """Aggregate 7 Yeo networks to 5 PMD domains."""
    domain_cols = {}
    for i, label in enumerate(YEO7_LABELS):
        d = YEO7_TO_PMD[label]
        if d not in domain_cols:
            domain_cols[d] = []
        domain_cols[d].append(i)
    
    ts_5dom = np.zeros((ts_7net.shape[0], len(PMD_DOMAINS)))
    for j, domain in enumerate(PMD_DOMAINS):
        cols = domain_cols[domain]
        ts_5dom[:, j] = np.mean(ts_7net[:, cols], axis=1)
    return ts_5dom


def compute_phi_pd(ts, labels):
    """Compute correlation-based Φ_PD."""
    N = ts.shape[1]
    corr = np.corrcoef(ts.T)
    corr = np.nan_to_num(corr, nan=0.0)
    
    I_mat = np.abs(corr)
    np.fill_diagonal(I_mat, 0)
    
    C_mat = np.zeros_like(corr)
    for i in range(N):
        for j in range(i+1, N):
            c = corr[i, j]
            C_mat[i, j] = 1.0 if c < 0 else (1.0 - c)
            C_mat[j, i] = C_mat[i, j]
    
    phi = sum(C_mat[i,j] * I_mat[i,j] for i in range(N) for j in range(i+1, N))
    phi_norm = (2.0 / (N * (N-1))) * phi
    
    J = np.nan_to_num(I_mat * C_mat, nan=0.0)
    try:
        eigs = eigvalsh(J)
        lmax = np.max(eigs)
    except:
        eigs = np.zeros(N)
        lmax = 0.0
    
    pairs = {}
    for i in range(N):
        for j in range(i+1, N):
            pairs[(labels[i], labels[j])] = {
                'corr': corr[i,j], 'I': I_mat[i,j], 'C': C_mat[i,j],
                'contribution': C_mat[i,j] * I_mat[i,j]
            }
    
    anticorr = sum(1 for i in range(N) for j in range(i+1, N) if corr[i,j] < 0)
    mean_conn = np.mean(np.abs(corr[np.triu_indices(N, k=1)]))
    
    return {
        'phi_pd': phi, 'phi_pd_norm': phi_norm, 'lambda_max': lmax,
        'eigenvalues': eigs, 'mean_conn': mean_conn,
        'n_anticorr': anticorr, 'corr_matrix': corr,
        'pairs': pairs, 'I_matrix': I_mat, 'C_matrix': C_mat, 'J_matrix': J,
    }


def transfer_entropy(source, target, lag=1, bins=8):
    """Transfer entropy from source to target."""
    n = len(source) - lag
    if n < 20:
        return 0.0
    
    src = np.digitize(source, np.linspace(source.min(), source.max(), bins)) - 1
    tgt = np.digitize(target, np.linspace(target.min(), target.max(), bins)) - 1
    
    tgt_f = tgt[lag:]
    tgt_p = tgt[:-lag]
    src_p = src[:-lag]
    
    joint_yy = np.zeros((bins, bins))
    for k in range(n):
        joint_yy[tgt_f[k], tgt_p[k]] += 1
    joint_yy /= joint_yy.sum() + 1e-10
    p_yt_past = joint_yy.sum(axis=0) + 1e-10
    h1 = -np.sum(joint_yy * np.log2(joint_yy / (p_yt_past[None,:] + 1e-10) + 1e-10))
    
    joint_yyx = np.zeros((bins, bins, bins))
    for k in range(n):
        joint_yyx[tgt_f[k], tgt_p[k], src_p[k]] += 1
    joint_yyx /= joint_yyx.sum() + 1e-10
    p_yx = joint_yyx.sum(axis=0) + 1e-10
    h2 = -np.sum(joint_yyx * np.log2(joint_yyx / (p_yx[None,:,:] + 1e-10) + 1e-10))
    
    return max(h1 - h2, 0.0)


def compute_phi_pd_info(ts, labels):
    """Compute information-theoretic Φ_PD (TE × JSD)."""
    N = ts.shape[1]
    
    TE_mat = np.zeros((N, N))
    JSD_mat = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i+1, N):
            te_ij = transfer_entropy(ts[:,i], ts[:,j])
            te_ji = transfer_entropy(ts[:,j], ts[:,i])
            te = (te_ij + te_ji) / 2.0
            TE_mat[i,j] = te
            TE_mat[j,i] = te
            
            # JSD of amplitude distributions
            bins = 20
            all_vals = np.concatenate([ts[:,i], ts[:,j]])
            edges = np.linspace(all_vals.min()-0.01, all_vals.max()+0.01, bins+1)
            h_a, _ = np.histogram(ts[:,i], bins=edges, density=True)
            h_b, _ = np.histogram(ts[:,j], bins=edges, density=True)
            h_a = h_a / (h_a.sum() + 1e-10) + 1e-10
            h_b = h_b / (h_b.sum() + 1e-10) + 1e-10
            h_a /= h_a.sum()
            h_b /= h_b.sum()
            jsd = jensenshannon(h_a, h_b) ** 2
            JSD_mat[i,j] = jsd
            JSD_mat[j,i] = jsd
    
    phi = sum(TE_mat[i,j] * JSD_mat[i,j] for i in range(N) for j in range(i+1, N))
    phi_norm = (2.0 / (N * (N-1))) * phi
    
    J = np.nan_to_num(TE_mat * JSD_mat, nan=0.0)
    try:
        lmax = np.max(eigvalsh(J))
    except:
        lmax = 0.0
    
    pairs = {}
    for i in range(N):
        for j in range(i+1, N):
            pairs[(labels[i], labels[j])] = {
                'TE': TE_mat[i,j], 'JSD': JSD_mat[i,j],
                'contribution': TE_mat[i,j] * JSD_mat[i,j],
                'TE_ij': transfer_entropy(ts[:,i], ts[:,j]),
                'TE_ji': transfer_entropy(ts[:,j], ts[:,i]),
            }
    
    return {
        'phi_pd_info': phi, 'phi_pd_info_norm': phi_norm, 'lambda_max_info': lmax,
        'mean_te': np.mean(TE_mat[np.triu_indices(N, k=1)]),
        'mean_jsd': np.mean(JSD_mat[np.triu_indices(N, k=1)]),
        'pairs': pairs,
    }


def compute_dynamic(ts, labels, window_size=120):
    """Sliding-window dynamic Φ_PD analysis."""
    N = ts.shape[1]
    T = ts.shape[0]
    
    step = window_size // 2
    windows = []
    
    for t in range(0, T - window_size, step):
        window = ts[t:t+window_size, :]
        corr = np.corrcoef(window.T)
        corr = np.nan_to_num(corr, nan=0.0)
        
        I_mat = np.abs(corr)
        np.fill_diagonal(I_mat, 0)
        C_mat = np.zeros_like(corr)
        for i in range(N):
            for j in range(i+1, N):
                c = corr[i,j]
                C_mat[i,j] = 1.0 if c < 0 else (1.0 - c)
                C_mat[j,i] = C_mat[i,j]
        
        phi = sum(C_mat[i,j] * I_mat[i,j] for i in range(N) for j in range(i+1, N))
        phi_norm = (2.0 / (N * (N-1))) * phi
        
        J = np.nan_to_num(I_mat * C_mat, nan=0.0)
        try:
            lmax = np.max(eigvalsh(J))
        except:
            lmax = 0.0
        
        anticorr = sum(1 for i in range(N) for j in range(i+1, N) if corr[i,j] < 0)
        
        windows.append({
            'time': t * 0.72,  # Convert to seconds (HCP TR = 0.72s)
            'phi_pd_norm': phi_norm, 'lambda_max': lmax,
            'n_anticorr': anticorr,
        })
    
    return windows


def find_rest_files(subject_dir):
    """Find all resting-state NIfTI files."""
    rest_files = {}
    for root, dirs, files in os.walk(subject_dir):
        for f in files:
            if 'rfMRI_REST' in f and 'hp2000_clean' in f and f.endswith('.nii.gz'):
                if 'rclean_tclean_vn' in f:
                    continue  # Skip variance-normalized
                fpath = os.path.join(root, f)
                # Determine run name
                if 'REST1_LR' in f:
                    rest_files['REST1_LR'] = fpath
                elif 'REST1_RL' in f:
                    rest_files['REST1_RL'] = fpath
                elif 'REST2_LR' in f:
                    rest_files['REST2_LR'] = fpath
                elif 'REST2_RL' in f:
                    rest_files['REST2_RL'] = fpath
                elif 'rfMRI_REST_hp2000' in f:
                    rest_files['REST_combined'] = fpath
    return rest_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject-dir', required=True)
    parser.add_argument('--output-dir', default='./pmd_results')
    args = parser.parse_args()
    
    print("=" * 75)
    print("PRESSURE MAKES DIAMONDS")
    print("Single Subject Deep Dive — HCP Subject 100307")
    print("=" * 75)
    
    # Find files
    rest_files = find_rest_files(args.subject_dir)
    print(f"\nResting-state files found:")
    for name, path in sorted(rest_files.items()):
        size = os.path.getsize(path) / 1024 / 1024
        print(f"  {name}: {os.path.basename(path)} ({size:.0f} MB)")
    
    if not rest_files:
        print("ERROR: No resting-state files found")
        return
    
    # =====================================================================
    # A. PER-RUN STABILITY
    # =====================================================================
    print(f"\n{'='*75}")
    print("A. PER-RUN Φ_PD STABILITY")
    print("   Do all 4 runs produce similar Φ_PD values?")
    print("   If yes: Φ_PD is a stable trait measure, not noise")
    print(f"{'='*75}")
    
    run_results = {}
    individual_runs = {k: v for k, v in rest_files.items() if k != 'REST_combined'}
    
    for run_name, run_path in sorted(individual_runs.items()):
        print(f"\n  Extracting {run_name}...")
        ts = extract_yeo7(run_path)
        print(f"    Shape: {ts.shape}")
        
        result = compute_phi_pd(ts, YEO7_LABELS)
        run_results[run_name] = result
        
        print(f"    Φ_PD_norm: {result['phi_pd_norm']:.6f}")
        print(f"    λ_max:     {result['lambda_max']:.6f}")
        print(f"    Anticorr:  {result['n_anticorr']}/21 pairs")
    
    if len(run_results) > 1:
        phi_values = [r['phi_pd_norm'] for r in run_results.values()]
        lambda_values = [r['lambda_max'] for r in run_results.values()]
        
        print(f"\n  --- Stability Summary ---")
        print(f"  Φ_PD across runs: {np.mean(phi_values):.6f} ± {np.std(phi_values):.6f}")
        print(f"  λ_max across runs: {np.mean(lambda_values):.6f} ± {np.std(lambda_values):.6f}")
        cv = np.std(phi_values) / np.mean(phi_values) * 100
        print(f"  Coefficient of variation: {cv:.1f}%")
        
        if cv < 10:
            print(f"  ★ STABLE: Φ_PD is consistent across runs (CV < 10%)")
            print(f"    Implication: Φ_PD reflects a stable brain property,")
            print(f"    not scan-to-scan noise. This is prerequisite for")
            print(f"    individual differences to be meaningful.")
        elif cv < 20:
            print(f"  Moderately stable (CV 10-20%)")
        else:
            print(f"  Variable across runs (CV > 20%) — may reflect state fluctuations")
    
    # =====================================================================
    # B. 7-NETWORK vs 5-DOMAIN COMPARISON
    # =====================================================================
    print(f"\n{'='*75}")
    print("B. 7-NETWORK vs 5-DOMAIN COMPARISON")
    print("   Does aggregating to PMD domains lose signal?")
    print(f"{'='*75}")
    
    # Use combined file for the remaining analyses
    if 'REST_combined' in rest_files:
        print(f"\n  Using combined rest file...")
        ts_7 = extract_yeo7(rest_files['REST_combined'])
    else:
        # Use first available run
        first_run = sorted(individual_runs.keys())[0]
        print(f"\n  Using {first_run}...")
        ts_7 = extract_yeo7(individual_runs[first_run])
    
    print(f"  Shape: {ts_7.shape}")
    
    ts_5 = aggregate_to_pmd(ts_7)
    
    r7 = compute_phi_pd(ts_7, YEO7_LABELS)
    r5 = compute_phi_pd(ts_5, PMD_DOMAINS)
    
    print(f"\n  7-Network Analysis:")
    print(f"    Φ_PD_norm: {r7['phi_pd_norm']:.6f}")
    print(f"    λ_max:     {r7['lambda_max']:.6f}")
    print(f"    Anticorr:  {r7['n_anticorr']}/21 pairs")
    
    print(f"\n  5-Domain Analysis:")
    print(f"    Φ_PD_norm: {r5['phi_pd_norm']:.6f}")
    print(f"    λ_max:     {r5['lambda_max']:.6f}")
    print(f"    Anticorr:  {r5['n_anticorr']}/10 pairs")
    
    print(f"\n  Aggregation effect:")
    print(f"    Φ_PD ratio (5dom/7net): {r5['phi_pd_norm']/r7['phi_pd_norm']:.3f}")
    if r5['phi_pd_norm'] < r7['phi_pd_norm']:
        print(f"    Aggregation REDUCES Φ_PD — merging networks that conflict")
        print(f"    hides inter-network conflict within domains")
    
    # =====================================================================
    # C. CORRELATION vs INFORMATION-THEORETIC
    # =====================================================================
    print(f"\n{'='*75}")
    print("C. CORRELATION-BASED vs INFORMATION-THEORETIC Φ_PD")
    print("   Transfer Entropy × JSD vs Correlation-based")
    print(f"{'='*75}")
    
    print(f"\n  Computing information-theoretic measures (this takes a few minutes)...")
    r7_info = compute_phi_pd_info(ts_7, YEO7_LABELS)
    
    print(f"\n  Correlation-based:")
    print(f"    Φ_PD_norm: {r7['phi_pd_norm']:.6f}")
    
    print(f"\n  Information-theoretic (TE × JSD):")
    print(f"    Φ_PD_norm: {r7_info['phi_pd_info_norm']:.6f}")
    print(f"    Mean TE:   {r7_info['mean_te']:.6f}")
    print(f"    Mean JSD:  {r7_info['mean_jsd']:.6f}")
    
    # Compare pair rankings
    print(f"\n  Pair ranking comparison:")
    print(f"  {'Pair':<35} {'Corr-based':>12} {'TE×JSD':>12}")
    print(f"  {'-'*60}")
    
    corr_pairs = sorted(r7['pairs'].items(), key=lambda x: x[1]['contribution'], reverse=True)
    info_pairs = sorted(r7_info['pairs'].items(), key=lambda x: x[1]['contribution'], reverse=True)
    
    # Create ranking maps
    corr_rank = {pair: i+1 for i, (pair, _) in enumerate(corr_pairs)}
    info_rank = {pair: i+1 for i, (pair, _) in enumerate(info_pairs)}
    
    for pair, data in corr_pairs:
        c_val = data['contribution']
        i_val = r7_info['pairs'][pair]['contribution']
        print(f"  {pair[0]+'-'+pair[1]:<35} {c_val:>12.6f} {i_val:>12.6f}")
    
    # =====================================================================
    # D. DYNAMIC ANALYSIS
    # =====================================================================
    print(f"\n{'='*75}")
    print("D. DYNAMIC Φ_PD OVER TIME")
    print("   How does Φ_PD fluctuate across the scan?")
    print(f"{'='*75}")
    
    print(f"\n  Computing sliding-window Φ_PD (window=120 TRs = 86.4s)...")
    windows = compute_dynamic(ts_7, YEO7_LABELS, window_size=120)
    
    phi_timecourse = [w['phi_pd_norm'] for w in windows]
    lambda_timecourse = [w['lambda_max'] for w in windows]
    anticorr_timecourse = [w['n_anticorr'] for w in windows]
    time_points = [w['time'] for w in windows]
    
    print(f"    Windows: {len(windows)}")
    print(f"    Time span: {time_points[0]:.0f}s to {time_points[-1]:.0f}s")
    print(f"\n  Dynamic Φ_PD:")
    print(f"    Mean:  {np.mean(phi_timecourse):.6f}")
    print(f"    Std:   {np.std(phi_timecourse):.6f}")
    print(f"    Min:   {np.min(phi_timecourse):.6f} (at {time_points[np.argmin(phi_timecourse)]:.0f}s)")
    print(f"    Max:   {np.max(phi_timecourse):.6f} (at {time_points[np.argmax(phi_timecourse)]:.0f}s)")
    print(f"    Range: {np.max(phi_timecourse) - np.min(phi_timecourse):.6f}")
    print(f"    CV:    {np.std(phi_timecourse)/np.mean(phi_timecourse)*100:.1f}%")
    
    print(f"\n  Dynamic λ_max:")
    print(f"    Mean:  {np.mean(lambda_timecourse):.6f}")
    print(f"    Always > 1? {'YES' if all(l > 1 for l in lambda_timecourse) else 'NO'}")
    pct_above = sum(1 for l in lambda_timecourse if l > 1) / len(lambda_timecourse) * 100
    print(f"    % windows with λ_max > 1: {pct_above:.1f}%")
    
    print(f"\n  Dynamic anticorrelation:")
    print(f"    Mean pairs anticorrelated: {np.mean(anticorr_timecourse):.1f}/21")
    print(f"    Range: {np.min(anticorr_timecourse)} to {np.max(anticorr_timecourse)}")
    
    if pct_above > 90:
        print(f"\n  ★ λ_max > 1 in {pct_above:.0f}% of windows")
        print(f"    This subject is CONSISTENTLY in the coordination regime")
        print(f"    throughout the entire resting-state scan.")
        print(f"    Framework prediction: healthy awake adult should be")
        print(f"    above critical threshold. CONFIRMED for this subject.")
    
    # =====================================================================
    # E. EIGENANALYSIS
    # =====================================================================
    print(f"\n{'='*75}")
    print("E. EIGENANALYSIS — Phase Transition Assessment")
    print(f"{'='*75}")
    
    eigs = sorted(r7['eigenvalues'], reverse=True)
    print(f"\n  Eigenvalues of J = I ⊙ C (7-network):")
    for i, e in enumerate(eigs):
        bar = "█" * int(e / max(eigs) * 40)
        above = " ← ABOVE THRESHOLD" if e > 1.0 else ""
        print(f"    λ_{i+1} = {e:>8.4f}  {bar}{above}")
    
    print(f"\n  λ_max = {eigs[0]:.4f}")
    print(f"  λ_max / λ_2 = {eigs[0]/eigs[1]:.2f} (dominance ratio)")
    
    if eigs[0] > 1.0:
        print(f"\n  ★ System is ABOVE critical threshold (λ_max > 1)")
        print(f"    The mean-field self-consistency equation m = tanh(β·λ_max·m)")
        print(f"    has a non-trivial solution → the system is in the ordered")
        print(f"    (coordinated) phase → consciousness predicted.")
    else:
        print(f"\n  System is below critical threshold")
    
    # =====================================================================
    # F. NETWORK CONFLICT HIERARCHY
    # =====================================================================
    print(f"\n{'='*75}")
    print("F. NETWORK CONFLICT HIERARCHY")
    print("   Which networks generate the most integration pressure?")
    print(f"{'='*75}")
    
    # Sum conflict contributions per network
    network_total_conflict = {}
    for label in YEO7_LABELS:
        total = 0
        for (a, b), data in r7['pairs'].items():
            if a == label or b == label:
                total += data['contribution']
        network_total_conflict[label] = total
    
    sorted_networks = sorted(network_total_conflict.items(), key=lambda x: x[1], reverse=True)
    
    max_val = max(v for _, v in sorted_networks)
    print(f"\n  Total Φ_PD contribution by network:")
    for label, total in sorted_networks:
        bar = "█" * int(total / max_val * 40)
        pct = total / r7['phi_pd'] * 100
        print(f"    {label:<20} {total:>8.4f} ({pct:>5.1f}%) {bar}")
    
    top_network = sorted_networks[0][0]
    print(f"\n  Dominant conflict hub: {top_network}")
    
    if top_network == "Default":
        print(f"  ★ Default Mode Network is the primary conflict hub")
        print(f"    Framework prediction: the self-model network should conflict")
        print(f"    with all other domains because it must arbitrate between them.")
        print(f"    CONFIRMED for this subject.")
    elif top_network in ["Frontoparietal", "VentralAttention"]:
        print(f"  {top_network} is the primary hub — an executive/attention network")
        print(f"  Framework predicts Default should dominate, but executive networks")
        print(f"  as primary hubs are consistent with active arbitration.")
    
    # =====================================================================
    # G. CORRELATION MATRIX STRUCTURE
    # =====================================================================
    print(f"\n{'='*75}")
    print("G. FULL CORRELATION MATRIX")
    print(f"{'='*75}")
    
    corr = r7['corr_matrix']
    print(f"\n  {'':>20}", end="")
    for l in YEO7_LABELS:
        print(f" {l[:6]:>8}", end="")
    print()
    
    for i, li in enumerate(YEO7_LABELS):
        print(f"  {li:<20}", end="")
        for j, lj in enumerate(YEO7_LABELS):
            val = corr[i, j]
            if i == j:
                marker = "  ----  "
            elif val < -0.1:
                marker = f" {val:>6.3f}*"  # Anticorrelated
            elif val < 0:
                marker = f" {val:>6.3f}-"
            else:
                marker = f" {val:>6.3f} "
            print(marker, end="")
        print()
    
    print(f"\n  * = anticorrelated (C_ij = 1.0, maximum conflict)")
    
    # =====================================================================
    # H. FRAMEWORK PREDICTION SUMMARY
    # =====================================================================
    print(f"\n{'='*75}")
    print("H. FRAMEWORK PREDICTION SUMMARY")
    print(f"{'='*75}")
    
    predictions = [
        ("λ_max > 1 (conscious coordination regime)",
         r7['lambda_max'] > 1.0,
         f"λ_max = {r7['lambda_max']:.4f}"),
        
        ("Multiple anticorrelated pairs (multi-domain conflict)",
         r7['n_anticorr'] >= 5,
         f"{r7['n_anticorr']}/21 pairs anticorrelated"),
        
        ("Default Mode Network as primary conflict hub",
         sorted_networks[0][0] == "Default",
         f"Top hub: {sorted_networks[0][0]} ({sorted_networks[0][1]:.4f})"),
        
        ("Φ_PD stable across runs (trait-like measure)",
         len(run_results) > 1 and np.std(phi_values)/np.mean(phi_values) < 0.15,
         f"CV = {np.std(phi_values)/np.mean(phi_values)*100:.1f}%" if len(run_results) > 1 else "Only 1 run"),
        
        ("λ_max > 1 consistently across time windows",
         pct_above > 80,
         f"{pct_above:.0f}% of windows above threshold"),
        
        ("High-order networks (Default, FPN) generate most conflict",
         sorted_networks[0][0] in ["Default", "Frontoparietal"] or sorted_networks[1][0] in ["Default", "Frontoparietal"],
         f"Top 2: {sorted_networks[0][0]}, {sorted_networks[1][0]}"),
    ]
    
    confirmed = 0
    for desc, result, detail in predictions:
        status = "✓ CONFIRMED" if result else "✗ NOT CONFIRMED"
        confirmed += 1 if result else 0
        print(f"\n  {status}")
        print(f"    Prediction: {desc}")
        print(f"    Result:     {detail}")
    
    print(f"\n{'='*75}")
    print(f"SCORECARD: {confirmed}/{len(predictions)} predictions confirmed")
    print(f"{'='*75}")
    
    if confirmed == len(predictions):
        print(f"\n  ★ ALL PREDICTIONS CONFIRMED for this subject")
        print(f"    If this pattern holds across 1003 subjects, the framework's")
        print(f"    description of resting-state consciousness is empirically")
        print(f"    validated at the individual level.")
    elif confirmed >= len(predictions) - 1:
        print(f"\n  Strong support: {confirmed}/{len(predictions)} predictions confirmed")
    else:
        print(f"\n  Partial support: {confirmed}/{len(predictions)} predictions confirmed")
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        'subject': '100307',
        'phi_pd_norm_7net': r7['phi_pd_norm'],
        'phi_pd_norm_5dom': r5['phi_pd_norm'],
        'phi_pd_info_norm': r7_info['phi_pd_info_norm'],
        'lambda_max': r7['lambda_max'],
        'n_anticorr': r7['n_anticorr'],
        'predictions_confirmed': confirmed,
        'predictions_total': len(predictions),
        'dominant_hub': sorted_networks[0][0],
        'dynamic_pct_above_threshold': pct_above,
    }
    
    if len(run_results) > 1:
        summary['per_run_phi_pd'] = {k: v['phi_pd_norm'] for k, v in run_results.items()}
        summary['per_run_cv'] = float(np.std(phi_values) / np.mean(phi_values) * 100)
    
    path = os.path.join(args.output_dir, 'hcp_single_subject_deep_dive.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: {path}")
    
    print(f"\n{'='*75}")
    print("DEEP DIVE COMPLETE")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
