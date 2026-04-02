#!/usr/bin/env python3
"""
===========================================================================
PRESSURE MAKES DIAMONDS — PSYCHEDELIC EFFECTIVE CONFLICT
Does the organized-conflict revision fix the LSD result too?
===========================================================================

The raw TE×JSD proxy showed Φ_PD increasing slightly under LSD (d=0.212, ns).
The correlation proxy showed it decreasing.
Neither was significant.

The anesthesia effective-conflict revision FIXED the ordering for propofol.
Does the same revision work for LSD?

If effective conflict shows LSD > Placebo (correct direction) AND the
anesthesia result also shows Awake > Deep (correct direction), then
the same proxy revision fixes BOTH pharmacological datasets.

USAGE:
  python pmd_psychedelic_effective_conflict.py --data-dir ./psychedelic_data --skip-download
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

YEO7_LABELS = [
    "Visual", "Somatomotor", "DorsalAttention",
    "VentralAttention", "Limbic", "Frontoparietal", "Default"
]

# =========================================================================
# TIMESERIES EXTRACTION
# =========================================================================

def extract_network_timeseries(bold_file):
    """Extract mean timeseries for each Yeo-7 network."""
    from nilearn import maskers
    import glob
    
    yeo_dir = os.path.join(os.path.expanduser('~'), 'nilearn_data', 'yeo_2011')
    candidates = glob.glob(os.path.join(yeo_dir, '**', 'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz'), recursive=True)
    if not candidates:
        candidates = glob.glob(os.path.join(yeo_dir, '**', '*7Networks*1mm.nii.gz'), recursive=True)
    if not candidates:
        raise RuntimeError(f"Cannot find Yeo atlas in {yeo_dir}")
    
    masker = maskers.NiftiLabelsMasker(
        labels_img=candidates[0],
        standardize='zscore_sample',
        detrend=True,
        memory='nilearn_cache',
        memory_level=1
    )
    
    timeseries = masker.fit_transform(bold_file)
    n_tp, n_net = timeseries.shape
    if n_tp < 30:
        raise ValueError(f"Too few timepoints: {n_tp}")
    if n_net != 7:
        ts_padded = np.zeros((n_tp, 7))
        ts_padded[:, :min(n_net, 7)] = timeseries[:, :min(n_net, 7)]
        timeseries = ts_padded
    return timeseries


# =========================================================================
# MEASURES (identical to anesthesia effective conflict script)
# =========================================================================

def transfer_entropy(source, target, lag=1, bins=10):
    n = len(source)
    if n <= lag + 1:
        return 0.0
    y_now = target[lag:]
    y_past = target[:-lag]
    x_past = source[:-lag]
    
    def bin_data(x, bins):
        mn, mx = x.min(), x.max()
        if mx == mn:
            return np.zeros(len(x), dtype=int)
        return np.clip(((x - mn) / (mx - mn) * (bins - 1)).astype(int), 0, bins - 1)
    
    y_now_b = bin_data(y_now, bins)
    y_past_b = bin_data(y_past, bins)
    x_past_b = bin_data(x_past, bins)
    
    def entropy_from_counts(counts):
        p = counts[counts > 0] / counts.sum()
        return -np.sum(p * np.log2(p))
    
    joint_yy = np.zeros((bins, bins))
    for i in range(len(y_now_b)):
        joint_yy[y_now_b[i], y_past_b[i]] += 1
    h_yy = entropy_from_counts(joint_yy.ravel())
    
    hist_ypast = np.bincount(y_past_b, minlength=bins).astype(float)
    h_ypast = entropy_from_counts(hist_ypast)
    
    joint_yyx = np.zeros((bins, bins, bins))
    for i in range(len(y_now_b)):
        joint_yyx[y_now_b[i], y_past_b[i], x_past_b[i]] += 1
    h_yyx = entropy_from_counts(joint_yyx.ravel())
    
    joint_yx = np.zeros((bins, bins))
    for i in range(len(y_past_b)):
        joint_yx[y_past_b[i], x_past_b[i]] += 1
    h_yx = entropy_from_counts(joint_yx.ravel())
    
    te = (h_yy - h_ypast) - (h_yyx - h_yx)
    return max(te, 0.0)


def symmetric_transfer_entropy(ts_i, ts_j, lag=1, bins=10):
    return (transfer_entropy(ts_i, ts_j, lag, bins) + 
            transfer_entropy(ts_j, ts_i, lag, bins)) / 2.0


def signal_jsd(ts_i, ts_j, bins=30):
    all_vals = np.concatenate([ts_i, ts_j])
    mn, mx = all_vals.min(), all_vals.max()
    if mx == mn:
        return 0.0
    edges = np.linspace(mn, mx, bins + 1)
    p = np.histogram(ts_i, bins=edges)[0].astype(float)
    q = np.histogram(ts_j, bins=edges)[0].astype(float)
    eps = 1e-10
    p = (p + eps); p = p / p.sum()
    q = (q + eps); q = q / q.sum()
    m = 0.5 * (p + q)
    jsd = 0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m))
    return max(jsd, 0.0)


def windowed_jsd(ts_i, ts_j, window=60, step=30, bins=20):
    n = len(ts_i)
    if n < window:
        return signal_jsd(ts_i, ts_j, bins), 0.0
    jsd_values = []
    for start in range(0, n - window + 1, step):
        jsd_values.append(signal_jsd(ts_i[start:start+window], ts_j[start:start+window], bins))
    if not jsd_values:
        return signal_jsd(ts_i, ts_j, bins), 0.0
    return np.mean(jsd_values), np.std(jsd_values)


def signal_coherence(ts, lag=1):
    if len(ts) < lag + 2:
        return 0.0
    ac = np.corrcoef(ts[:-lag], ts[lag:])[0, 1]
    if np.isnan(ac):
        return 0.0
    return max(0.0, ac)


def soft_gate(x, x0, tau):
    z = (x - x0) / tau if tau > 0 else 0
    z = np.clip(z, -20, 20)
    return 1.0 / (1.0 + np.exp(-z))


# =========================================================================
# EFFECTIVE CONFLICT COMPUTATION
# =========================================================================

def compute_pmd_effective_conflict(timeseries, i_min=None, tau=0.05):
    N = timeseries.shape[1]
    
    TE = np.zeros((N, N))
    JSD_mean = np.zeros((N, N))
    JSD_std = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i + 1, N):
            te = symmetric_transfer_entropy(timeseries[:, i], timeseries[:, j])
            jsd_m, jsd_s = windowed_jsd(timeseries[:, i], timeseries[:, j])
            TE[i, j] = TE[j, i] = te
            JSD_mean[i, j] = JSD_mean[j, i] = jsd_m
            JSD_std[i, j] = JSD_std[j, i] = jsd_s
    
    coherence = np.zeros(N)
    for i in range(N):
        coherence[i] = signal_coherence(timeseries[:, i])
    
    te_values = [TE[i, j] for i in range(N) for j in range(i+1, N)]
    if i_min is None:
        i_min = np.percentile(te_values, 25)
    
    J_raw = np.zeros((N, N))
    J_effective = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i + 1, N):
            te = TE[i, j]
            jsd_m = JSD_mean[i, j]
            jsd_s = JSD_std[i, j]
            
            s_gate = 1.0 / (1.0 + jsd_s) if jsd_s >= 0 else 1.0
            q_gate = soft_gate(te, i_min, tau)
            r_gate = np.sqrt(max(coherence[i], 0) * max(coherence[j], 0))
            
            c_eff = jsd_m * s_gate * q_gate * r_gate
            
            J_raw[i, j] = J_raw[j, i] = te * jsd_m
            J_effective[i, j] = J_effective[j, i] = te * c_eff
    
    np.fill_diagonal(J_effective, 0)
    np.fill_diagonal(J_raw, 0)
    
    def get_metrics(J):
        phi = sum(J[i, j] for i in range(N) for j in range(i+1, N))
        n_pairs = N * (N - 1) // 2
        phi_norm = phi / n_pairs if n_pairs > 0 else 0
        J_sym = np.nan_to_num((J + J.T) / 2.0)
        try:
            eigs = np.linalg.eigvalsh(J_sym)
        except:
            eigs = np.real(np.linalg.eigvals(J_sym))
        eigs = np.sort(np.real(eigs))[::-1]
        return {'phi_pd_norm': float(phi_norm), 'lambda_max': float(eigs[0])}
    
    raw = get_metrics(J_raw)
    eff = get_metrics(J_effective)
    
    mean_te = float(np.mean(te_values))
    mean_jsd = float(np.mean([JSD_mean[i,j] for i in range(N) for j in range(i+1,N)]))
    mean_jsd_std = float(np.mean([JSD_std[i,j] for i in range(N) for j in range(i+1,N)]))
    mean_coh = float(np.mean(coherence))
    
    return {
        'raw': raw, 'effective': eff,
        'mean_te': mean_te, 'mean_jsd': mean_jsd,
        'mean_jsd_std': mean_jsd_std, 'mean_coherence': mean_coh,
    }


# =========================================================================
# FIND LSD/PLACEBO FILES
# =========================================================================

def find_psychedelic_files(data_dir):
    """Find LSD and Placebo BOLD files from ds003059."""
    data_dir = Path(data_dir)
    results = {}
    
    # ds003059 structure: sub-XX/ses-LSD/sub-XX_ses-LSD_task-rest_run-01_bold.nii.gz
    # (files directly in ses-* folder, no func/ subfolder)
    patterns = [
        "sub-*/ses-*/*task-rest*bold.nii.gz",
        "sub-*/ses-*/func/*task-rest*bold.nii.gz",
        "sub-*/ses-*/*bold.nii.gz",
        "sub-*/ses-*/func/*bold.nii.gz",
    ]
    
    all_files = []
    for p in patterns:
        all_files.extend(data_dir.glob(p))
    all_files = list(set(all_files))
    
    if not all_files:
        # Try looking in the psychedelic_data subdirectory
        for p in patterns:
            all_files.extend(data_dir.glob(f"ds003059/{p}"))
            all_files.extend(data_dir.glob(f"**/{p}"))
        all_files = list(set(all_files))
    
    if not all_files:
        print(f"\n  No BOLD files found in {data_dir}")
        print(f"  Contents:")
        try:
            for item in sorted(data_dir.iterdir())[:20]:
                print(f"    {item.name}")
        except:
            pass
        return results
    
    print(f"\n  Found {len(all_files)} BOLD files")
    
    for fpath in sorted(all_files):
        parts = str(fpath).replace('\\', '/')
        fname = fpath.name
        
        sub_id = None
        for part in parts.split('/'):
            if part.startswith('sub-'):
                sub_id = part
                break
        if not sub_id:
            continue
        
        # Determine condition from session name
        condition = None
        for part in parts.split('/'):
            if 'ses-' in part:
                ses = part.lower()
                if 'lsd' in ses:
                    condition = 'LSD'
                elif 'plcb' in ses or 'placebo' in ses:
                    condition = 'Placebo'
                break
        
        if not condition:
            continue
        
        if sub_id not in results:
            results[sub_id] = {}
        if condition not in results[sub_id]:
            results[sub_id][condition] = []
        results[sub_id][condition].append(str(fpath))
    
    return results


# =========================================================================
# MAIN
# =========================================================================

def run_psychedelic_effective_conflict(data_dir, output_dir, skip_download=False):
    print("=" * 75)
    print("PRESSURE MAKES DIAMONDS — PSYCHEDELIC EFFECTIVE CONFLICT")
    print("Does the organized-conflict revision fix the LSD result?")
    print("=" * 75)
    print()
    print("ANESTHESIA RESULT: Effective conflict FIXED the ordering")
    print("  Raw: Deep > Awake (WRONG)")
    print("  Effective: Awake > Deep (CORRECT)")
    print()
    print("PSYCHEDELIC PREDICTION: LSD should show HIGHER effective Φ_PD")
    print("  than Placebo, because psychedelics increase ORGANIZED conflict")
    print("  (not just noise-like fragmentation).")
    print("=" * 75)
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 75)
    print("STEP 1: Finding BOLD files")
    print("=" * 75)
    
    subject_files = find_psychedelic_files(data_dir)
    if not subject_files:
        print("No subjects found.")
        sys.exit(1)
    
    # Count subjects with both conditions
    paired = {s: f for s, f in subject_files.items() if 'LSD' in f and 'Placebo' in f}
    print(f"  Subjects with both LSD and Placebo: {len(paired)}")
    
    print("\n" + "=" * 75)
    print("STEP 2: Computing effective conflict metrics")
    print("=" * 75)
    
    subject_results = {}
    
    for sub_id in sorted(paired.keys()):
        print(f"\n  {sub_id}:")
        subject_results[sub_id] = {}
        
        for condition in ['LSD', 'Placebo']:
            files = paired[sub_id][condition]
            
            # Average across runs
            all_raw_phi = []
            all_eff_phi = []
            all_raw_lmax = []
            all_eff_lmax = []
            all_te = []
            all_jsd = []
            all_jsd_std = []
            all_coh = []
            
            for fpath in files:
                fname = os.path.basename(fpath)
                print(f"    {condition}: {fname}...", end="", flush=True)
                
                try:
                    ts = extract_network_timeseries(fpath)
                    metrics = compute_pmd_effective_conflict(ts)
                    
                    all_raw_phi.append(metrics['raw']['phi_pd_norm'])
                    all_eff_phi.append(metrics['effective']['phi_pd_norm'])
                    all_raw_lmax.append(metrics['raw']['lambda_max'])
                    all_eff_lmax.append(metrics['effective']['lambda_max'])
                    all_te.append(metrics['mean_te'])
                    all_jsd.append(metrics['mean_jsd'])
                    all_jsd_std.append(metrics['mean_jsd_std'])
                    all_coh.append(metrics['mean_coherence'])
                    
                    print(f" eff_λ={metrics['effective']['lambda_max']:.4f}, coh={metrics['mean_coherence']:.3f}")
                    
                except Exception as e:
                    print(f" ERROR: {e}")
                    continue
            
            if all_eff_phi:
                subject_results[sub_id][condition] = {
                    'raw_phi': float(np.mean(all_raw_phi)),
                    'eff_phi': float(np.mean(all_eff_phi)),
                    'raw_lmax': float(np.mean(all_raw_lmax)),
                    'eff_lmax': float(np.mean(all_eff_lmax)),
                    'mean_te': float(np.mean(all_te)),
                    'mean_jsd': float(np.mean(all_jsd)),
                    'mean_jsd_std': float(np.mean(all_jsd_std)),
                    'mean_coherence': float(np.mean(all_coh)),
                }
    
    # =====================================================================
    # GROUP RESULTS
    # =====================================================================
    print("\n" + "=" * 75)
    print("STEP 3: GROUP-LEVEL RESULTS")
    print("=" * 75)
    
    # Collect paired data
    lsd_raw_phi, plcb_raw_phi = [], []
    lsd_eff_phi, plcb_eff_phi = [], []
    lsd_raw_lmax, plcb_raw_lmax = [], []
    lsd_eff_lmax, plcb_eff_lmax = [], []
    lsd_te, plcb_te = [], []
    lsd_jsd, plcb_jsd = [], []
    lsd_jsd_std, plcb_jsd_std = [], []
    lsd_coh, plcb_coh = [], []
    
    for sub_id in sorted(subject_results.keys()):
        if 'LSD' in subject_results[sub_id] and 'Placebo' in subject_results[sub_id]:
            l = subject_results[sub_id]['LSD']
            p = subject_results[sub_id]['Placebo']
            lsd_raw_phi.append(l['raw_phi']); plcb_raw_phi.append(p['raw_phi'])
            lsd_eff_phi.append(l['eff_phi']); plcb_eff_phi.append(p['eff_phi'])
            lsd_raw_lmax.append(l['raw_lmax']); plcb_raw_lmax.append(p['raw_lmax'])
            lsd_eff_lmax.append(l['eff_lmax']); plcb_eff_lmax.append(p['eff_lmax'])
            lsd_te.append(l['mean_te']); plcb_te.append(p['mean_te'])
            lsd_jsd.append(l['mean_jsd']); plcb_jsd.append(p['mean_jsd'])
            lsd_jsd_std.append(l['mean_jsd_std']); plcb_jsd_std.append(p['mean_jsd_std'])
            lsd_coh.append(l['mean_coherence']); plcb_coh.append(p['mean_coherence'])
    
    n = len(lsd_eff_phi)
    print(f"\n  Paired subjects: {n}")
    
    print(f"\n  === RAW (TE × JSD) ===")
    print(f"  {'':>20} {'LSD':>12} {'Placebo':>12} {'Direction':>10}")
    print(f"  {'Φ_PD':>20} {np.mean(lsd_raw_phi):>12.6f} {np.mean(plcb_raw_phi):>12.6f} {'LSD ↑' if np.mean(lsd_raw_phi) > np.mean(plcb_raw_phi) else 'LSD ↓'}")
    print(f"  {'λ_max':>20} {np.mean(lsd_raw_lmax):>12.6f} {np.mean(plcb_raw_lmax):>12.6f} {'LSD ↑' if np.mean(lsd_raw_lmax) > np.mean(plcb_raw_lmax) else 'LSD ↓'}")
    
    print(f"\n  === EFFECTIVE (TE × C_effective) ===")
    print(f"  {'':>20} {'LSD':>12} {'Placebo':>12} {'Direction':>10}")
    print(f"  {'Φ_PD_eff':>20} {np.mean(lsd_eff_phi):>12.6f} {np.mean(plcb_eff_phi):>12.6f} {'LSD ↑' if np.mean(lsd_eff_phi) > np.mean(plcb_eff_phi) else 'LSD ↓'}")
    print(f"  {'λ_max_eff':>20} {np.mean(lsd_eff_lmax):>12.6f} {np.mean(plcb_eff_lmax):>12.6f} {'LSD ↑' if np.mean(lsd_eff_lmax) > np.mean(plcb_eff_lmax) else 'LSD ↓'}")
    
    print(f"\n  === DECOMPOSITION ===")
    print(f"  {'':>20} {'LSD':>12} {'Placebo':>12} {'Direction':>10}")
    print(f"  {'Integration (TE)':>20} {np.mean(lsd_te):>12.4f} {np.mean(plcb_te):>12.4f} {'LSD ↑' if np.mean(lsd_te) > np.mean(plcb_te) else 'LSD ↓'}")
    print(f"  {'Raw conflict (JSD)':>20} {np.mean(lsd_jsd):>12.4f} {np.mean(plcb_jsd):>12.4f} {'LSD ↑' if np.mean(lsd_jsd) > np.mean(plcb_jsd) else 'LSD ↓'}")
    print(f"  {'JSD volatility':>20} {np.mean(lsd_jsd_std):>12.4f} {np.mean(plcb_jsd_std):>12.4f} {'LSD ↑' if np.mean(lsd_jsd_std) > np.mean(plcb_jsd_std) else 'LSD ↓'}")
    print(f"  {'Coherence':>20} {np.mean(lsd_coh):>12.4f} {np.mean(plcb_coh):>12.4f} {'LSD ↑' if np.mean(lsd_coh) > np.mean(plcb_coh) else 'LSD ↓'}")
    
    # =====================================================================
    # STATISTICAL TESTS
    # =====================================================================
    print("\n" + "=" * 75)
    print("STEP 4: STATISTICAL TESTS")
    print("=" * 75)
    
    from scipy import stats
    
    def paired_test(a, b, name):
        a, b = np.array(a), np.array(b)
        diff = a - b
        t, p = stats.ttest_rel(a, b)
        d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        direction = "LSD ↑" if np.mean(a) > np.mean(b) else "LSD ↓"
        print(f"  {name:<25} {direction} d={d:+.3f}, t({n-1})={t:.3f}, p={p:.4f} {sig}")
    
    print(f"\n  Paired t-tests (LSD vs Placebo, N={n}):")
    paired_test(lsd_raw_phi, plcb_raw_phi, "Raw Φ_PD:")
    paired_test(lsd_raw_lmax, plcb_raw_lmax, "Raw λ_max:")
    paired_test(lsd_eff_phi, plcb_eff_phi, "Effective Φ_PD:")
    paired_test(lsd_eff_lmax, plcb_eff_lmax, "Effective λ_max:")
    paired_test(lsd_te, plcb_te, "Integration (TE):")
    paired_test(lsd_jsd, plcb_jsd, "Raw Conflict (JSD):")
    paired_test(lsd_jsd_std, plcb_jsd_std, "JSD Volatility:")
    paired_test(lsd_coh, plcb_coh, "Coherence:")
    
    # =====================================================================
    # THE KEY TEST
    # =====================================================================
    print("\n" + "=" * 75)
    print("STEP 5: THE KEY TEST — Does effective conflict show LSD > Placebo?")
    print("=" * 75)
    
    raw_direction = "LSD ↑ (correct)" if np.mean(lsd_raw_phi) > np.mean(plcb_raw_phi) else "LSD ↓ (WRONG)"
    eff_direction = "LSD ↑ (correct)" if np.mean(lsd_eff_phi) > np.mean(plcb_eff_phi) else "LSD ↓ (WRONG)"
    
    print(f"\n  Raw TE×JSD direction:       {raw_direction}")
    print(f"  Effective conflict direction: {eff_direction}")
    
    # Cross-dataset comparison
    print(f"\n  === CROSS-DATASET PROXY COMPARISON ===")
    print(f"  Anesthesia (Awake vs Deep):")
    print(f"    Raw:       Deep > Awake (WRONG)")
    print(f"    Effective: Awake > Deep (CORRECT) ★")
    print(f"  Psychedelics (LSD vs Placebo):")
    print(f"    Raw:       {raw_direction}")
    print(f"    Effective: {eff_direction}")
    
    if np.mean(lsd_eff_phi) > np.mean(plcb_eff_phi):
        print(f"\n  ★ EFFECTIVE CONFLICT SHOWS CORRECT DIRECTION ON BOTH DATASETS")
        print(f"    The organized-conflict revision works for both propofol AND LSD.")
    
    # =====================================================================
    # SAVE
    # =====================================================================
    output_file = output_dir / "psychedelic_effective_conflict_results.json"
    
    def convert(obj):
        if isinstance(obj, (np.floating, float)): return float(obj)
        if isinstance(obj, (np.integer, int)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(i) for i in obj]
        return obj
    
    with open(output_file, 'w') as f:
        json.dump(convert({
            'n_subjects': n,
            'raw_lsd_phi_mean': np.mean(lsd_raw_phi),
            'raw_plcb_phi_mean': np.mean(plcb_raw_phi),
            'eff_lsd_phi_mean': np.mean(lsd_eff_phi),
            'eff_plcb_phi_mean': np.mean(plcb_eff_phi),
            'subject_results': subject_results,
        }), f, indent=2)
    print(f"\n  Saved: {output_file}")
    
    print("\n" + "=" * 75)
    print("PSYCHEDELIC EFFECTIVE CONFLICT TEST COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./psychedelic_data")
    parser.add_argument("--output-dir", default="./pmd_results")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    run_psychedelic_effective_conflict(args.data_dir, args.output_dir, args.skip_download)
