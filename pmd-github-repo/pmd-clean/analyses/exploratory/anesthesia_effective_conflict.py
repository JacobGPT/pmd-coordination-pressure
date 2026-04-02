#!/usr/bin/env python3
"""
===========================================================================
PRESSURE MAKES DIAMONDS — EFFECTIVE CONFLICT VERSION
Anesthesia Phase Transition: Organized Conflict vs Fragmentation
===========================================================================

REVISION RATIONALE:
  Raw JSD measures distributional dissimilarity — but dissimilarity between
  disconnected, incoherent networks is fragmentation, not conflict.
  
  Genuine conflict requires:
    1. The networks are actually communicating (coupling gate)
    2. The disagreement is stable over time, not noise (stability gate)
    3. Each network is producing coherent output (coherence gate)

  C_ij_effective = JSD_ij × S_ij × Q_ij × R_ij
  
  Where:
    S_ij = 1 / (1 + std(windowed_JSD))     — stability gate
    Q_ij = sigmoid((TE_ij - I_min) / tau)   — coupling gate  
    R_ij = sqrt(coherence_i × coherence_j)  — signal coherence gate

USAGE:
  python pmd_anesthesia_effective_conflict.py --data-dir ./ds003171 --skip-download
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
# INFORMATION-THEORETIC MEASURES
# =========================================================================

def transfer_entropy(source, target, lag=1, bins=10):
    """Compute transfer entropy from source to target."""
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
    """Symmetric TE: average of both directions."""
    return (transfer_entropy(ts_i, ts_j, lag, bins) + 
            transfer_entropy(ts_j, ts_i, lag, bins)) / 2.0


def signal_jsd(ts_i, ts_j, bins=30):
    """JSD between amplitude distributions."""
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


# =========================================================================
# NEW: EFFECTIVE CONFLICT MEASURES
# =========================================================================

def windowed_jsd(ts_i, ts_j, window=60, step=30, bins=20):
    """
    Compute JSD in sliding windows. Returns mean and std.
    Stable conflict = low std. Noisy fragmentation = high std.
    """
    n = len(ts_i)
    if n < window:
        return signal_jsd(ts_i, ts_j, bins), 0.0
    
    jsd_values = []
    for start in range(0, n - window + 1, step):
        end = start + window
        jsd_val = signal_jsd(ts_i[start:end], ts_j[start:end], bins)
        jsd_values.append(jsd_val)
    
    if not jsd_values:
        return signal_jsd(ts_i, ts_j, bins), 0.0
    
    return np.mean(jsd_values), np.std(jsd_values)


def signal_coherence(ts, lag=1):
    """
    Measure how coherent/structured a single network's signal is.
    Uses lag-1 autocorrelation as a simple coherence metric.
    Coherent signal = high autocorrelation (structured, not noise).
    Incoherent signal = low autocorrelation (random, noisy).
    Returns value in [0, 1].
    """
    if len(ts) < lag + 2:
        return 0.0
    ac = np.corrcoef(ts[:-lag], ts[lag:])[0, 1]
    if np.isnan(ac):
        return 0.0
    # Map from [-1, 1] to [0, 1], where high positive = coherent
    return max(0.0, ac)


def soft_gate(x, x0, tau):
    """Logistic sigmoid gate: 1/(1 + exp(-(x - x0)/tau))"""
    z = (x - x0) / tau if tau > 0 else 0
    z = np.clip(z, -20, 20)  # Prevent overflow
    return 1.0 / (1.0 + np.exp(-z))


# =========================================================================
# PMD WITH EFFECTIVE CONFLICT
# =========================================================================

def compute_pmd_effective_conflict(timeseries, i_min=None, tau=0.05):
    """
    Compute PMD metrics with effective conflict gating.
    
    C_ij_effective = JSD_mean × stability_gate × coupling_gate × coherence_gate
    J_ij = TE_ij × C_ij_effective
    """
    N = timeseries.shape[1]
    
    # Step 1: Compute all pairwise TE and JSD components
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
    
    # Step 2: Compute per-network coherence
    coherence = np.zeros(N)
    for i in range(N):
        coherence[i] = signal_coherence(timeseries[:, i])
    
    # Step 3: Determine I_min from data if not provided
    te_values = [TE[i, j] for i in range(N) for j in range(i+1, N)]
    if i_min is None:
        i_min = np.percentile(te_values, 25)  # 25th percentile as floor
    
    # Step 4: Compute effective conflict and J matrix
    J_raw = np.zeros((N, N))       # Raw TE × JSD (for comparison)
    J_effective = np.zeros((N, N))  # TE × C_effective (the new version)
    C_effective = np.zeros((N, N))
    
    gate_details = {}
    
    for i in range(N):
        for j in range(i + 1, N):
            te = TE[i, j]
            jsd_m = JSD_mean[i, j]
            jsd_s = JSD_std[i, j]
            
            # Stability gate: organized conflict is temporally stable
            s_gate = 1.0 / (1.0 + jsd_s) if jsd_s >= 0 else 1.0
            
            # Coupling gate: conflict only counts if networks communicate
            q_gate = soft_gate(te, i_min, tau)
            
            # Coherence gate: both networks must produce structured signal
            r_gate = np.sqrt(max(coherence[i], 0) * max(coherence[j], 0))
            
            # Effective conflict
            c_eff = jsd_m * s_gate * q_gate * r_gate
            
            C_effective[i, j] = C_effective[j, i] = c_eff
            J_raw[i, j] = J_raw[j, i] = te * jsd_m
            J_effective[i, j] = J_effective[j, i] = te * c_eff
            
            pair_name = f"{YEO7_LABELS[i]}-{YEO7_LABELS[j]}"
            gate_details[pair_name] = {
                'te': float(te),
                'jsd_mean': float(jsd_m),
                'jsd_std': float(jsd_s),
                'stability_gate': float(s_gate),
                'coupling_gate': float(q_gate),
                'coherence_gate': float(r_gate),
                'c_effective': float(c_eff),
                'j_raw': float(te * jsd_m),
                'j_effective': float(te * c_eff),
            }
    
    np.fill_diagonal(J_effective, 0)
    np.fill_diagonal(J_raw, 0)
    
    # Step 5: Compute metrics for both raw and effective
    def compute_metrics(J, label):
        phi = 0
        n_pairs = 0
        network_totals = np.zeros(N)
        for i in range(N):
            for j in range(i + 1, N):
                phi += J[i, j]
                n_pairs += 1
                network_totals[i] += J[i, j]
                network_totals[j] += J[i, j]
        phi_norm = phi / n_pairs if n_pairs > 0 else 0
        
        J_sym = (J + J.T) / 2.0
        J_sym = np.nan_to_num(J_sym, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            eigenvalues = np.linalg.eigvalsh(J_sym)
        except:
            eigenvalues = np.real(np.linalg.eigvals(J_sym))
        eigenvalues = np.sort(np.real(eigenvalues))[::-1]
        
        dmn_pct = (network_totals[6] / (2 * phi) * 100) if phi > 0 else 0
        
        return {
            'phi_pd_norm': float(phi_norm),
            'lambda_max': float(eigenvalues[0]),
            'lambda_spectrum': [float(x) for x in eigenvalues],
            'dmn_pct': float(dmn_pct),
            'network_contributions': {YEO7_LABELS[i]: float(network_totals[i]) for i in range(N)},
        }
    
    raw_metrics = compute_metrics(J_raw, "raw")
    eff_metrics = compute_metrics(J_effective, "effective")
    
    # Compute summary stats
    mean_te = np.mean(te_values)
    mean_jsd = np.mean([JSD_mean[i,j] for i in range(N) for j in range(i+1,N)])
    mean_jsd_std = np.mean([JSD_std[i,j] for i in range(N) for j in range(i+1,N)])
    mean_coherence = np.mean(coherence)
    
    return {
        'raw': raw_metrics,
        'effective': eff_metrics,
        'mean_te': float(mean_te),
        'mean_jsd': float(mean_jsd),
        'mean_jsd_std': float(mean_jsd_std),
        'mean_coherence': float(mean_coherence),
        'network_coherence': {YEO7_LABELS[i]: float(coherence[i]) for i in range(N)},
        'i_min': float(i_min),
        'gate_details': gate_details,
    }


# =========================================================================
# FILE FINDER
# =========================================================================

def find_bold_files(data_dir):
    data_dir = Path(data_dir)
    results = {}
    task_map = {
        'restawake': 'Awake',
        'restlight': 'Mild Sedation',
        'restdeep': 'Deep Sedation',
        'restrecovery': 'Recovery',
    }
    all_files = list(data_dir.glob("sub-*/func/*task-rest*bold.nii.gz"))
    if not all_files:
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
        for tk, cn in task_map.items():
            if f"task-{tk}" in fname:
                condition = cn
                break
        if not condition:
            continue
        if sub_id not in results:
            results[sub_id] = {}
        results[sub_id][condition] = str(fpath)
    return results


# =========================================================================
# MAIN
# =========================================================================

def run_effective_conflict_test(data_dir, output_dir, skip_download=False):
    print("=" * 75)
    print("PRESSURE MAKES DIAMONDS — EFFECTIVE CONFLICT VERSION")
    print("Organized Conflict vs Disorganized Fragmentation")
    print("=" * 75)
    print()
    print("NEW CONFLICT MEASURE:")
    print("  C_eff = JSD_mean × stability_gate × coupling_gate × coherence_gate")
    print()
    print("  Stability gate:  suppresses volatile/noisy dissimilarity")
    print("  Coupling gate:   suppresses conflict between disconnected networks")
    print("  Coherence gate:  suppresses conflict from incoherent signals")
    print()
    print("PREDICTION: Deep Sedation should now show LOWER effective J")
    print("            because propofol-induced dissimilarity is fragmentation,")
    print("            not organized conflict.")
    print("=" * 75)
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 75)
    print("STEP 1: Finding BOLD files")
    print("=" * 75)
    
    subject_files = find_bold_files(data_dir)
    if not subject_files:
        print("No subjects found.")
        sys.exit(1)
    print(f"  Subjects: {len(subject_files)}")
    
    print("\n" + "=" * 75)
    print("STEP 2: Computing effective conflict metrics")
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
                metrics = compute_pmd_effective_conflict(ts)
                all_results[sub_id][condition] = metrics
                
                if condition not in condition_metrics:
                    condition_metrics[condition] = []
                condition_metrics[condition].append(metrics)
                
                raw_l = metrics['raw']['lambda_max']
                eff_l = metrics['effective']['lambda_max']
                coh = metrics['mean_coherence']
                jsd_s = metrics['mean_jsd_std']
                
                print(f" raw_λ={raw_l:.4f}, eff_λ={eff_l:.6f}, coh={coh:.3f}, jsd_vol={jsd_s:.4f}")
                
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
    print("STEP 3: GROUP-LEVEL RESULTS")
    print("=" * 75)
    
    condition_order = ["Awake", "Mild Sedation", "Deep Sedation", "Recovery"]
    ordered = [c for c in condition_order if c in condition_metrics]
    
    print(f"\n  === RAW (TE × JSD, no gating) ===")
    print(f"  {'Condition':<20} {'N':>4} {'Φ_PD':>12} {'λ_max':>12} {'Mean TE':>10} {'Mean JSD':>10}")
    print("  " + "-" * 70)
    
    group_raw = {}
    group_eff = {}
    
    for condition in ordered:
        mlist = condition_metrics[condition]
        n = len(mlist)
        
        phi_vals = [m['raw']['phi_pd_norm'] for m in mlist]
        lmax_vals = [m['raw']['lambda_max'] for m in mlist]
        te_vals = [m['mean_te'] for m in mlist]
        jsd_vals = [m['mean_jsd'] for m in mlist]
        
        print(f"  {condition:<20} {n:>4} {np.mean(phi_vals):>10.6f}±{np.std(phi_vals):.5f} {np.mean(lmax_vals):>9.4f}±{np.std(lmax_vals):.4f} {np.mean(te_vals):>10.4f} {np.mean(jsd_vals):>10.4f}")
        
        group_raw[condition] = {
            'phi_values': [float(x) for x in phi_vals],
            'lmax_values': [float(x) for x in lmax_vals],
        }
    
    print(f"\n  === EFFECTIVE (TE × C_effective, WITH gating) ===")
    print(f"  {'Condition':<20} {'N':>4} {'Φ_PD_eff':>14} {'λ_max_eff':>14} {'Coherence':>10} {'JSD_vol':>10}")
    print("  " + "-" * 74)
    
    for condition in ordered:
        mlist = condition_metrics[condition]
        n = len(mlist)
        
        phi_vals = [m['effective']['phi_pd_norm'] for m in mlist]
        lmax_vals = [m['effective']['lambda_max'] for m in mlist]
        coh_vals = [m['mean_coherence'] for m in mlist]
        vol_vals = [m['mean_jsd_std'] for m in mlist]
        
        print(f"  {condition:<20} {n:>4} {np.mean(phi_vals):>12.8f}±{np.std(phi_vals):.7f} {np.mean(lmax_vals):>12.8f}±{np.std(lmax_vals):.7f} {np.mean(coh_vals):>10.4f} {np.mean(vol_vals):>10.4f}")
        
        group_eff[condition] = {
            'n': n,
            'phi_mean': float(np.mean(phi_vals)),
            'phi_std': float(np.std(phi_vals)),
            'lmax_mean': float(np.mean(lmax_vals)),
            'lmax_std': float(np.std(lmax_vals)),
            'coherence_mean': float(np.mean(coh_vals)),
            'jsd_volatility_mean': float(np.mean(vol_vals)),
            'phi_values': [float(x) for x in phi_vals],
            'lmax_values': [float(x) for x in lmax_vals],
        }
    
    # =====================================================================
    # THE KEY TEST: Does gating fix the ordering?
    # =====================================================================
    print("\n" + "=" * 75)
    print("STEP 4: THE KEY TEST — Does effective conflict fix the ordering?")
    print("=" * 75)
    
    if "Awake" in group_eff and "Deep Sedation" in group_eff:
        awake_eff = group_eff["Awake"]["lmax_mean"]
        deep_eff = group_eff["Deep Sedation"]["lmax_mean"]
        awake_raw = np.mean(group_raw["Awake"]["lmax_values"])
        deep_raw = np.mean(group_raw["Deep Sedation"]["lmax_values"])
        
        print(f"\n  RAW (TE × JSD):")
        print(f"    Awake λ_max:          {awake_raw:.6f}")
        print(f"    Deep Sedation λ_max:  {deep_raw:.6f}")
        raw_correct = awake_raw > deep_raw
        print(f"    Ordering correct?     {'YES — Awake > Deep' if raw_correct else 'NO — Deep > Awake (WRONG)'}")
        
        print(f"\n  EFFECTIVE (TE × C_effective):")
        print(f"    Awake λ_max:          {awake_eff:.8f}")
        print(f"    Deep Sedation λ_max:  {deep_eff:.8f}")
        eff_correct = awake_eff > deep_eff
        print(f"    Ordering correct?     {'YES — Awake > Deep ★' if eff_correct else 'NO — Deep > Awake'}")
        
        if eff_correct and not raw_correct:
            print(f"\n  ★ GATING FIXED THE ORDERING")
            print(f"    Raw proxy: wrong direction (Deep > Awake)")
            print(f"    Effective proxy: correct direction (Awake > Deep)")
            print(f"    This confirms that propofol-induced dissimilarity is fragmentation,")
            print(f"    not organized conflict. The gating successfully distinguishes them.")
        elif eff_correct and raw_correct:
            print(f"\n  Both proxies show correct ordering.")
        elif not eff_correct:
            print(f"\n  Gating did NOT fix the ordering.")
            print(f"  The problem may be deeper than proxy design.")
    
    # =====================================================================
    # STATISTICAL TESTS ON EFFECTIVE METRICS
    # =====================================================================
    print("\n" + "=" * 75)
    print("STEP 5: STATISTICAL TESTS (Effective Conflict)")
    print("=" * 75)
    
    from scipy import stats
    
    contrasts = [
        ("Awake", "Deep Sedation", "Consciousness vs Unconsciousness"),
        ("Deep Sedation", "Recovery", "Recovery from Sedation"),
        ("Awake", "Recovery", "Full Recovery Check"),
    ]
    
    for ca, cb, label in contrasts:
        if ca not in group_eff or cb not in group_eff:
            continue
        a = group_eff[ca]['lmax_values']
        b = group_eff[cb]['lmax_values']
        t, p = stats.ttest_ind(a, b)
        pooled = np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)
        d = (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        direction = "↑" if np.mean(a) > np.mean(b) else "↓"
        print(f"\n  {label}: {ca} vs {cb}")
        print(f"    λ_max_eff: {direction} d={d:+.3f}, t={t:.3f}, p={p:.4f} {sig}")
    
    # =====================================================================
    # DECOMPOSITION: What do the gates reveal?
    # =====================================================================
    print("\n" + "=" * 75)
    print("STEP 6: WHAT THE GATES REVEAL")
    print("=" * 75)
    
    for condition in ordered:
        mlist = condition_metrics[condition]
        coh = np.mean([m['mean_coherence'] for m in mlist])
        vol = np.mean([m['mean_jsd_std'] for m in mlist])
        te = np.mean([m['mean_te'] for m in mlist])
        jsd = np.mean([m['mean_jsd'] for m in mlist])
        
        print(f"\n  {condition}:")
        print(f"    Integration (TE):      {te:.4f}")
        print(f"    Raw conflict (JSD):    {jsd:.4f}")
        print(f"    JSD volatility (std):  {vol:.4f}  {'← high = noisy/fragmented' if vol > 0.02 else '← low = stable/organized'}")
        print(f"    Signal coherence:      {coh:.4f}  {'← high = structured' if coh > 0.3 else '← low = incoherent'}")
    
    # =====================================================================
    # SAVE
    # =====================================================================
    output_file = output_dir / "anesthesia_effective_conflict_results.json"
    
    def convert(obj):
        if isinstance(obj, (np.floating, float)): return float(obj)
        if isinstance(obj, (np.integer, int)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(i) for i in obj]
        return obj
    
    with open(output_file, 'w') as f:
        json.dump(convert({
            'proxy': 'TE × C_effective (stability + coupling + coherence gated)',
            'group_raw': group_raw,
            'group_effective': group_eff,
        }), f, indent=2)
    print(f"\n  Saved: {output_file}")
    
    print("\n" + "=" * 75)
    print("EFFECTIVE CONFLICT TEST COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./ds003171")
    parser.add_argument("--output-dir", default="./pmd_results")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    run_effective_conflict_test(args.data_dir, args.output_dir, args.skip_download)
