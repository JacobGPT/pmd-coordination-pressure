#!/usr/bin/env python3
"""
==========================================================================
PRESSURE MAKES DIAMONDS — Information-Theoretic Psychedelic Analysis
==========================================================================

The primary analysis used BOLD correlation as a proxy for inter-domain
policy disagreement. This is a fundamentally limited proxy — correlation
measures signal co-fluctuation, not representational divergence.

This script uses information-theoretic measures that are closer to
what the framework actually describes:

1. Transfer Entropy: Directed information flow between networks
   (proxy for integration strength I_ij — measures causal influence)

2. Signal Divergence: Jensen-Shannon divergence between network
   amplitude distributions (proxy for conflict C_ij — measures whether
   networks are in different "states")

3. Spectral Dissimilarity: Divergence between power spectral density
   profiles (proxy for whether networks are operating in different
   frequency regimes — different "modes" of processing)

4. Multivariate Volatility: How much each network's pattern changes
   moment-to-moment (proxy for instability of domain policy)

Φ_PD_info = Σ TransferEntropy_ij × SignalDivergence_ij

This separates integration (directed information flow) from conflict
(distributional divergence) using independent measures — exactly what
the paper's limitations section identified as the recommended approach.

Usage: python pmd_info_theoretic_analysis.py --skip-download --data-dir ./psychedelic_data
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

# =========================================================================
# CONFIGURATION
# =========================================================================

LSD_SUBJECTS = [
    "sub-001", "sub-002", "sub-003", "sub-004", "sub-006",
    "sub-009", "sub-010", "sub-011", "sub-012", "sub-013",
    "sub-015", "sub-017", "sub-018", "sub-019", "sub-020"
]

LSD_SESSION = "ses-LSD"
PLACEBO_SESSION = "ses-PLCB"

YEO7_LABELS = ["Visual", "Somatomotor", "DorsalAttention",
               "VentralAttention", "Limbic", "Frontoparietal", "Default"]

# PMD domain mapping
PMD_DOMAINS = ["Sensory", "Emotional", "Social", "Reasoning", "Memory"]
YEO7_TO_PMD = {
    "Visual": "Sensory", "Somatomotor": "Sensory",
    "DorsalAttention": "Memory", "VentralAttention": "Memory",
    "Limbic": "Emotional", "Frontoparietal": "Reasoning", "Default": "Social",
}


# =========================================================================
# INFORMATION-THEORETIC MEASURES
# =========================================================================

def transfer_entropy(source, target, lag=1, bins=8):
    """
    Compute transfer entropy from source to target.
    
    TE(X→Y) = H(Y_t | Y_{t-lag}) - H(Y_t | Y_{t-lag}, X_{t-lag})
    
    Higher TE = more directed information flow = stronger integration.
    This is a proxy for I_ij that captures CAUSAL influence,
    not just co-fluctuation.
    """
    n = len(source) - lag
    if n < 20:
        return 0.0
    
    # Discretize signals
    src = np.digitize(source, np.linspace(source.min(), source.max(), bins)) - 1
    tgt = np.digitize(target, np.linspace(target.min(), target.max(), bins)) - 1
    
    # Joint and marginal distributions
    tgt_future = tgt[lag:]
    tgt_past = tgt[:-lag]
    src_past = src[:-lag]
    
    # H(Y_t | Y_{t-lag})
    joint_yy = np.zeros((bins, bins))
    for i in range(n):
        joint_yy[tgt_future[i], tgt_past[i]] += 1
    joint_yy /= joint_yy.sum() + 1e-10
    
    p_yt = joint_yy.sum(axis=1) + 1e-10
    p_yt_past = joint_yy.sum(axis=0) + 1e-10
    
    h_yt_given_past = -np.sum(joint_yy * np.log2(joint_yy / (p_yt_past[None, :] + 1e-10) + 1e-10))
    
    # H(Y_t | Y_{t-lag}, X_{t-lag})
    joint_yyx = np.zeros((bins, bins, bins))
    for i in range(n):
        joint_yyx[tgt_future[i], tgt_past[i], src_past[i]] += 1
    joint_yyx /= joint_yyx.sum() + 1e-10
    
    p_yx = joint_yyx.sum(axis=0) + 1e-10
    
    h_yt_given_past_src = -np.sum(joint_yyx * np.log2(joint_yyx / (p_yx[None, :, :] + 1e-10) + 1e-10))
    
    te = h_yt_given_past - h_yt_given_past_src
    return max(te, 0.0)  # TE is non-negative


def symmetric_transfer_entropy(sig_a, sig_b, lag=1, bins=8):
    """
    Symmetric transfer entropy: average of both directions.
    Measures bidirectional information flow = integration strength.
    """
    te_ab = transfer_entropy(sig_a, sig_b, lag, bins)
    te_ba = transfer_entropy(sig_b, sig_a, lag, bins)
    return (te_ab + te_ba) / 2.0


def signal_jsd(sig_a, sig_b, bins=20):
    """
    Jensen-Shannon divergence between amplitude distributions.
    
    This measures whether two networks are in different "states" — 
    different distributions over activity levels. High JSD = networks
    are doing fundamentally different things. Low JSD = similar activity
    patterns.
    
    This is a direct proxy for C_ij as defined in the framework:
    JS(P_i, P_j) where P_i is domain i's "policy distribution."
    The amplitude distribution is the closest fMRI proxy for
    what "policy distribution" means in neural terms.
    """
    # Create normalized histograms
    all_vals = np.concatenate([sig_a, sig_b])
    bin_edges = np.linspace(all_vals.min() - 0.01, all_vals.max() + 0.01, bins + 1)
    
    hist_a, _ = np.histogram(sig_a, bins=bin_edges, density=True)
    hist_b, _ = np.histogram(sig_b, bins=bin_edges, density=True)
    
    # Normalize to probability distributions
    hist_a = hist_a / (hist_a.sum() + 1e-10)
    hist_b = hist_b / (hist_b.sum() + 1e-10)
    
    # Add small epsilon for numerical stability
    hist_a = hist_a + 1e-10
    hist_b = hist_b + 1e-10
    hist_a = hist_a / hist_a.sum()
    hist_b = hist_b / hist_b.sum()
    
    return jensenshannon(hist_a, hist_b) ** 2  # Squared JSD for proper metric


def spectral_dissimilarity(sig_a, sig_b, fs=0.5):
    """
    Spectral JSD: divergence between power spectral density profiles.
    
    If two networks have different frequency profiles, they're operating
    in different temporal modes — which is a strong signal that they're
    doing different things (high policy disagreement).
    """
    # Compute PSDs
    freqs_a, psd_a = signal.welch(sig_a, fs=fs, nperseg=min(64, len(sig_a)//2))
    freqs_b, psd_b = signal.welch(sig_b, fs=fs, nperseg=min(64, len(sig_b)//2))
    
    # Normalize to distributions
    psd_a = psd_a / (psd_a.sum() + 1e-10) + 1e-10
    psd_b = psd_b / (psd_b.sum() + 1e-10) + 1e-10
    psd_a = psd_a / psd_a.sum()
    psd_b = psd_b / psd_b.sum()
    
    return jensenshannon(psd_a, psd_b) ** 2


def multivariate_volatility(timeseries, window=10):
    """
    Measure how much each network's signal changes moment-to-moment.
    High volatility = unstable policy = the domain keeps changing its mind.
    
    Returns per-network volatility scores.
    """
    N = timeseries.shape[1]
    volatilities = np.zeros(N)
    
    for i in range(N):
        sig = timeseries[:, i]
        # Rolling standard deviation
        vol = np.array([np.std(sig[max(0,t-window):t+1]) for t in range(len(sig))])
        volatilities[i] = np.mean(vol)
    
    return volatilities


def compute_phi_pd_info(timeseries, labels, use_7net=True):
    """
    Compute Φ_PD using information-theoretic measures.
    
    Φ_PD_info = Σ_{i<j} I_ij × C_ij
    
    where:
      I_ij = symmetric transfer entropy (directed information flow)
      C_ij = JSD of amplitude distributions (distributional divergence)
    
    Also computes:
      Φ_PD_spectral = Σ I_ij × SpectralJSD_ij  
      Φ_PD_combined = Σ I_ij × (α·SignalJSD + β·SpectralJSD)
    """
    N = timeseries.shape[1]
    
    # Handle zero-variance
    for col in range(N):
        if np.std(timeseries[:, col]) < 1e-10:
            timeseries[:, col] = np.random.randn(timeseries.shape[0]) * 1e-6
    
    # Compute pairwise measures
    TE_matrix = np.zeros((N, N))
    JSD_matrix = np.zeros((N, N))
    SJSD_matrix = np.zeros((N, N))
    
    pair_data = {}
    
    for i in range(N):
        for j in range(i+1, N):
            # Integration: symmetric transfer entropy
            te = symmetric_transfer_entropy(timeseries[:, i], timeseries[:, j])
            TE_matrix[i, j] = te
            TE_matrix[j, i] = te
            
            # Conflict: JSD of amplitude distributions
            jsd = signal_jsd(timeseries[:, i], timeseries[:, j])
            JSD_matrix[i, j] = jsd
            JSD_matrix[j, i] = jsd
            
            # Spectral conflict: JSD of PSD profiles
            sjsd = spectral_dissimilarity(timeseries[:, i], timeseries[:, j])
            SJSD_matrix[i, j] = sjsd
            SJSD_matrix[j, i] = sjsd
            
            pair_data[(labels[i], labels[j])] = {
                'TE': te,
                'JSD': jsd,
                'SpectralJSD': sjsd,
                'contribution_jsd': te * jsd,
                'contribution_spectral': te * sjsd,
                'contribution_combined': te * (0.5 * jsd + 0.5 * sjsd),
            }
    
    # Compute Φ_PD variants
    phi_pd_jsd = 0.0
    phi_pd_spectral = 0.0
    phi_pd_combined = 0.0
    n_pairs = 0
    
    for i in range(N):
        for j in range(i+1, N):
            phi_pd_jsd += TE_matrix[i, j] * JSD_matrix[i, j]
            phi_pd_spectral += TE_matrix[i, j] * SJSD_matrix[i, j]
            phi_pd_combined += TE_matrix[i, j] * (0.5 * JSD_matrix[i, j] + 0.5 * SJSD_matrix[i, j])
            n_pairs += 1
    
    # Normalize
    norm = 2.0 / (N * (N - 1)) if n_pairs > 0 else 1.0
    
    # Eigenanalysis on J = TE ⊙ JSD
    J_jsd = TE_matrix * JSD_matrix
    J_jsd = np.nan_to_num(J_jsd, nan=0.0)
    try:
        lambda_max_jsd = np.max(eigvalsh(J_jsd))
    except:
        lambda_max_jsd = 0.0
    
    J_combined = TE_matrix * (0.5 * JSD_matrix + 0.5 * SJSD_matrix)
    J_combined = np.nan_to_num(J_combined, nan=0.0)
    try:
        lambda_max_combined = np.max(eigvalsh(J_combined))
    except:
        lambda_max_combined = 0.0
    
    # Volatility per network
    volatilities = multivariate_volatility(timeseries)
    
    # Mean transfer entropy (overall integration)
    te_vals = TE_matrix[np.triu_indices(N, k=1)]
    mean_te = np.mean(te_vals)
    
    # Mean JSD (overall conflict)
    jsd_vals = JSD_matrix[np.triu_indices(N, k=1)]
    mean_jsd = np.mean(jsd_vals)
    
    # Mean spectral JSD
    sjsd_vals = SJSD_matrix[np.triu_indices(N, k=1)]
    mean_sjsd = np.mean(sjsd_vals)
    
    return {
        'phi_pd_jsd': phi_pd_jsd,
        'phi_pd_jsd_norm': phi_pd_jsd * norm,
        'phi_pd_spectral': phi_pd_spectral,
        'phi_pd_spectral_norm': phi_pd_spectral * norm,
        'phi_pd_combined': phi_pd_combined,
        'phi_pd_combined_norm': phi_pd_combined * norm,
        'lambda_max_jsd': lambda_max_jsd,
        'lambda_max_combined': lambda_max_combined,
        'mean_te': mean_te,
        'mean_jsd': mean_jsd,
        'mean_sjsd': mean_sjsd,
        'mean_volatility': np.mean(volatilities),
        'volatility_by_network': dict(zip(labels, volatilities.tolist())),
        'pair_data': pair_data,
        'n_domains': N,
    }


def extract_yeo7_timeseries(fmri_file):
    """Extract time series for all 7 Yeo networks."""
    from nilearn import datasets, maskers, image
    
    yeo = datasets.fetch_atlas_yeo_2011(n_networks=7)
    if hasattr(yeo, 'maps'):
        atlas_img = yeo.maps
    elif isinstance(yeo, dict) and 'thick_7' in yeo:
        atlas_img = yeo['thick_7']
    else:
        atlas_img = yeo['maps']
    
    masker = maskers.NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=True,
        detrend=True,
        low_pass=0.08,
        high_pass=0.01,
        t_r=2.0,
        memory='nilearn_cache',
        verbose=0
    )
    
    try:
        img = image.load_img(fmri_file)
        tr = img.header.get_zooms()[3]
        if 0 < tr < 10:
            masker.t_r = tr
    except:
        pass
    
    try:
        return masker.fit_transform(fmri_file)
    except Exception as e:
        print(f" FAILED ({e})")
        return None


def aggregate_to_pmd(timeseries):
    """Aggregate 7 Yeo networks to 5 PMD domains."""
    domain_ts = {}
    for i, label in enumerate(YEO7_LABELS):
        domain = YEO7_TO_PMD[label]
        if domain not in domain_ts:
            domain_ts[domain] = []
        domain_ts[domain].append(timeseries[:, i])
    
    result = np.zeros((timeseries.shape[0], len(PMD_DOMAINS)))
    for j, domain in enumerate(PMD_DOMAINS):
        if domain in domain_ts:
            result[:, j] = np.mean(domain_ts[domain], axis=0)
    return result


def process_subject(subject, data_dir, sessions):
    """Process one subject with information-theoretic analysis."""
    results = {}
    
    for session in sessions:
        session_dir = os.path.join(data_dir, subject, session)
        
        bold_files = sorted([
            os.path.join(session_dir, f) for f in os.listdir(session_dir)
            if f.endswith('_bold.nii.gz') and not f.startswith('._')
        ]) if os.path.exists(session_dir) else []
        
        if not bold_files:
            results[session] = None
            continue
        
        run_results_7 = []
        run_results_5 = []
        
        for bold_file in bold_files:
            print(f"    {os.path.basename(bold_file)}...", end="", flush=True)
            
            ts = extract_yeo7_timeseries(bold_file)
            if ts is None:
                continue
            
            # 7-network analysis
            r7 = compute_phi_pd_info(ts, YEO7_LABELS, use_7net=True)
            run_results_7.append(r7)
            
            # 5-domain analysis
            ts5 = aggregate_to_pmd(ts)
            r5 = compute_phi_pd_info(ts5, PMD_DOMAINS, use_7net=False)
            run_results_5.append(r5)
            
            print(f" OK (Φ_JSD={r7['phi_pd_jsd_norm']:.4f}, Φ_spec={r7['phi_pd_spectral_norm']:.4f})")
        
        if run_results_7:
            # Average metrics across runs
            avg = {}
            for key in ['phi_pd_jsd_norm', 'phi_pd_spectral_norm', 'phi_pd_combined_norm',
                        'lambda_max_jsd', 'lambda_max_combined',
                        'mean_te', 'mean_jsd', 'mean_sjsd', 'mean_volatility']:
                avg[f'7net_{key}'] = np.mean([r[key] for r in run_results_7])
                avg[f'5dom_{key}'] = np.mean([r[key] for r in run_results_5])
            
            avg['n_runs'] = len(run_results_7)
            
            # Pair-level data (7-network)
            pair_data = {}
            for pair in run_results_7[0]['pair_data']:
                for metric in ['TE', 'JSD', 'SpectralJSD', 'contribution_jsd', 'contribution_combined']:
                    key = f"{pair}_{metric}"
                    pair_data[(pair, metric)] = np.mean([r['pair_data'][pair][metric] for r in run_results_7])
            avg['pair_data_7net'] = pair_data
            
            results[session] = avg
        else:
            results[session] = None
    
    return results


def run_test(drug, placebo, name):
    """Paired t-test with effect size."""
    t, p = stats.ttest_rel(drug, placebo)
    diff = drug - placebo
    d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
    direction = "INCREASE" if np.mean(drug) > np.mean(placebo) else "DECREASE"
    try:
        w, wp = stats.wilcoxon(drug, placebo)
    except:
        w, wp = np.nan, np.nan
    return {'name': name, 'drug': np.mean(drug), 'placebo': np.mean(placebo),
            'direction': direction, 'diff': np.mean(diff),
            't': t, 'p': p, 'd': d, 'w': w, 'wp': wp, 'n': len(drug)}


def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    if p < 0.1: return "†"
    return "ns"


def print_result(r):
    sl = sig_label(r['p'])
    print(f"\n--- {r['name']} ---")
    print(f"  Drug:     {r['drug']:.6f}")
    print(f"  Placebo:  {r['placebo']:.6f}")
    print(f"  {r['direction']} | t({r['n']-1})={r['t']:.3f}, p={r['p']:.4f}, d={r['d']:.3f} {sl}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./psychedelic_data')
    parser.add_argument('--output-dir', default='./pmd_results')
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--subjects', type=int, default=None)
    args = parser.parse_args()
    
    print("=" * 75)
    print("PRESSURE MAKES DIAMONDS")
    print("Information-Theoretic Psychedelic Analysis")
    print("=" * 75)
    print(f"""
This analysis replaces correlation-based proxies with measures that
directly capture what the framework describes:

  Integration (I_ij): Transfer entropy — directed causal information flow
  Conflict (C_ij):    Jensen-Shannon divergence of amplitude distributions
                      + Spectral dissimilarity of frequency profiles

  Φ_PD_info = Σ TE_ij × JSD_ij

This is the recommended operational proxy from the paper's limitations
section: connectivity × disagreement, measured independently.
""")
    
    subjects = LSD_SUBJECTS[:args.subjects] if args.subjects else LSD_SUBJECTS
    sessions = [LSD_SESSION, PLACEBO_SESSION]
    
    all_data = []
    for subject in subjects:
        print(f"\n  {subject}:")
        results = process_subject(subject, args.data_dir, sessions)
        
        drug = results.get(LSD_SESSION)
        placebo = results.get(PLACEBO_SESSION)
        
        if drug is None or placebo is None:
            print(f"    Skipping — incomplete data")
            continue
        
        all_data.append({'subject': subject, 'drug': drug, 'placebo': placebo})
    
    if len(all_data) < 3:
        print("\nERROR: Too few subjects")
        return
    
    n = len(all_data)
    print(f"\n{'='*75}")
    print(f"RESULTS: {n} subjects")
    print(f"{'='*75}")
    
    # =====================================================================
    # 7-NETWORK RESULTS
    # =====================================================================
    print(f"\n{'='*75}")
    print("7-NETWORK ANALYSIS (Yeo-7, no aggregation)")
    print(f"{'='*75}")
    
    metrics_7 = [
        ('7net_phi_pd_jsd_norm', 'Φ_PD (TE × JSD)'),
        ('7net_phi_pd_spectral_norm', 'Φ_PD (TE × SpectralJSD)'),
        ('7net_phi_pd_combined_norm', 'Φ_PD (TE × Combined)'),
        ('7net_lambda_max_jsd', 'λ_max (TE × JSD)'),
        ('7net_lambda_max_combined', 'λ_max (TE × Combined)'),
        ('7net_mean_te', 'Mean Transfer Entropy (Integration)'),
        ('7net_mean_jsd', 'Mean JSD (Conflict)'),
        ('7net_mean_sjsd', 'Mean Spectral JSD (Conflict)'),
        ('7net_mean_volatility', 'Mean Volatility'),
    ]
    
    results_7 = []
    for key, name in metrics_7:
        drug_vals = np.array([s['drug'][key] for s in all_data])
        plac_vals = np.array([s['placebo'][key] for s in all_data])
        r = run_test(drug_vals, plac_vals, name)
        results_7.append(r)
        print_result(r)
    
    # =====================================================================
    # 5-DOMAIN RESULTS
    # =====================================================================
    print(f"\n{'='*75}")
    print("5-DOMAIN ANALYSIS (PMD pressure domains)")
    print(f"{'='*75}")
    
    metrics_5 = [
        ('5dom_phi_pd_jsd_norm', 'Φ_PD (TE × JSD)'),
        ('5dom_phi_pd_spectral_norm', 'Φ_PD (TE × SpectralJSD)'),
        ('5dom_phi_pd_combined_norm', 'Φ_PD (TE × Combined)'),
        ('5dom_lambda_max_jsd', 'λ_max (TE × JSD)'),
        ('5dom_mean_te', 'Mean Transfer Entropy'),
        ('5dom_mean_jsd', 'Mean JSD'),
        ('5dom_mean_sjsd', 'Mean Spectral JSD'),
    ]
    
    results_5 = []
    for key, name in metrics_5:
        drug_vals = np.array([s['drug'][key] for s in all_data])
        plac_vals = np.array([s['placebo'][key] for s in all_data])
        r = run_test(drug_vals, plac_vals, f"5-Domain {name}")
        results_5.append(r)
        print_result(r)
    
    # =====================================================================
    # PAIR-LEVEL ANALYSIS
    # =====================================================================
    print(f"\n{'='*75}")
    print("PAIR-LEVEL: Which pairs show increased conflict × integration?")
    print(f"{'='*75}")
    
    # Get all pairs from first subject
    all_pairs = set()
    for s in all_data:
        for (pair, metric) in s['drug']['pair_data_7net']:
            if metric == 'contribution_combined':
                all_pairs.add(pair)
    
    pair_results = []
    for pair in sorted(all_pairs):
        drug_vals = np.array([s['drug']['pair_data_7net'].get((pair, 'contribution_combined'), 0) for s in all_data])
        plac_vals = np.array([s['placebo']['pair_data_7net'].get((pair, 'contribution_combined'), 0) for s in all_data])
        r = run_test(drug_vals, plac_vals, f"{pair[0]}-{pair[1]}")
        pair_results.append(r)
    
    pair_results.sort(key=lambda r: r['d'], reverse=True)
    
    print(f"\n{'Pair':<45} {'Dir':>5} {'d':>8} {'p':>8}")
    print("-" * 70)
    for r in pair_results:
        sl = sig_label(r['p'])
        arrow = "↑" if r['direction'] == "INCREASE" else "↓"
        print(f"{r['name']:<45} {arrow:>5} {r['d']:>8.3f} {r['p']:>8.4f} {sl}")
    
    # =====================================================================
    # DECOMPOSED: Integration vs Conflict separately
    # =====================================================================
    print(f"\n{'='*75}")
    print("DECOMPOSED: Integration (TE) and Conflict (JSD) separately by pair")
    print(f"{'='*75}")
    
    print(f"\n--- Transfer Entropy (Integration) by pair ---")
    te_pairs = []
    for pair in sorted(all_pairs):
        drug_vals = np.array([s['drug']['pair_data_7net'].get((pair, 'TE'), 0) for s in all_data])
        plac_vals = np.array([s['placebo']['pair_data_7net'].get((pair, 'TE'), 0) for s in all_data])
        r = run_test(drug_vals, plac_vals, f"{pair[0]}-{pair[1]} TE")
        te_pairs.append(r)
    te_pairs.sort(key=lambda r: r['d'], reverse=True)
    print(f"{'Pair':<45} {'Dir':>5} {'d':>8} {'p':>8}")
    print("-" * 70)
    for r in te_pairs:
        sl = sig_label(r['p'])
        arrow = "↑" if r['direction'] == "INCREASE" else "↓"
        print(f"{r['name']:<45} {arrow:>5} {r['d']:>8.3f} {r['p']:>8.4f} {sl}")
    
    print(f"\n--- JSD (Conflict) by pair ---")
    jsd_pairs = []
    for pair in sorted(all_pairs):
        drug_vals = np.array([s['drug']['pair_data_7net'].get((pair, 'JSD'), 0) for s in all_data])
        plac_vals = np.array([s['placebo']['pair_data_7net'].get((pair, 'JSD'), 0) for s in all_data])
        r = run_test(drug_vals, plac_vals, f"{pair[0]}-{pair[1]} JSD")
        jsd_pairs.append(r)
    jsd_pairs.sort(key=lambda r: r['d'], reverse=True)
    print(f"{'Pair':<45} {'Dir':>5} {'d':>8} {'p':>8}")
    print("-" * 70)
    for r in jsd_pairs:
        sl = sig_label(r['p'])
        arrow = "↑" if r['direction'] == "INCREASE" else "↓"
        print(f"{r['name']:<45} {arrow:>5} {r['d']:>8.3f} {r['p']:>8.4f} {sl}")
    
    # =====================================================================
    # FRAMEWORK PREDICTION CHECK
    # =====================================================================
    print(f"\n{'='*75}")
    print("FRAMEWORK PREDICTION CHECK")
    print(f"{'='*75}")
    
    # Check the main prediction: Φ_PD increases
    for r in results_7:
        if 'Φ_PD (TE × Combined)' in r['name']:
            phi_result = r
            break
    
    print(f"\n  Primary metric: Φ_PD (Transfer Entropy × Combined JSD)")
    print(f"  Drug:     {phi_result['drug']:.6f}")
    print(f"  Placebo:  {phi_result['placebo']:.6f}")
    print(f"  Direction: {phi_result['direction']}")
    print(f"  p = {phi_result['p']:.4f}, d = {phi_result['d']:.3f}")
    
    if phi_result['direction'] == "INCREASE" and phi_result['p'] < 0.05:
        print(f"\n  ★ PREDICTION CONFIRMED: Φ_PD increases under LSD")
        print(f"    Using information-theoretic measures that separately capture")
        print(f"    integration (transfer entropy) and conflict (JSD).")
    elif phi_result['direction'] == "INCREASE" and phi_result['p'] < 0.1:
        print(f"\n  † TREND: Φ_PD shows trend toward increase under LSD")
        print(f"    Larger sample may be needed for significance.")
    elif phi_result['direction'] == "INCREASE":
        print(f"\n  → Φ_PD increases numerically but not significantly.")
    else:
        print(f"\n  ✗ Φ_PD decreases. Prediction not supported even with")
        print(f"    information-theoretic measures.")
    
    # Check decomposed: is it integration or conflict driving the change?
    for r in results_7:
        if r['name'] == 'Mean Transfer Entropy (Integration)':
            te_result = r
        if r['name'] == 'Mean JSD (Conflict)':
            jsd_result = r
    
    print(f"\n  Decomposition:")
    print(f"    Integration (TE):  {te_result['direction']} (p={te_result['p']:.4f}, d={te_result['d']:.3f})")
    print(f"    Conflict (JSD):    {jsd_result['direction']} (p={jsd_result['p']:.4f}, d={jsd_result['d']:.3f})")
    
    if jsd_result['direction'] == "INCREASE":
        print(f"\n    ★ CONFLICT INCREASES under LSD — consistent with framework core claim")
        if te_result['direction'] == "DECREASE":
            print(f"    Integration decreases — the C↑ × I↓ pattern the framework predicted")
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        'analysis': 'info_theoretic',
        'n_subjects': n,
        '7net_results': {r['name']: {'direction': r['direction'], 'p': float(r['p']), 'd': float(r['d']), 'sig': sig_label(r['p'])} for r in results_7},
        '5dom_results': {r['name']: {'direction': r['direction'], 'p': float(r['p']), 'd': float(r['d']), 'sig': sig_label(r['p'])} for r in results_5},
    }
    
    path = os.path.join(args.output_dir, 'LSD_info_theoretic.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: {path}")
    
    print(f"\n{'='*75}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
