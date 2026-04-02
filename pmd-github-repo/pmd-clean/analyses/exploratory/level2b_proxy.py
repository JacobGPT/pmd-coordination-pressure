#!/usr/bin/env python3
"""
===========================================================================
PRESSURE MAKES DIAMONDS — LEVEL 2B PROXY
Directional Opposition + Effective Conflict
===========================================================================

UPGRADE FROM 2A:
  2A gated raw JSD by stability, coupling, and coherence.
  2B adds DIRECTIONAL OPPOSITION STABILITY:
    - Measures whether two networks are consistently anticorrelated
      (organized opposition) vs randomly fluctuating in relationship
    - Windowed correlation sign consistency as opposition metric
    - This is the closest fMRI proxy to "policy disagreement" without
      actually decoding policies

  C_ij_2B = JSD × stability × coupling × coherence × opposition

  Where opposition = fraction of time windows where the networks
  are anticorrelated (pulling in genuinely opposite directions)

RUNS ON: Both anesthesia (ds003171) and psychedelic (ds003059) data
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
    from nilearn import maskers
    import glob
    yeo_dir = os.path.join(os.path.expanduser('~'), 'nilearn_data', 'yeo_2011')
    candidates = glob.glob(os.path.join(yeo_dir, '**', 'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz'), recursive=True)
    if not candidates:
        candidates = glob.glob(os.path.join(yeo_dir, '**', '*7Networks*1mm.nii.gz'), recursive=True)
    if not candidates:
        raise RuntimeError(f"Cannot find Yeo atlas")
    masker = maskers.NiftiLabelsMasker(
        labels_img=candidates[0], standardize='zscore_sample',
        detrend=True, memory='nilearn_cache', memory_level=1)
    ts = masker.fit_transform(bold_file)
    n_tp, n_net = ts.shape
    if n_tp < 30:
        raise ValueError(f"Too few timepoints: {n_tp}")
    if n_net != 7:
        pad = np.zeros((n_tp, 7))
        pad[:, :min(n_net, 7)] = ts[:, :min(n_net, 7)]
        ts = pad
    return ts


# =========================================================================
# MEASURES
# =========================================================================

def transfer_entropy(source, target, lag=1, bins=10):
    n = len(source)
    if n <= lag + 1:
        return 0.0
    y_now = target[lag:]
    y_past = target[:-lag]
    x_past = source[:-lag]
    def bin_data(x, b):
        mn, mx = x.min(), x.max()
        if mx == mn: return np.zeros(len(x), dtype=int)
        return np.clip(((x - mn) / (mx - mn) * (b - 1)).astype(int), 0, b - 1)
    yb = bin_data(y_now, bins)
    ypb = bin_data(y_past, bins)
    xpb = bin_data(x_past, bins)
    def H(c):
        p = c[c > 0] / c.sum()
        return -np.sum(p * np.log2(p))
    jyy = np.zeros((bins, bins))
    for i in range(len(yb)): jyy[yb[i], ypb[i]] += 1
    h_yy = H(jyy.ravel())
    h_yp = H(np.bincount(ypb, minlength=bins).astype(float))
    jyyx = np.zeros((bins, bins, bins))
    for i in range(len(yb)): jyyx[yb[i], ypb[i], xpb[i]] += 1
    h_yyx = H(jyyx.ravel())
    jyx = np.zeros((bins, bins))
    for i in range(len(ypb)): jyx[ypb[i], xpb[i]] += 1
    h_yx = H(jyx.ravel())
    return max((h_yy - h_yp) - (h_yyx - h_yx), 0.0)

def sym_te(ts_i, ts_j, lag=1, bins=10):
    return (transfer_entropy(ts_i, ts_j, lag, bins) +
            transfer_entropy(ts_j, ts_i, lag, bins)) / 2.0

def signal_jsd(ts_i, ts_j, bins=30):
    all_v = np.concatenate([ts_i, ts_j])
    mn, mx = all_v.min(), all_v.max()
    if mx == mn: return 0.0
    edges = np.linspace(mn, mx, bins + 1)
    p = np.histogram(ts_i, bins=edges)[0].astype(float)
    q = np.histogram(ts_j, bins=edges)[0].astype(float)
    eps = 1e-10
    p = (p + eps) / (p + eps).sum()
    q = (q + eps) / (q + eps).sum()
    m = 0.5 * (p + q)
    return max(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)), 0.0)

def windowed_jsd(ts_i, ts_j, win=60, step=30, bins=20):
    n = len(ts_i)
    if n < win: return signal_jsd(ts_i, ts_j, bins), 0.0
    vals = [signal_jsd(ts_i[s:s+win], ts_j[s:s+win], bins) for s in range(0, n-win+1, step)]
    return (np.mean(vals), np.std(vals)) if vals else (signal_jsd(ts_i, ts_j, bins), 0.0)

def signal_coherence(ts, lag=1):
    if len(ts) < lag + 2: return 0.0
    ac = np.corrcoef(ts[:-lag], ts[lag:])[0, 1]
    return max(0.0, ac) if not np.isnan(ac) else 0.0

def soft_gate(x, x0, tau):
    z = np.clip((x - x0) / tau if tau > 0 else 0, -20, 20)
    return 1.0 / (1.0 + np.exp(-z))

# =========================================================================
# NEW: DIRECTIONAL OPPOSITION STABILITY
# =========================================================================

def opposition_stability(ts_i, ts_j, win=60, step=30):
    """
    Measure how consistently two networks are in opposition.
    
    Computes windowed correlations and returns the fraction of windows
    where the correlation is negative (networks pulling in opposite
    directions). High opposition stability = consistent anticorrelation
    = organized policy disagreement. Low = random fluctuating relationship.
    
    Also returns the mean signed correlation for interpretation.
    """
    n = len(ts_i)
    if n < win:
        r = np.corrcoef(ts_i, ts_j)[0, 1]
        return (1.0 if r < 0 else 0.0, r if not np.isnan(r) else 0.0)
    
    corrs = []
    for s in range(0, n - win + 1, step):
        r = np.corrcoef(ts_i[s:s+win], ts_j[s:s+win])[0, 1]
        if not np.isnan(r):
            corrs.append(r)
    
    if not corrs:
        return 0.0, 0.0
    
    corrs = np.array(corrs)
    frac_negative = np.mean(corrs < 0)  # Fraction of windows anticorrelated
    mean_corr = np.mean(corrs)
    
    return frac_negative, mean_corr


# =========================================================================
# LEVEL 2B COMPUTATION
# =========================================================================

def compute_pmd_level2b(timeseries, i_min=None, tau=0.05):
    """
    Level 2B: TE × JSD × stability × coupling × coherence × opposition
    
    The opposition gate is the new addition. It measures whether the
    dissimilarity between networks reflects consistent directional
    opposition (real conflict) vs random relationship (noise).
    """
    N = timeseries.shape[1]
    
    # Pairwise measures
    TE = np.zeros((N, N))
    JSD_m = np.zeros((N, N))
    JSD_s = np.zeros((N, N))
    OPP_frac = np.zeros((N, N))  # Fraction of windows anticorrelated
    OPP_corr = np.zeros((N, N))  # Mean windowed correlation
    
    for i in range(N):
        for j in range(i + 1, N):
            te = sym_te(timeseries[:, i], timeseries[:, j])
            jm, js = windowed_jsd(timeseries[:, i], timeseries[:, j])
            of, mc = opposition_stability(timeseries[:, i], timeseries[:, j])
            TE[i,j] = TE[j,i] = te
            JSD_m[i,j] = JSD_m[j,i] = jm
            JSD_s[i,j] = JSD_s[j,i] = js
            OPP_frac[i,j] = OPP_frac[j,i] = of
            OPP_corr[i,j] = OPP_corr[j,i] = mc
    
    # Per-network coherence
    coh = np.array([signal_coherence(timeseries[:, i]) for i in range(N)])
    
    # I_min from data
    te_vals = [TE[i,j] for i in range(N) for j in range(i+1,N)]
    if i_min is None:
        i_min = np.percentile(te_vals, 25)
    
    # Build J matrices at multiple levels for comparison
    J_raw = np.zeros((N, N))        # Raw TE × JSD
    J_2a = np.zeros((N, N))         # Level 2A (stability + coupling + coherence)
    J_2b = np.zeros((N, N))         # Level 2B (+ opposition)
    
    pair_details = {}
    
    for i in range(N):
        for j in range(i + 1, N):
            te = TE[i,j]
            jm = JSD_m[i,j]
            js = JSD_s[i,j]
            opp = OPP_frac[i,j]
            
            # Gates
            g_stab = 1.0 / (1.0 + js)
            g_coup = soft_gate(te, i_min, tau)
            g_coh = np.sqrt(max(coh[i], 0) * max(coh[j], 0))
            
            # Opposition gate: scales from 0 (never anticorrelated) to 1 (always anticorrelated)
            # Use the raw fraction — networks that are consistently anticorrelated
            # have high opposition, networks with random relationship have ~0.5
            g_opp = opp  # Direct fraction of anticorrelated windows
            
            # Level computations
            j_raw = te * jm
            c_2a = jm * g_stab * g_coup * g_coh
            c_2b = jm * g_stab * g_coup * g_coh * g_opp
            j_2a = te * c_2a
            j_2b = te * c_2b
            
            J_raw[i,j] = J_raw[j,i] = j_raw
            J_2a[i,j] = J_2a[j,i] = j_2a
            J_2b[i,j] = J_2b[j,i] = j_2b
            
            pn = f"{YEO7_LABELS[i]}-{YEO7_LABELS[j]}"
            pair_details[pn] = {
                'te': float(te), 'jsd_mean': float(jm), 'jsd_std': float(js),
                'opp_frac': float(opp), 'mean_corr': float(OPP_corr[i,j]),
                'g_stab': float(g_stab), 'g_coup': float(g_coup),
                'g_coh': float(g_coh), 'g_opp': float(g_opp),
                'j_raw': float(j_raw), 'j_2a': float(j_2a), 'j_2b': float(j_2b),
            }
    
    for M in [J_raw, J_2a, J_2b]:
        np.fill_diagonal(M, 0)
    
    def metrics(J):
        phi = sum(J[i,j] for i in range(N) for j in range(i+1,N))
        np_ = N*(N-1)//2
        Js = np.nan_to_num((J+J.T)/2.0)
        try: ev = np.linalg.eigvalsh(Js)
        except: ev = np.real(np.linalg.eigvals(Js))
        ev = np.sort(np.real(ev))[::-1]
        return {'phi': float(phi/np_ if np_>0 else 0), 'lmax': float(ev[0])}
    
    r_raw = metrics(J_raw)
    r_2a = metrics(J_2a)
    r_2b = metrics(J_2b)
    
    mean_opp = float(np.mean([OPP_frac[i,j] for i in range(N) for j in range(i+1,N)]))
    
    return {
        'raw': r_raw, 'level_2a': r_2a, 'level_2b': r_2b,
        'mean_te': float(np.mean(te_vals)),
        'mean_jsd': float(np.mean([JSD_m[i,j] for i in range(N) for j in range(i+1,N)])),
        'mean_jsd_std': float(np.mean([JSD_s[i,j] for i in range(N) for j in range(i+1,N)])),
        'mean_coherence': float(np.mean(coh)),
        'mean_opposition': mean_opp,
        'pair_details': pair_details,
    }


# =========================================================================
# FILE FINDERS
# =========================================================================

def find_anesthesia_files(data_dir):
    data_dir = Path(data_dir)
    results = {}
    tmap = {'restawake':'Awake','restlight':'Mild Sedation','restdeep':'Deep Sedation','restrecovery':'Recovery'}
    for f in sorted(data_dir.glob("sub-*/func/*task-rest*bold.nii.gz")):
        parts = str(f).replace('\\','/')
        sub = next((p for p in parts.split('/') if p.startswith('sub-')), None)
        cond = next((v for k,v in tmap.items() if f"task-{k}" in f.name), None)
        if sub and cond:
            results.setdefault(sub, {})[cond] = str(f)
    return results

def find_psychedelic_files(data_dir):
    data_dir = Path(data_dir)
    results = {}
    for f in sorted(data_dir.glob("sub-*/ses-*/*task-rest*bold.nii.gz")):
        parts = str(f).replace('\\','/')
        sub = next((p for p in parts.split('/') if p.startswith('sub-')), None)
        cond = None
        for p in parts.split('/'):
            if 'ses-' in p:
                if 'LSD' in p.upper(): cond = 'LSD'
                elif 'PLCB' in p.upper(): cond = 'Placebo'
        if sub and cond:
            results.setdefault(sub, {}).setdefault(cond, []).append(str(f))
    return results


# =========================================================================
# ANALYSIS RUNNERS
# =========================================================================

def run_anesthesia(data_dir, output_dir):
    print("\n" + "=" * 75)
    print("ANESTHESIA ANALYSIS (Level 2B)")
    print("=" * 75)
    
    files = find_anesthesia_files(data_dir)
    if not files:
        print("  No anesthesia data found.")
        return
    print(f"  Subjects: {len(files)}")
    
    cond_data = {}
    for sub in sorted(files):
        print(f"\n  {sub}:")
        for cond, fp in sorted(files[sub].items()):
            print(f"    {cond}...", end="", flush=True)
            try:
                ts = extract_network_timeseries(fp)
                m = compute_pmd_level2b(ts)
                cond_data.setdefault(cond, []).append(m)
                print(f" raw={m['raw']['lmax']:.4f} 2A={m['level_2a']['lmax']:.4f} 2B={m['level_2b']['lmax']:.4f} opp={m['mean_opposition']:.3f}")
            except Exception as e:
                print(f" ERROR: {e}")
    
    print("\n  " + "-" * 70)
    print(f"  {'Condition':<20} {'Raw λ':>10} {'2A λ':>10} {'2B λ':>10} {'Opp':>8} {'Coh':>8}")
    print("  " + "-" * 70)
    
    order = ["Awake", "Mild Sedation", "Deep Sedation", "Recovery"]
    summary = {}
    for c in order:
        if c not in cond_data: continue
        ml = cond_data[c]
        raw = np.mean([m['raw']['lmax'] for m in ml])
        l2a = np.mean([m['level_2a']['lmax'] for m in ml])
        l2b = np.mean([m['level_2b']['lmax'] for m in ml])
        opp = np.mean([m['mean_opposition'] for m in ml])
        coh = np.mean([m['mean_coherence'] for m in ml])
        print(f"  {c:<20} {raw:>10.4f} {l2a:>10.4f} {l2b:>10.6f} {opp:>8.3f} {coh:>8.3f}")
        summary[c] = {
            'raw_lmax': [m['raw']['lmax'] for m in ml],
            'l2a_lmax': [m['level_2a']['lmax'] for m in ml],
            'l2b_lmax': [m['level_2b']['lmax'] for m in ml],
            'opposition': [m['mean_opposition'] for m in ml],
            'raw_mean': float(raw), 'l2a_mean': float(l2a), 'l2b_mean': float(l2b),
        }
    
    print("\n  KEY TEST — Awake vs Deep Sedation ordering:")
    if "Awake" in summary and "Deep Sedation" in summary:
        for level, key in [("Raw", "raw_mean"), ("Level 2A", "l2a_mean"), ("Level 2B", "l2b_mean")]:
            aw = summary["Awake"][key]
            dp = summary["Deep Sedation"][key]
            correct = aw > dp
            print(f"    {level:<12} Awake={aw:.6f} Deep={dp:.6f} {'CORRECT ★' if correct else 'WRONG'}")
    
    return summary


def run_psychedelic(data_dir, output_dir):
    print("\n" + "=" * 75)
    print("PSYCHEDELIC ANALYSIS (Level 2B)")
    print("=" * 75)
    
    files = find_psychedelic_files(data_dir)
    paired = {s: f for s, f in files.items() if 'LSD' in f and 'Placebo' in f}
    if not paired:
        print("  No paired psychedelic data found.")
        return
    print(f"  Paired subjects: {len(paired)}")
    
    lsd_metrics = []
    plcb_metrics = []
    
    for sub in sorted(paired):
        print(f"\n  {sub}:")
        for cond in ['LSD', 'Placebo']:
            runs = []
            for fp in paired[sub][cond]:
                print(f"    {cond}: {os.path.basename(fp)}...", end="", flush=True)
                try:
                    ts = extract_network_timeseries(fp)
                    m = compute_pmd_level2b(ts)
                    runs.append(m)
                    print(f" 2B={m['level_2b']['lmax']:.4f} opp={m['mean_opposition']:.3f}")
                except Exception as e:
                    print(f" ERROR: {e}")
            
            if runs:
                avg = {
                    'raw_lmax': float(np.mean([r['raw']['lmax'] for r in runs])),
                    'l2a_lmax': float(np.mean([r['level_2a']['lmax'] for r in runs])),
                    'l2b_lmax': float(np.mean([r['level_2b']['lmax'] for r in runs])),
                    'opposition': float(np.mean([r['mean_opposition'] for r in runs])),
                    'coherence': float(np.mean([r['mean_coherence'] for r in runs])),
                    'te': float(np.mean([r['mean_te'] for r in runs])),
                    'jsd': float(np.mean([r['mean_jsd'] for r in runs])),
                }
                if cond == 'LSD':
                    lsd_metrics.append(avg)
                else:
                    plcb_metrics.append(avg)
    
    n = min(len(lsd_metrics), len(plcb_metrics))
    if n == 0:
        print("  No paired results.")
        return
    
    print(f"\n  " + "-" * 70)
    print(f"  {'Metric':<20} {'LSD':>12} {'Placebo':>12} {'Direction':>12}")
    print("  " + "-" * 70)
    
    from scipy import stats
    
    for label, key in [("Raw λ_max", "raw_lmax"), ("Level 2A λ", "l2a_lmax"), 
                        ("Level 2B λ", "l2b_lmax"), ("Opposition", "opposition"),
                        ("Coherence", "coherence"), ("TE", "te"), ("JSD", "jsd")]:
        lv = [m[key] for m in lsd_metrics[:n]]
        pv = [m[key] for m in plcb_metrics[:n]]
        lm, pm = np.mean(lv), np.mean(pv)
        d = "LSD ↑" if lm > pm else "LSD ↓"
        t, p = stats.ttest_rel(lv, pv)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        diff_d = (np.mean(np.array(lv)-np.array(pv))) / (np.std(np.array(lv)-np.array(pv)) + 1e-10)
        print(f"  {label:<20} {lm:>12.6f} {pm:>12.6f} {d:>6} d={diff_d:+.3f} p={p:.4f} {sig}")
    
    print(f"\n  KEY TEST — LSD vs Placebo direction:")
    for level, key in [("Raw", "raw_lmax"), ("Level 2A", "l2a_lmax"), ("Level 2B", "l2b_lmax")]:
        lm = np.mean([m[key] for m in lsd_metrics[:n]])
        pm = np.mean([m[key] for m in plcb_metrics[:n]])
        correct = lm > pm
        print(f"    {level:<12} LSD={lm:.6f} Plcb={pm:.6f} {'LSD > Placebo ★' if correct else 'LSD < Placebo'}")
    
    return {'lsd': lsd_metrics, 'placebo': plcb_metrics}


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anesthesia-dir", default="./ds003171")
    parser.add_argument("--psychedelic-dir", default="./psychedelic_data")
    parser.add_argument("--output-dir", default="./pmd_results")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--anesthesia-only", action="store_true")
    parser.add_argument("--psychedelic-only", action="store_true")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 75)
    print("PRESSURE MAKES DIAMONDS — LEVEL 2B PROXY")
    print("All Three Proxy Levels Side by Side")
    print("=" * 75)
    print()
    print("  Raw:      TE × JSD (no gating)")
    print("  Level 2A: TE × JSD × stability × coupling × coherence")
    print("  Level 2B: TE × JSD × stability × coupling × coherence × OPPOSITION")
    print()
    print("  Opposition = fraction of time windows where networks are anticorrelated")
    print("  (consistently pulling in opposite directions = organized conflict)")
    print("=" * 75)
    
    results = {}
    
    if not args.psychedelic_only:
        anes = run_anesthesia(args.anesthesia_dir, args.output_dir)
        if anes:
            results['anesthesia'] = anes
    
    if not args.anesthesia_only:
        psych = run_psychedelic(args.psychedelic_dir, args.output_dir)
        if psych:
            results['psychedelic'] = psych
    
    # Save
    def convert(obj):
        if isinstance(obj, (np.floating, float)): return float(obj)
        if isinstance(obj, (np.integer, int)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(i) for i in obj]
        return obj
    
    with open(output_dir / "level_2b_results.json", 'w') as f:
        json.dump(convert(results), f, indent=2)
    
    print("\n" + "=" * 75)
    print("LEVEL 2B ANALYSIS COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
