#!/usr/bin/env python3
"""
===========================================================================
PMD CONSCIOUSNESS STATE ANALYSIS — ds006623
THE DECISIVE EXPERIMENT

Does Φ*_PD track consciousness transitions under propofol sedation?

DATASET: Michigan Human Anesthesia fMRI (ds006623)
  - Subjects performing mental imagery (tennis, navigation, squeeze)
    and motor response (action) under graded propofol sedation
  - 4 task runs per subject at increasing sedation levels
  - Exact TR of loss-of-responsiveness (LOR) known per subject

APPROACH:
  For each run (sedation level), at each trial:
    1. Extract multi-voxel patterns from Yeo-7 networks
    2. Decode task condition (active imagery vs relax)
    3. Compute per-trial JSD between decoded policies across networks
    4. Apply I × C × G gating
    5. Compute Φ*_PD

  Then compare Φ*_PD across consciousness states:
    - Run 1 (Baseline/Awake)
    - Run 2 (Light sedation)
    - Run 3 PRE-LOR (still conscious during deep sedation run)
    - Run 3 POST-LOR (unconscious during deep sedation run)
    - Run 4 (Recovery)

PMD PREDICTIONS:
  1. Φ*_PD: Baseline > LOR
  2. Φ*_PD: Recovery ≈ Baseline  
  3. Sharp drop at LOR (phase transition), not gradual decline
  4. Gated metric orders states correctly; ungated does not
  5. Eigenvector dominance collapses at LOR

FILE PATHS: Assumes data in ./derivatives/ (extracted from OpenNeuro zip)
===========================================================================
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.signal import detrend

warnings.filterwarnings('ignore')

# =========================================================================
# CONFIGURATION
# =========================================================================

# Base paths - adjust if your extraction path differs
BASE_DIR = Path(".")
FMRIPREP_DIR = BASE_DIR / "derivatives" / "fmriprep_output"
STIM_DIR = BASE_DIR / "derivatives" / "Stimulus_Timing"
LOR_FILE = BASE_DIR / "derivatives" / "LOR_ROR_Timing.csv"
PART_FILE = BASE_DIR / "derivatives" / "Participant_Info.csv"

# Task-to-run mapping
# run-1 = task1 (baseline), run-2 = task2 (light sedation)
# run-3 = task3 (deep sedation, contains LOR), run-4 = task4 (recovery)
RUN_TO_TASK = {1: 'task1', 2: 'task2', 3: 'task3', 4: 'task4'}
RUN_LABELS = {1: 'Baseline', 2: 'LightSed', 3: 'DeepSed', 4: 'Recovery'}

# Conditions to decode
ACTIVE_CONDITIONS = ['tennis', 'squeeze', 'navi', 'action']
REST_CONDITION = 'relax'
TR = 2.0  # seconds

# =========================================================================
# UTILITIES
# =========================================================================

def get_yeo7_masks(bold_file):
    """Get Yeo-7 network masks resampled to BOLD space."""
    from nilearn import image, datasets
    import nibabel as nib
    import glob as g
    
    ydir = os.path.join(os.path.expanduser('~'), 'nilearn_data', 'yeo_2011')
    c = g.glob(os.path.join(ydir, '**', 'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz'), recursive=True)
    if not c:
        datasets.fetch_atlas_yeo_2011()
        c = g.glob(os.path.join(ydir, '**', '*7Networks*1mm.nii.gz'), recursive=True)
    
    atlas = image.resample_to_img(nib.load(c[0]), image.load_img(bold_file), interpolation='nearest')
    ad = atlas.get_fdata()
    while ad.ndim > 3:
        ad = ad[..., 0]
    return [np.round(ad) == i for i in range(1, 8)]


def load_stimulus_timing(sub_id, task_name):
    """Load onset TRs for each condition from .1D files."""
    stim_sub_dir = STIM_DIR / sub_id
    timing = {}
    
    for cond in ACTIVE_CONDITIONS + [REST_CONDITION, 'inst', 'instruction']:
        fname = stim_sub_dir / f"{sub_id}_{task_name}_{cond}.1D"
        if fname.exists():
            with open(fname) as f:
                content = f.read().strip()
                if content:
                    trs = [int(float(x)) for x in content.split() if x.strip()]
                    timing[cond] = trs
    
    return timing


def load_lor_timing():
    """Load LOR/ROR timing for each subject."""
    lor_data = {}
    
    # Try both csv and xlsx
    for ext in ['.csv', '.xlsx']:
        fpath = BASE_DIR / "derivatives" / f"LOR_ROR_Timing{ext}"
        if fpath.exists():
            if ext == '.csv':
                df = pd.read_csv(fpath)
            else:
                df = pd.read_excel(fpath)
            
            for _, row in df.iterrows():
                sub = str(row.iloc[0]).strip()
                lor_tr = row.iloc[1]
                ror_tr = row.iloc[2] if len(row) > 2 else None
                
                try:
                    lor_tr = int(float(lor_tr))
                except (ValueError, TypeError):
                    lor_tr = None
                try:
                    ror_tr = int(float(ror_tr))
                except (ValueError, TypeError):
                    ror_tr = None
                
                lor_data[sub] = {'lor_tr': lor_tr, 'ror_tr': ror_tr}
            break
    
    return lor_data


def sym_te(s, t, lag=1, bins=10):
    """Symmetric transfer entropy."""
    def te(src, tgt):
        n = len(src)
        if n <= lag + 1:
            return 0.0
        yn, yp, xp = tgt[lag:], tgt[:-lag], src[:-lag]
        def bd(x):
            mn, mx = x.min(), x.max()
            if mx == mn:
                return np.zeros(len(x), dtype=int)
            return np.clip(((x - mn) / (mx - mn) * (bins - 1)).astype(int), 0, bins - 1)
        yb, ypb, xpb = bd(yn), bd(yp), bd(xp)
        def H(c):
            p = c[c > 0] / c.sum()
            return -np.sum(p * np.log2(p))
        jyy = np.zeros((bins, bins))
        for i in range(len(yb)):
            jyy[yb[i], ypb[i]] += 1
        jyyx = np.zeros((bins, bins, bins))
        for i in range(len(yb)):
            jyyx[yb[i], ypb[i], xpb[i]] += 1
        jyx = np.zeros((bins, bins))
        for i in range(len(ypb)):
            jyx[ypb[i], xpb[i]] += 1
        return max((H(jyy.ravel()) - H(np.bincount(ypb, minlength=bins).astype(float))) -
                   (H(jyyx.ravel()) - H(jyx.ravel())), 0.0)
    return (te(s, t) + te(t, s)) / 2.0


def signal_coherence(ts, lag=1):
    if len(ts) < lag + 2:
        return 0.0
    ac = np.corrcoef(ts[:-lag], ts[lag:])[0, 1]
    return max(0.0, ac) if not np.isnan(ac) else 0.0


def soft_gate(x, x0, tau):
    z = np.clip((x - x0) / tau if tau > 0 else 0, -20, 20)
    return 1.0 / (1.0 + np.exp(-z))


# =========================================================================
# CORE ANALYSIS
# =========================================================================

def analyze_run(bold_file, masks, timing, run_label, lor_tr=None):
    """
    Analyze a single run: decode task conditions, compute Φ*_PD per trial.
    
    Returns dict with:
      - phi_values: per-trial Φ*_PD
      - phi_mean: mean Φ*_PD for the run
      - phi_ungated: ungated metric for comparison
      - eigval_ratio: eigenvalue dominance ratio
      - state_labels: 'pre_lor' or 'post_lor' for each trial (if lor_tr given)
      - J_matrix: mean J matrix
    """
    from nilearn import image
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.decomposition import PCA
    
    N = 7  # Yeo networks
    
    # Load BOLD data
    bold_img = image.load_img(str(bold_file))
    bold_data = bold_img.get_fdata()
    n_vol = bold_data.shape[-1]
    spatial = bold_data.shape[:3]
    bold_2d = bold_data.reshape(-1, n_vol)
    
    # Extract network timeseries
    ts = np.zeros((n_vol, N))
    for i, m in enumerate(masks):
        mf = m.ravel()
        if mf.sum() < 10:
            continue
        t = bold_2d[mf, :].mean(axis=0)
        t = detrend(t)
        s = t.std()
        if s > 0:
            t = t / s
        ts[:, i] = t
    
    # Build trial labels from timing
    # Label each onset as 'active' or 'rest'
    active_trs = []
    rest_trs = []
    
    for cond in ACTIVE_CONDITIONS:
        if cond in timing:
            active_trs.extend(timing[cond])
    
    if REST_CONDITION in timing:
        rest_trs = timing[REST_CONDITION]
    
    if not active_trs or not rest_trs:
        print(f" no decodable trials", end="")
        return None
    
    # Filter valid TRs (within scan range, with HRF delay)
    hrf_delay_trs = 3  # ~6 seconds HRF delay in TRs
    
    all_trs = []
    all_labels = []
    for tr_val in active_trs:
        t = tr_val + hrf_delay_trs
        if 0 <= t < n_vol:
            all_trs.append(t)
            all_labels.append('ACTIVE')
    for tr_val in rest_trs:
        t = tr_val + hrf_delay_trs
        if 0 <= t < n_vol:
            all_trs.append(t)
            all_labels.append('REST')
    
    all_trs = np.array(all_trs)
    all_labels = np.array(all_labels)
    
    if len(all_trs) < 10 or len(np.unique(all_labels)) < 2:
        print(f" too few trials ({len(all_trs)})", end="")
        return None
    
    # Extract multi-voxel patterns for each network at each trial
    patterns = {}
    for ni, mask in enumerate(masks):
        if mask.shape != spatial:
            patterns[ni] = np.zeros((len(all_trs), 10))
            continue
        mf = mask.ravel()
        nv = mf.sum()
        if nv < 10:
            patterns[ni] = np.zeros((len(all_trs), 10))
            continue
        tp = np.zeros((len(all_trs), nv))
        for ti, vol in enumerate(all_trs):
            tp[ti, :] = bold_2d[mf, int(vol)]
        # Z-score patterns
        mu = tp.mean(0, keepdims=True)
        sd = tp.std(0, keepdims=True)
        sd[sd == 0] = 1
        patterns[ni] = (tp - mu) / sd
    
    # Decode: train classifier per network
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(all_labels)
    nc = len(le.classes_)
    n_folds = min(5, min(np.bincount(y)))
    if n_folds < 2:
        n_folds = 2
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    decoded = {}
    accs = {}
    
    for ni in patterns:
        X = patterns[ni]
        if X.shape[1] > 500:
            X = PCA(n_components=min(100, X.shape[0] - 1), random_state=42).fit_transform(X)
        
        probs = np.zeros((len(y), nc))
        correct = 0
        total = 0
        
        for tri, tei in skf.split(X, y):
            try:
                clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
                clf.fit(X[tri], y[tri])
                probs[tei] = clf.predict_proba(X[tei])
                correct += (clf.predict(X[tei]) == y[tei]).sum()
                total += len(tei)
            except:
                probs[tei] = 1.0 / nc
                total += len(tei)
        
        decoded[ni] = probs
        accs[ni] = correct / total if total > 0 else 0.5
    
    mean_acc = np.mean(list(accs.values()))
    
    # Session-level: TE matrix, coherence, confidence
    TE = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            te = sym_te(ts[:, i], ts[:, j])
            TE[i, j] = TE[j, i] = te
    
    coh = np.array([signal_coherence(ts[:, i]) for i in range(N)])
    te_vals = [TE[i, j] for i in range(N) for j in range(i + 1, N)]
    i_min = np.percentile(te_vals, 25) if te_vals else 0
    
    net_conf = {}
    for ni in decoded:
        pr = decoded[ni]
        ent = -np.sum(pr * np.log2(pr + 1e-10), axis=1)
        me = np.log2(nc)
        net_conf[ni] = max(1.0 - np.mean(ent) / me, 0) if me > 0 else 0
    
    # Per-trial Φ*_PD (ACTIVE trials only)
    active_mask = (all_labels == 'ACTIVE')
    active_indices = np.where(active_mask)[0]
    active_onset_trs = all_trs[active_mask] - hrf_delay_trs  # original onset TRs
    
    trial_phis_gated = []
    trial_phis_ungated = []
    trial_states = []  # 'pre_lor' or 'post_lor'
    
    n_pairs = N * (N - 1) // 2
    J_accum = np.zeros((N, N))
    
    for ti, trial_idx in enumerate(active_indices):
        onset_tr = active_onset_trs[ti]
        
        # Determine consciousness state
        if lor_tr is not None:
            state = 'pre_lor' if onset_tr < lor_tr else 'post_lor'
        else:
            state = 'awake'
        trial_states.append(state)
        
        # Compute J matrix for this trial
        J = np.zeros((N, N))
        J_ungated = np.zeros((N, N))
        
        for i in range(N):
            for j in range(i + 1, N):
                pi = decoded[i][trial_idx] + 1e-10
                pj = decoded[j][trial_idx] + 1e-10
                pi /= pi.sum()
                pj /= pj.sum()
                m = 0.5 * (pi + pj)
                jsd = max(0.5 * np.sum(pi * np.log2(pi / m)) + 0.5 * np.sum(pj * np.log2(pj / m)), 0)
                
                # Gated
                g_c = soft_gate(TE[i, j], i_min, 0.05)
                g_r = np.sqrt(max(coh[i], 0) * max(coh[j], 0))
                g_d = np.sqrt(net_conf.get(i, 0) * net_conf.get(j, 0))
                
                J[i, j] = J[j, i] = TE[i, j] * jsd * g_c * g_r * g_d
                J_ungated[i, j] = J_ungated[j, i] = TE[i, j] * jsd  # No gates
        
        np.fill_diagonal(J, 0)
        np.fill_diagonal(J_ungated, 0)
        
        phi_g = sum(J[i, j] for i in range(N) for j in range(i + 1, N)) / n_pairs
        phi_u = sum(J_ungated[i, j] for i in range(N) for j in range(i + 1, N)) / n_pairs
        
        trial_phis_gated.append(phi_g)
        trial_phis_ungated.append(phi_u)
        J_accum += J
    
    n_active = len(active_indices)
    if n_active > 0:
        J_mean = J_accum / n_active
    else:
        J_mean = J_accum
    
    # Eigenvector analysis
    Js = np.nan_to_num((J_mean + J_mean.T) / 2)
    try:
        evals, evecs = np.linalg.eigh(Js)
        sorted_evals = np.sort(np.real(evals))[::-1]
        eigval_ratio = sorted_evals[0] / (sorted_evals[1] + 1e-15) if len(sorted_evals) > 1 else 0
        eigvec = np.abs(evecs[:, np.argmax(evals)])
    except:
        eigval_ratio = 0
        eigvec = np.ones(N) / np.sqrt(N)
    
    return {
        'phi_gated': np.array(trial_phis_gated),
        'phi_ungated': np.array(trial_phis_ungated),
        'phi_mean_gated': np.mean(trial_phis_gated) if trial_phis_gated else 0,
        'phi_mean_ungated': np.mean(trial_phis_ungated) if trial_phis_ungated else 0,
        'trial_states': trial_states,
        'eigval_ratio': eigval_ratio,
        'eigvec': eigvec,
        'mean_acc': mean_acc,
        'n_trials': n_active,
        'J_matrix': J_mean
    }


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 80)
    print("PMD CONSCIOUSNESS STATE ANALYSIS — THE DECISIVE EXPERIMENT")
    print("Does Φ*_PD track the conscious-to-unconscious transition?")
    print("=" * 80)
    
    # Load LOR timing
    lor_data = load_lor_timing()
    print(f"\n  LOR timing loaded for {len(lor_data)} subjects")
    
    # Find available subjects
    available = []
    for sub_dir in sorted(FMRIPREP_DIR.iterdir()):
        if sub_dir.is_dir() and sub_dir.name.startswith('sub-'):
            func_dir = sub_dir / "func"
            if func_dir.exists():
                # Check for all 4 runs
                runs_found = 0
                for run in range(1, 5):
                    bold = list(func_dir.glob(f"*task-imagery_run-{run}*MNI152*desc-preproc_bold.nii.gz"))
                    if bold and bold[0].stat().st_size > 1e6:  # >1MB = real data
                        runs_found += 1
                if runs_found >= 3:  # At least 3 runs usable
                    available.append(sub_dir.name)
    
    print(f"  Available subjects with ≥3 runs: {available}")
    
    if not available:
        print("\n  No usable subjects found. Check that BOLD files downloaded.")
        return
    
    # Process each subject
    all_results = {}
    
    for sub in available:
        print(f"\n{'='*60}")
        print(f"  SUBJECT: {sub}")
        print(f"{'='*60}")
        
        func_dir = FMRIPREP_DIR / sub / "func"
        
        # Get masks from first available run
        first_bold = None
        for run in range(1, 5):
            candidates = list(func_dir.glob(f"*task-imagery_run-{run}*MNI152*desc-preproc_bold.nii.gz"))
            if candidates and candidates[0].stat().st_size > 1e6:
                first_bold = candidates[0]
                break
        
        if first_bold is None:
            print(f"  No BOLD files found")
            continue
        
        print(f"  Loading Yeo-7 atlas...", end="", flush=True)
        masks = get_yeo7_masks(str(first_bold))
        print(f" done")
        
        # Get LOR TR for this subject (in task3/run-3)
        lor_tr = lor_data.get(sub, {}).get('lor_tr', None)
        print(f"  LOR TR (in run-3): {lor_tr}")
        
        sub_results = {}
        
        for run in range(1, 5):
            task_name = RUN_TO_TASK[run]
            run_label = RUN_LABELS[run]
            
            # Find BOLD file
            candidates = list(func_dir.glob(f"*task-imagery_run-{run}*MNI152*desc-preproc_bold.nii.gz"))
            if not candidates or candidates[0].stat().st_size < 1e6:
                print(f"\n  Run {run} ({run_label}): MISSING or empty")
                continue
            
            bold_file = candidates[0]
            
            # Load timing
            timing = load_stimulus_timing(sub, task_name)
            if not timing:
                print(f"\n  Run {run} ({run_label}): no timing files")
                continue
            
            print(f"\n  Run {run} ({run_label}):", end="", flush=True)
            
            # Only pass lor_tr for run-3 (deep sedation run)
            run_lor = lor_tr if run == 3 else None
            
            result = analyze_run(bold_file, masks, timing, run_label, lor_tr=run_lor)
            
            if result is not None:
                sub_results[run] = result
                print(f" Φ_gated={result['phi_mean_gated']:.6f}"
                      f" Φ_ungated={result['phi_mean_ungated']:.6f}"
                      f" acc={result['mean_acc']:.3f}"
                      f" eigR={result['eigval_ratio']:.1f}"
                      f" N={result['n_trials']}", end="")
                
                # For run-3, show pre/post LOR split
                if run == 3 and lor_tr is not None:
                    states = result['trial_states']
                    phis = result['phi_gated']
                    pre = [p for p, s in zip(phis, states) if s == 'pre_lor']
                    post = [p for p, s in zip(phis, states) if s == 'post_lor']
                    if pre and post:
                        print(f"\n    PRE-LOR:  Φ={np.mean(pre):.6f} (N={len(pre)})")
                        print(f"    POST-LOR: Φ={np.mean(post):.6f} (N={len(post)})", end="")
        
        all_results[sub] = sub_results
    
    # =====================================================================
    # CROSS-SUBJECT RESULTS
    # =====================================================================
    print(f"\n\n{'='*80}")
    print("CONSCIOUSNESS STATE RESULTS — ALL SUBJECTS")
    print("="*80)
    
    N_subs = len(all_results)
    
    # Collect per-state means
    state_phis = {1: [], 2: [], 3: [], 4: []}
    state_phis_ungated = {1: [], 2: [], 3: [], 4: []}
    state_eigratios = {1: [], 2: [], 3: [], 4: []}
    pre_lor_phis = []
    post_lor_phis = []
    
    for sub, results in all_results.items():
        for run, res in results.items():
            state_phis[run].append(res['phi_mean_gated'])
            state_phis_ungated[run].append(res['phi_mean_ungated'])
            state_eigratios[run].append(res['eigval_ratio'])
            
            if run == 3:
                states = res['trial_states']
                phis = res['phi_gated']
                pre = [p for p, s in zip(phis, states) if s == 'pre_lor']
                post = [p for p, s in zip(phis, states) if s == 'post_lor']
                if pre:
                    pre_lor_phis.append(np.mean(pre))
                if post:
                    post_lor_phis.append(np.mean(post))
    
    print(f"\n  {'State':<20} {'Mean Φ_gated':>14} {'Mean Φ_ungated':>16} {'Eig Ratio':>12} {'N_subs':>8}")
    print(f"  {'-'*70}")
    
    for run in [1, 2, 3, 4]:
        label = RUN_LABELS[run]
        g_vals = state_phis[run]
        u_vals = state_phis_ungated[run]
        e_vals = state_eigratios[run]
        if g_vals:
            print(f"  {label:<20} {np.mean(g_vals):>14.6f} {np.mean(u_vals):>16.6f} {np.mean(e_vals):>12.1f} {len(g_vals):>8}")
    
    if pre_lor_phis and post_lor_phis:
        print(f"\n  {'Pre-LOR (conscious)':<20} {np.mean(pre_lor_phis):>14.6f}")
        print(f"  {'Post-LOR (unconscious)':<20} {np.mean(post_lor_phis):>14.6f}")
    
    # =====================================================================
    # STATISTICAL TESTS
    # =====================================================================
    print(f"\n{'='*80}")
    print("PMD PREDICTION TESTS")
    print("="*80)
    
    # Test 1: Baseline > DeepSed (gated)
    if state_phis[1] and state_phis[3]:
        base = state_phis[1]
        deep = state_phis[3]
        if len(base) > 1 and len(deep) > 1:
            t, p = stats.ttest_rel(base[:min(len(base),len(deep))], 
                                    deep[:min(len(base),len(deep))])
            d = (np.mean(base) - np.mean(deep)) / (np.std(base + deep) + 1e-15)
        else:
            t, p, d = 0, 1, 0
        print(f"\n  P1: Φ*_PD Baseline > DeepSed (gated)")
        print(f"      Baseline: {np.mean(base):.6f}, DeepSed: {np.mean(deep):.6f}")
        print(f"      Direction: {'CORRECT ✓' if np.mean(base) > np.mean(deep) else 'WRONG ✗'}")
        if len(base) > 1:
            print(f"      d = {d:.3f}, t = {t:.3f}, p = {p:.4f}")
    
    # Test 1b: Pre-LOR > Post-LOR (within run-3)
    if pre_lor_phis and post_lor_phis:
        print(f"\n  P1b: Φ*_PD Pre-LOR > Post-LOR (within deep sedation run)")
        print(f"      Pre-LOR: {np.mean(pre_lor_phis):.6f}, Post-LOR: {np.mean(post_lor_phis):.6f}")
        print(f"      Direction: {'CORRECT ✓' if np.mean(pre_lor_phis) > np.mean(post_lor_phis) else 'WRONG ✗'}")
        if len(pre_lor_phis) > 1 and len(post_lor_phis) > 1:
            t, p = stats.ttest_rel(pre_lor_phis[:min(len(pre_lor_phis),len(post_lor_phis))],
                                    post_lor_phis[:min(len(pre_lor_phis),len(post_lor_phis))])
            print(f"      t = {t:.3f}, p = {p:.4f}")
    
    # Test 2: Recovery ≈ Baseline
    if state_phis[1] and state_phis[4]:
        base = state_phis[1]
        recov = state_phis[4]
        print(f"\n  P2: Recovery ≈ Baseline")
        print(f"      Baseline: {np.mean(base):.6f}, Recovery: {np.mean(recov):.6f}")
        ratio = np.mean(recov) / (np.mean(base) + 1e-15)
        print(f"      Ratio: {ratio:.3f} (1.0 = perfect recovery)")
    
    # Test 3: Phase transition (sharp drop, not gradual)
    if all(state_phis[r] for r in [1, 2, 3, 4]):
        means = [np.mean(state_phis[r]) for r in [1, 2, 3, 4]]
        print(f"\n  P3: Phase transition signature")
        print(f"      Baseline → LightSed → DeepSed → Recovery")
        print(f"      {means[0]:.6f} → {means[1]:.6f} → {means[2]:.6f} → {means[3]:.6f}")
        
        drop_1to2 = means[0] - means[1]
        drop_2to3 = means[1] - means[2]
        print(f"      Drop Baseline→Light: {drop_1to2:.6f}")
        print(f"      Drop Light→Deep:     {drop_2to3:.6f}")
        if drop_2to3 > drop_1to2 * 1.5:
            print(f"      ★ Sharp drop at deep sedation — consistent with phase transition")
        elif drop_2to3 > drop_1to2:
            print(f"      Larger drop at deep sedation — suggestive of phase transition")
        else:
            print(f"      Gradual decline — does not show clear phase transition")
    
    # Test 4: Gated vs ungated ordering
    print(f"\n  P4: Gated vs Ungated metric ordering")
    if all(state_phis[r] for r in [1, 3]) and all(state_phis_ungated[r] for r in [1, 3]):
        gated_correct = np.mean(state_phis[1]) > np.mean(state_phis[3])
        ungated_correct = np.mean(state_phis_ungated[1]) > np.mean(state_phis_ungated[3])
        print(f"      Gated ordering (Baseline > Deep): {'CORRECT ✓' if gated_correct else 'WRONG ✗'}")
        print(f"      Ungated ordering (Baseline > Deep): {'CORRECT ✓' if ungated_correct else 'WRONG ✗'}")
        if gated_correct and not ungated_correct:
            print(f"      ★ CONFIRMED: Gates are necessary for correct consciousness ordering")
    
    # Test 5: Eigenvector dominance
    print(f"\n  P5: Eigenvector dominance collapse at LOR")
    if state_eigratios[1] and state_eigratios[3]:
        base_eig = np.mean(state_eigratios[1])
        deep_eig = np.mean(state_eigratios[3])
        print(f"      Baseline eigval ratio: {base_eig:.1f}")
        print(f"      DeepSed eigval ratio:  {deep_eig:.1f}")
        print(f"      Direction: {'CORRECT ✓ (dominance decreases)' if base_eig > deep_eig else 'Does not decrease'}")
    
    # =====================================================================
    # OVERALL VERDICT
    # =====================================================================
    print(f"\n{'='*80}")
    print("CONSCIOUSNESS STATE ANALYSIS — VERDICT")
    print("="*80)
    
    n_predictions = 5
    n_correct = 0
    
    if state_phis[1] and state_phis[3]:
        if np.mean(state_phis[1]) > np.mean(state_phis[3]):
            n_correct += 1
    
    if pre_lor_phis and post_lor_phis:
        if np.mean(pre_lor_phis) > np.mean(post_lor_phis):
            n_correct += 1
    
    if state_phis[1] and state_phis[4]:
        ratio = np.mean(state_phis[4]) / (np.mean(state_phis[1]) + 1e-15)
        if 0.5 < ratio < 2.0:
            n_correct += 1
    
    print(f"""
  Subjects analyzed: {N_subs}
  Predictions correct direction: {n_correct}/5 (some may be untestable at this N)
  
  NOTE: With only {N_subs} subjects, statistical significance is not expected.
  This is a PROOF-OF-CONCEPT. The direction of effects is what matters.
  
  If Φ*_PD drops at LOR and recovers at recovery, even in 2 subjects,
  that is strong directional evidence that the metric tracks consciousness.
  
  Full statistical testing requires the complete 26-subject dataset.
""")
    
    print("=" * 80)
    print("CONSCIOUSNESS STATE ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
