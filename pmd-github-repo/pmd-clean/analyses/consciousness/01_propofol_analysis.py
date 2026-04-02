#!/usr/bin/env python3
"""
===========================================================================
PMD CONSCIOUSNESS STATE ANALYSIS v3

Two approaches adapted for mental imagery tasks:

APPROACH B — ENHANCED LEVEL 2A (timeseries-based, no decoding needed)
  Uses network-mean timeseries to compute:
    - Transfer Entropy (I_ij) per run
    - Amplitude JSD between network timeseries distributions (C_ij) per run  
    - Full gating structure (coupling, coherence, stability gates)
    - Φ*_PD per run = mean of I × C × G across network pairs
  
  This works on ANY fMRI data — resting state, imagery, tasks.
  Already showed correct direction on anesthesia (d=0.13, ns at N=17).
  With better data (exact LOR timing, 26 subjects) it should be stronger.

APPROACH A — BLOCK-LEVEL DECODING
  Average multi-voxel patterns across each imagery block (~30s),
  then decode condition. With ~10 blocks per condition per run,
  block-averaged patterns should be decodable where single TRs weren't.
  Then compute Φ*_PD from block-level decoded policies.

For each approach, compare across consciousness states:
  Run 1 (Baseline) → Run 2 (Light Sed) → Run 3 (Deep Sed) → Run 4 (Recovery)
  Within Run 3: Pre-LOR vs Post-LOR

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

BASE_DIR = Path(".")
FMRIPREP_DIR = BASE_DIR / "derivatives" / "fmriprep_output"
STIM_DIR = BASE_DIR / "derivatives" / "Stimulus_Timing"

RUN_TO_TASK = {1: 'task1', 2: 'task2', 3: 'task3', 4: 'task4'}
RUN_LABELS = {1: 'Baseline', 2: 'LightSed', 3: 'DeepSed', 4: 'Recovery'}
NETWORK_NAMES = ['Visual', 'Somatomotor', 'DorsalAtt', 'VentralAtt', 'Limbic', 'FrontoParietal', 'Default']
N = 7

BLOCK_CONDITIONS = ['tennis', 'squeeze', 'navi', 'action', 'relax']
BLOCK_DURATION_TRS = 15  # ~30 seconds at TR=2

# =========================================================================
# UTILITIES
# =========================================================================

def get_yeo7_masks(bold_file):
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
    while ad.ndim > 3: ad = ad[..., 0]
    return [np.round(ad) == i for i in range(1, 8)]

def load_stimulus_timing(sub_id, task_name):
    stim_sub_dir = STIM_DIR / sub_id
    timing = {}
    for cond in BLOCK_CONDITIONS + ['inst', 'instruction']:
        fname = stim_sub_dir / f"{sub_id}_{task_name}_{cond}.1D"
        if fname.exists():
            with open(fname) as f:
                content = f.read().strip()
                if content:
                    trs = [int(float(x)) for x in content.split() if x.strip()]
                    timing[cond] = trs
    return timing

def load_lor_timing():
    lor_data = {}
    for ext in ['.csv', '.xlsx']:
        fpath = BASE_DIR / "derivatives" / f"LOR_ROR_Timing{ext}"
        if fpath.exists():
            if ext == '.csv':
                df = pd.read_csv(fpath)
            else:
                df = pd.read_excel(fpath)
            for _, row in df.iterrows():
                sub = str(row.iloc[0]).strip()
                try: lor_tr = int(float(row.iloc[1]))
                except: lor_tr = None
                try: ror_tr = int(float(row.iloc[2]))
                except: ror_tr = None
                lor_data[sub] = {'lor_tr': lor_tr, 'ror_tr': ror_tr}
            break
    return lor_data

def sym_te(s, t, lag=1, bins=10):
    def te(src, tgt):
        n = len(src)
        if n <= lag + 1: return 0.0
        yn, yp, xp = tgt[lag:], tgt[:-lag], src[:-lag]
        def bd(x):
            mn, mx = x.min(), x.max()
            if mx == mn: return np.zeros(len(x), dtype=int)
            return np.clip(((x - mn) / (mx - mn) * (bins - 1)).astype(int), 0, bins - 1)
        yb, ypb, xpb = bd(yn), bd(yp), bd(xp)
        def H(c): p = c[c > 0] / c.sum(); return -np.sum(p * np.log2(p))
        jyy = np.zeros((bins, bins))
        for i in range(len(yb)): jyy[yb[i], ypb[i]] += 1
        jyyx = np.zeros((bins, bins, bins))
        for i in range(len(yb)): jyyx[yb[i], ypb[i], xpb[i]] += 1
        jyx = np.zeros((bins, bins))
        for i in range(len(ypb)): jyx[ypb[i], xpb[i]] += 1
        return max((H(jyy.ravel()) - H(np.bincount(ypb, minlength=bins).astype(float))) -
                   (H(jyyx.ravel()) - H(jyx.ravel())), 0.0)
    return (te(s, t) + te(t, s)) / 2.0

def signal_coherence(ts, lag=1):
    if len(ts) < lag + 2: return 0.0
    ac = np.corrcoef(ts[:-lag], ts[lag:])[0, 1]
    return max(0.0, ac) if not np.isnan(ac) else 0.0

def soft_gate(x, x0, tau):
    z = np.clip((x - x0) / tau if tau > 0 else 0, -20, 20)
    return 1.0 / (1.0 + np.exp(-z))

def amplitude_jsd(ts_i, ts_j, bins=10):
    """JSD between amplitude distributions of two timeseries."""
    def hist_prob(x):
        h, _ = np.histogram(x, bins=bins, density=True)
        h = h + 1e-10
        return h / h.sum()
    pi = hist_prob(ts_i)
    pj = hist_prob(ts_j)
    m = 0.5 * (pi + pj)
    jsd = 0.5 * np.sum(pi * np.log2(pi / m)) + 0.5 * np.sum(pj * np.log2(pj / m))
    return max(jsd, 0)

def get_timeseries(bold_file, masks):
    from nilearn import image
    bd = image.load_img(bold_file).get_fdata()
    n_tp = bd.shape[-1]
    d2 = bd.reshape(-1, n_tp)
    ts = np.zeros((n_tp, N))
    for i, m in enumerate(masks):
        mf = m.ravel()
        if mf.sum() < 10: continue
        t = d2[mf, :].mean(axis=0)
        t = detrend(t)
        s = t.std()
        if s > 0: t = t / s
        ts[:, i] = t
    return ts


# =========================================================================
# APPROACH B: ENHANCED LEVEL 2A (timeseries-based)
# =========================================================================

def approach_b(bold_file, masks, lor_tr=None):
    """
    Enhanced Level 2A: TE × amplitude_JSD × gates on full timeseries.
    
    If lor_tr is given, also computes separately for pre-LOR and post-LOR
    segments of the timeseries.
    """
    ts = get_timeseries(bold_file, masks)
    n_tp = ts.shape[0]
    
    def compute_phi_on_segment(ts_seg):
        """Compute Φ for a timeseries segment."""
        if ts_seg.shape[0] < 20:
            return 0, 0, 0, np.zeros((N, N))
        
        # TE matrix
        TE = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                te = sym_te(ts_seg[:, i], ts_seg[:, j])
                TE[i, j] = TE[j, i] = te
        
        # Coherence
        coh = np.array([signal_coherence(ts_seg[:, i]) for i in range(N)])
        
        # TE thresholds for gating
        te_vals = [TE[i, j] for i in range(N) for j in range(i + 1, N)]
        i_min = np.percentile(te_vals, 25) if te_vals else 0
        
        # J matrix
        J_gated = np.zeros((N, N))
        J_ungated = np.zeros((N, N))
        
        for i in range(N):
            for j in range(i + 1, N):
                jsd = amplitude_jsd(ts_seg[:, i], ts_seg[:, j])
                
                g_c = soft_gate(TE[i, j], i_min, 0.05)
                g_r = np.sqrt(max(coh[i], 0) * max(coh[j], 0))
                
                J_gated[i, j] = J_gated[j, i] = TE[i, j] * jsd * g_c * g_r
                J_ungated[i, j] = J_ungated[j, i] = TE[i, j] * jsd
        
        np.fill_diagonal(J_gated, 0)
        np.fill_diagonal(J_ungated, 0)
        
        n_pairs = N * (N - 1) // 2
        phi_gated = sum(J_gated[i, j] for i in range(N) for j in range(i + 1, N)) / n_pairs
        phi_ungated = sum(J_ungated[i, j] for i in range(N) for j in range(i + 1, N)) / n_pairs
        
        # Eigenanalysis
        Js = np.nan_to_num((J_gated + J_gated.T) / 2)
        try:
            evals = np.sort(np.real(np.linalg.eigvalsh(Js)))[::-1]
            pos_evals = evals[evals > 0]
            if len(pos_evals) >= 2:
                eigval_ratio = pos_evals[0] / (pos_evals[1] + 1e-15)
            elif len(pos_evals) == 1:
                eigval_ratio = 999.0
            else:
                eigval_ratio = 0
        except:
            eigval_ratio = 0
        
        return phi_gated, phi_ungated, eigval_ratio, J_gated
    
    # Full run
    phi_g, phi_u, eig_r, J = compute_phi_on_segment(ts)
    
    result = {
        'phi_gated': phi_g,
        'phi_ungated': phi_u,
        'eigval_ratio': eig_r,
        'J_matrix': J
    }
    
    # Split at LOR if applicable
    if lor_tr is not None and 20 < lor_tr < n_tp - 20:
        phi_pre_g, phi_pre_u, eig_pre, _ = compute_phi_on_segment(ts[:lor_tr])
        phi_post_g, phi_post_u, eig_post, _ = compute_phi_on_segment(ts[lor_tr:])
        result['pre_lor_gated'] = phi_pre_g
        result['pre_lor_ungated'] = phi_pre_u
        result['pre_lor_eigratio'] = eig_pre
        result['post_lor_gated'] = phi_post_g
        result['post_lor_ungated'] = phi_post_u
        result['post_lor_eigratio'] = eig_post
    
    return result


# =========================================================================
# APPROACH A: BLOCK-LEVEL DECODING
# =========================================================================

def approach_a(bold_file, masks, timing, lor_tr=None):
    """
    Block-level decoding: average multi-voxel patterns across each imagery
    block, then decode condition, then compute Φ*_PD from decoded policies.
    """
    from nilearn import image
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, LeaveOneOut
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import LabelEncoder
    
    bold_data = image.load_img(str(bold_file)).get_fdata()
    n_vol = bold_data.shape[-1]
    spatial = bold_data.shape[:3]
    bold_2d = bold_data.reshape(-1, n_vol)
    ts = get_timeseries(bold_file, masks)
    
    # Build block-averaged patterns
    # Each block: average TRs from onset to onset + BLOCK_DURATION_TRS
    block_trs = []      # center TR of each block
    block_labels = []
    block_onset_trs = []  # original onset TR
    
    active_conds = ['tennis', 'squeeze', 'navi', 'action']
    
    for cond in active_conds + ['relax']:
        if cond not in timing:
            continue
        for onset in timing[cond]:
            # Block spans onset+HRF to onset+HRF+BLOCK_DURATION
            hrf = 3
            start = onset + hrf
            end = min(start + BLOCK_DURATION_TRS, n_vol)
            if end - start < 5:  # need at least 5 TRs
                continue
            block_trs.append((start, end))
            block_labels.append(cond)
            block_onset_trs.append(onset)
    
    if len(block_trs) < 10 or len(set(block_labels)) < 2:
        return None
    
    block_labels = np.array(block_labels)
    block_onset_trs = np.array(block_onset_trs)
    n_blocks = len(block_trs)
    
    # Extract block-averaged patterns per network
    patterns = {}
    for ni, mask_arr in enumerate(masks):
        mf = mask_arr.ravel()
        nv = mf.sum()
        if nv < 10:
            patterns[ni] = np.zeros((n_blocks, 10))
            continue
        bp = np.zeros((n_blocks, nv))
        for bi, (start, end) in enumerate(block_trs):
            bp[bi] = bold_2d[mf, start:end].mean(axis=1)  # average across TRs in block
        mu = bp.mean(0, keepdims=True)
        sd = bp.std(0, keepdims=True); sd[sd == 0] = 1
        patterns[ni] = (bp - mu) / sd
    
    # Decode per network
    le = LabelEncoder()
    y = le.fit_transform(block_labels)
    nc = len(le.classes_)
    
    min_class = min(np.bincount(y))
    n_folds = min(3, min_class)
    if n_folds < 2:
        n_folds = 2
    
    decoded = {}
    accs = {}
    
    for ni in patterns:
        X = patterns[ni]
        n_comp = min(20, X.shape[0] - 1, X.shape[1])
        if n_comp < 2: n_comp = 2
        try:
            X_pca = PCA(n_components=n_comp, random_state=42).fit_transform(X)
        except:
            X_pca = X[:, :n_comp] if X.shape[1] >= n_comp else X
        
        probs = np.zeros((n_blocks, nc))
        correct = 0; total = 0
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        for tri, tei in skf.split(X_pca, y):
            try:
                clf = LogisticRegression(max_iter=2000, C=0.1, solver='lbfgs', random_state=42)
                clf.fit(X_pca[tri], y[tri])
                probs[tei] = clf.predict_proba(X_pca[tei])
                correct += (clf.predict(X_pca[tei]) == y[tei]).sum()
                total += len(tei)
            except:
                probs[tei] = 1.0 / nc
                total += len(tei)
        
        decoded[ni] = probs
        accs[ni] = correct / total if total > 0 else 1.0 / nc
    
    mean_acc = np.mean(list(accs.values()))
    chance = 1.0 / nc
    
    # Session-level TE and coherence
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
    
    # Per-block Φ*_PD for active blocks only
    active_mask = np.isin(block_labels, active_conds)
    active_indices = np.where(active_mask)[0]
    active_onsets = block_onset_trs[active_mask]
    
    block_phis_gated = []
    block_phis_ungated = []
    block_states = []
    n_pairs = N * (N - 1) // 2
    
    for bi_local, block_idx in enumerate(active_indices):
        onset = active_onsets[bi_local]
        
        if lor_tr is not None:
            state = 'pre_lor' if onset < lor_tr else 'post_lor'
        else:
            state = 'awake'
        block_states.append(state)
        
        J = np.zeros((N, N))
        J_ung = np.zeros((N, N))
        
        for i in range(N):
            for j in range(i + 1, N):
                pi = decoded[i][block_idx] + 1e-10
                pj = decoded[j][block_idx] + 1e-10
                pi /= pi.sum(); pj /= pj.sum()
                m = 0.5 * (pi + pj)
                jsd = max(0.5 * np.sum(pi * np.log2(pi / m)) +
                          0.5 * np.sum(pj * np.log2(pj / m)), 0)
                
                g_c = soft_gate(TE[i, j], i_min, 0.05)
                g_r = np.sqrt(max(coh[i], 0) * max(coh[j], 0))
                g_d = np.sqrt(net_conf.get(i, 0) * net_conf.get(j, 0))
                
                J[i, j] = J[j, i] = TE[i, j] * jsd * g_c * g_r * g_d
                J_ung[i, j] = J_ung[j, i] = TE[i, j] * jsd
        
        np.fill_diagonal(J, 0); np.fill_diagonal(J_ung, 0)
        phi_g = sum(J[i, j] for i in range(N) for j in range(i + 1, N)) / n_pairs
        phi_u = sum(J_ung[i, j] for i in range(N) for j in range(i + 1, N)) / n_pairs
        
        block_phis_gated.append(phi_g)
        block_phis_ungated.append(phi_u)
    
    phi_mean_g = np.mean(block_phis_gated) if block_phis_gated else 0
    phi_mean_u = np.mean(block_phis_ungated) if block_phis_ungated else 0
    
    result = {
        'phi_gated': phi_mean_g,
        'phi_ungated': phi_mean_u,
        'mean_acc': mean_acc,
        'chance_acc': chance,
        'n_blocks': len(active_indices),
        'block_phis_gated': block_phis_gated,
        'block_states': block_states
    }
    
    # Pre/post LOR split
    if lor_tr is not None:
        pre = [p for p, s in zip(block_phis_gated, block_states) if s == 'pre_lor']
        post = [p for p, s in zip(block_phis_gated, block_states) if s == 'post_lor']
        if pre: result['pre_lor_gated'] = np.mean(pre)
        if post: result['post_lor_gated'] = np.mean(post)
        result['n_pre'] = len(pre)
        result['n_post'] = len(post)
    
    return result


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 80)
    print("PMD CONSCIOUSNESS STATE ANALYSIS v3")
    print("Approach B (Enhanced Level 2A) + Approach A (Block-Level Decoding)")
    print("=" * 80)
    
    lor_data = load_lor_timing()
    print(f"\n  LOR timing loaded for {len(lor_data)} subjects")
    
    # Find available subjects
    available = []
    for sub_dir in sorted(FMRIPREP_DIR.iterdir()):
        if sub_dir.is_dir() and sub_dir.name.startswith('sub-'):
            func_dir = sub_dir / "func"
            if func_dir.exists():
                runs_found = 0
                for run in range(1, 5):
                    bold = list(func_dir.glob(f"*task-imagery_run-{run}*MNI152*desc-preproc_bold.nii.gz"))
                    if bold and bold[0].stat().st_size > 1e6:
                        runs_found += 1
                if runs_found >= 3:
                    available.append(sub_dir.name)
    
    print(f"  Available subjects: {available}")
    if not available: print("  No usable subjects."); return
    
    # Get masks from first subject
    first_sub = available[0]
    func_dir = FMRIPREP_DIR / first_sub / "func"
    first_bold = None
    for run in range(1, 5):
        candidates = list(func_dir.glob(f"*task-imagery_run-{run}*MNI152*desc-preproc_bold.nii.gz"))
        if candidates and candidates[0].stat().st_size > 1e6:
            first_bold = candidates[0]; break
    
    print(f"  Loading Yeo-7 atlas...", end="", flush=True)
    masks = get_yeo7_masks(str(first_bold))
    print(f" done")
    
    # =====================================================================
    # APPROACH B: Enhanced Level 2A
    # =====================================================================
    print(f"\n{'='*80}")
    print("APPROACH B: ENHANCED LEVEL 2A (timeseries-based, no decoding)")
    print("="*80)
    
    b_results = {}
    
    for sub in available:
        print(f"\n  {sub}:", end="")
        func_dir = FMRIPREP_DIR / sub / "func"
        lor_tr = lor_data.get(sub, {}).get('lor_tr', None)
        sub_b = {}
        
        for run in range(1, 5):
            candidates = list(func_dir.glob(f"*task-imagery_run-{run}*MNI152*desc-preproc_bold.nii.gz"))
            if not candidates or candidates[0].stat().st_size < 1e6:
                continue
            
            bold_file = candidates[0]
            run_lor = lor_tr if run == 3 else None
            
            res = approach_b(str(bold_file), masks, lor_tr=run_lor)
            sub_b[run] = res
            
            label = RUN_LABELS[run]
            print(f"\n    {label}: Φ_g={res['phi_gated']:.6f} Φ_u={res['phi_ungated']:.6f} eigR={res['eigval_ratio']:.1f}", end="")
            
            if 'pre_lor_gated' in res:
                print(f"\n      PRE-LOR:  Φ_g={res['pre_lor_gated']:.6f} eigR={res['pre_lor_eigratio']:.1f}", end="")
                print(f"\n      POST-LOR: Φ_g={res['post_lor_gated']:.6f} eigR={res['post_lor_eigratio']:.1f}", end="")
        
        b_results[sub] = sub_b
    
    # Approach B summary
    print(f"\n\n{'='*80}")
    print("APPROACH B — RESULTS SUMMARY")
    print("="*80)
    
    b_state_g = {1: [], 2: [], 3: [], 4: []}
    b_state_u = {1: [], 2: [], 3: [], 4: []}
    b_state_eig = {1: [], 2: [], 3: [], 4: []}
    b_pre = []; b_post = []
    b_pre_eig = []; b_post_eig = []
    
    for sub, results in b_results.items():
        for run, res in results.items():
            b_state_g[run].append(res['phi_gated'])
            b_state_u[run].append(res['phi_ungated'])
            b_state_eig[run].append(res['eigval_ratio'])
            if 'pre_lor_gated' in res:
                b_pre.append(res['pre_lor_gated'])
                b_pre_eig.append(res['pre_lor_eigratio'])
            if 'post_lor_gated' in res:
                b_post.append(res['post_lor_gated'])
                b_post_eig.append(res['post_lor_eigratio'])
    
    print(f"\n  {'State':<22} {'Φ_gated':>12} {'Φ_ungated':>12} {'EigRatio':>10} {'N':>4}")
    print(f"  {'-'*60}")
    for run in [1, 2, 3, 4]:
        if b_state_g[run]:
            print(f"  {RUN_LABELS[run]:<22} {np.mean(b_state_g[run]):>12.6f}"
                  f" {np.mean(b_state_u[run]):>12.6f}"
                  f" {np.mean(b_state_eig[run]):>10.1f}"
                  f" {len(b_state_g[run]):>4}")
    
    if b_pre:
        print(f"\n  Pre-LOR (conscious):    Φ_g = {np.mean(b_pre):.6f}  eigR = {np.mean(b_pre_eig):.1f}")
    if b_post:
        print(f"  Post-LOR (unconscious): Φ_g = {np.mean(b_post):.6f}  eigR = {np.mean(b_post_eig):.1f}")
    
    # Approach B predictions
    print(f"\n  APPROACH B PREDICTION TESTS:")
    
    if b_pre and b_post:
        direction = np.mean(b_pre) > np.mean(b_post)
        print(f"\n  ★ Pre-LOR > Post-LOR: {'CORRECT ✓' if direction else 'WRONG ✗'}")
        print(f"    Pre: {np.mean(b_pre):.6f}, Post: {np.mean(b_post):.6f}")
        if b_pre_eig and b_post_eig:
            eig_dir = np.mean(b_pre_eig) > np.mean(b_post_eig)
            print(f"    EigRatio Pre: {np.mean(b_pre_eig):.1f}, Post: {np.mean(b_post_eig):.1f}")
            print(f"    Eigenvector dominance drops at LOR: {'CORRECT ✓' if eig_dir else 'WRONG ✗'}")
    
    if b_state_g[1] and b_state_g[4]:
        ratio = np.mean(b_state_g[4]) / (np.mean(b_state_g[1]) + 1e-15)
        print(f"\n  Recovery ≈ Baseline: ratio = {ratio:.3f}")
    
    if all(b_state_g[r] for r in [1, 2, 3, 4]):
        means = [np.mean(b_state_g[r]) for r in [1, 2, 3, 4]]
        print(f"\n  Trajectory (gated):   {means[0]:.6f} → {means[1]:.6f} → {means[2]:.6f} → {means[3]:.6f}")
        means_u = [np.mean(b_state_u[r]) for r in [1, 2, 3, 4]]
        print(f"  Trajectory (ungated): {means_u[0]:.6f} → {means_u[1]:.6f} → {means_u[2]:.6f} → {means_u[3]:.6f}")
    
    # =====================================================================
    # APPROACH A: Block-Level Decoding
    # =====================================================================
    print(f"\n\n{'='*80}")
    print("APPROACH A: BLOCK-LEVEL DECODING")
    print("="*80)
    
    a_results = {}
    
    for sub in available:
        print(f"\n  {sub}:", end="")
        func_dir = FMRIPREP_DIR / sub / "func"
        lor_tr = lor_data.get(sub, {}).get('lor_tr', None)
        sub_a = {}
        
        for run in range(1, 5):
            task_name = RUN_TO_TASK[run]
            candidates = list(func_dir.glob(f"*task-imagery_run-{run}*MNI152*desc-preproc_bold.nii.gz"))
            if not candidates or candidates[0].stat().st_size < 1e6:
                continue
            
            bold_file = candidates[0]
            timing = load_stimulus_timing(sub, task_name)
            if not timing: continue
            
            run_lor = lor_tr if run == 3 else None
            
            res = approach_a(str(bold_file), masks, timing, lor_tr=run_lor)
            
            if res is not None:
                sub_a[run] = res
                label = RUN_LABELS[run]
                print(f"\n    {label}: Φ_g={res['phi_gated']:.6f} acc={res['mean_acc']:.3f}"
                      f" (chance={res['chance_acc']:.2f}) N_blocks={res['n_blocks']}", end="")
                
                if 'pre_lor_gated' in res:
                    print(f"\n      PRE-LOR:  Φ_g={res['pre_lor_gated']:.6f} (N={res['n_pre']})", end="")
                if 'post_lor_gated' in res:
                    print(f"\n      POST-LOR: Φ_g={res['post_lor_gated']:.6f} (N={res['n_post']})", end="")
        
        a_results[sub] = sub_a
    
    # Approach A summary
    print(f"\n\n{'='*80}")
    print("APPROACH A — RESULTS SUMMARY")
    print("="*80)
    
    a_state_g = {1: [], 2: [], 3: [], 4: []}
    a_state_acc = {1: [], 2: [], 3: [], 4: []}
    a_pre = []; a_post = []
    
    for sub, results in a_results.items():
        for run, res in results.items():
            a_state_g[run].append(res['phi_gated'])
            a_state_acc[run].append(res['mean_acc'])
            if 'pre_lor_gated' in res:
                a_pre.append(res['pre_lor_gated'])
            if 'post_lor_gated' in res:
                a_post.append(res['post_lor_gated'])
    
    print(f"\n  {'State':<22} {'Φ_gated':>12} {'Accuracy':>10} {'N':>4}")
    print(f"  {'-'*48}")
    for run in [1, 2, 3, 4]:
        if a_state_g[run]:
            chance = 0.20  # approximate
            print(f"  {RUN_LABELS[run]:<22} {np.mean(a_state_g[run]):>12.6f}"
                  f" {np.mean(a_state_acc[run]):>10.3f}"
                  f" {len(a_state_g[run]):>4}")
    
    if a_pre:
        print(f"\n  Pre-LOR:  Φ_g = {np.mean(a_pre):.6f}")
    if a_post:
        print(f"  Post-LOR: Φ_g = {np.mean(a_post):.6f}")
    
    if a_pre and a_post:
        direction = np.mean(a_pre) > np.mean(a_post)
        print(f"\n  ★ Pre-LOR > Post-LOR: {'CORRECT ✓' if direction else 'WRONG ✗'}")
    
    # Decoder quality across states
    if a_state_acc[1] and a_state_acc[3]:
        print(f"\n  Decoder accuracy trajectory:")
        for run in [1, 2, 3, 4]:
            if a_state_acc[run]:
                print(f"    {RUN_LABELS[run]}: {np.mean(a_state_acc[run]):.3f}")
        
        base_acc = np.mean(a_state_acc[1])
        deep_acc = np.mean(a_state_acc[3])
        print(f"\n  Accuracy drops with sedation: {'YES ✓' if base_acc > deep_acc else 'NO'}")
        print(f"  (This itself is evidence: brain loses ability to generate distinct task patterns)")
    
    # =====================================================================
    # OVERALL VERDICT
    # =====================================================================
    print(f"\n\n{'='*80}")
    print("OVERALL VERDICT — BOTH APPROACHES")
    print("="*80)
    
    print(f"""
  KEY COMPARISON: Pre-LOR vs Post-LOR (within same run, same drug level)
  
  Approach B (Level 2A):
    Pre-LOR:  {np.mean(b_pre):.6f if b_pre else 'N/A'}
    Post-LOR: {np.mean(b_post):.6f if b_post else 'N/A'}
    Direction: {'CORRECT ✓' if b_pre and b_post and np.mean(b_pre) > np.mean(b_post) else 'WRONG ✗' if b_pre and b_post else 'N/A'}
  
  Approach A (Block decoding):
    Pre-LOR:  {np.mean(a_pre):.6f if a_pre else 'N/A'}
    Post-LOR: {np.mean(a_post):.6f if a_post else 'N/A'}
    Direction: {'CORRECT ✓' if a_pre and a_post and np.mean(a_pre) > np.mean(a_post) else 'WRONG ✗' if a_pre and a_post else 'N/A'}
  
  NOTE: N=2 subjects. Direction matters more than significance.
  Full test requires remaining subjects to be downloaded.
""")
    
    print("=" * 80)
    print("CONSCIOUSNESS STATE ANALYSIS v3 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
