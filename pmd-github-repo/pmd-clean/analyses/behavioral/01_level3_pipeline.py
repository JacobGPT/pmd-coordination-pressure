#!/usr/bin/env python3
"""
===========================================================================
PRESSURE MAKES DIAMONDS — LEVEL 3 PROOF OF CONCEPT
Decoded Policy Divergence × Effective Connectivity
===========================================================================

DATASET: ds000030 (UCLA Consortium)
TASK:    Stop Signal (GO vs STOP, 256 trials)
         + SCAP Working Memory (Load 1/3/5, 48 blocks)

THE KEY UPGRADE:
  Level 2A: C_ij = JSD of amplitude distributions (signal shape)
  Level 3:  C_ij = JSD of DECODED POLICY DISTRIBUTIONS

  For each network, train a decoder: can this network's multi-voxel
  pattern predict the task condition? The decoded probability vector
  IS the network's "policy" — what it thinks is happening / what it
  wants to do. Two networks with different decoded probabilities are
  in genuine policy conflict.

USAGE:
  python pmd_level3_poc.py --data-dir ./ds000030 --subject sub-10159
===========================================================================
"""

import os, sys, json, argparse, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')

YEO7_LABELS = ["Visual","Somatomotor","DorsalAttention","VentralAttention","Limbic","Frontoparietal","Default"]

# =========================================================================
# ATLAS + MASKING
# =========================================================================

def get_network_masks(bold_file):
    """Get Yeo-7 network masks resampled to BOLD space. Returns list of boolean masks."""
    from nilearn import image
    import nibabel as nib
    import glob
    
    yeo_dir = os.path.join(os.path.expanduser('~'), 'nilearn_data', 'yeo_2011')
    cands = glob.glob(os.path.join(yeo_dir, '**', 'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz'), recursive=True)
    if not cands:
        from nilearn import datasets
        datasets.fetch_atlas_yeo_2011()
        cands = glob.glob(os.path.join(yeo_dir, '**', 'Yeo2011_7Networks*1mm.nii.gz'), recursive=True)
    
    atlas_img = nib.load(cands[0])
    bold_img = image.load_img(bold_file)
    
    # Resample atlas to BOLD space
    atlas_resampled = image.resample_to_img(atlas_img, bold_img, interpolation='nearest')
    atlas_data = atlas_resampled.get_fdata()
    
    # Squeeze any extra dimensions (e.g. (64,64,34,1) -> (64,64,34))
    while atlas_data.ndim > 3:
        atlas_data = atlas_data[..., 0]
    
    masks = []
    for net_idx in range(1, 8):
        mask = (np.round(atlas_data) == net_idx)
        masks.append(mask)
    
    return masks


def extract_multivoxel_patterns(bold_file, masks, events_df, condition_col='trial_type',
                                  conditions=None, tr=2.0, hrf_delay=5.0):
    """
    Extract multi-voxel patterns per trial/block for each network.
    
    For each trial: take BOLD volume at onset + HRF delay, extract
    voxel values within each network mask.
    
    Returns:
        patterns: dict {network_idx: array[n_trials, n_voxels]}
        labels: array[n_trials] of condition labels
    """
    from nilearn import image
    
    bold_img = image.load_img(bold_file)
    bold_data = bold_img.get_fdata()
    n_volumes = bold_data.shape[-1]
    
    # Filter to requested conditions
    df = events_df.copy()
    if conditions is not None:
        df = df[df[condition_col].isin(conditions)].reset_index(drop=True)
    
    # Remove NaN conditions
    df = df.dropna(subset=[condition_col]).reset_index(drop=True)
    
    labels = df[condition_col].values
    onsets = df['onset'].values
    
    # For each trial, pick the volume closest to onset + HRF delay
    target_times = onsets + hrf_delay
    target_volumes = np.clip((target_times / tr).astype(int), 0, n_volumes - 1)
    
    # Extract patterns per network
    patterns = {}
    # Reshape BOLD to 2D: (n_spatial_voxels, n_volumes)
    spatial_shape = bold_data.shape[:3]
    bold_2d = bold_data.reshape(-1, n_volumes)
    
    for net_idx, mask in enumerate(masks):
        # Ensure mask matches spatial shape
        if mask.shape != spatial_shape:
            print(f"    WARNING: mask shape {mask.shape} != bold shape {spatial_shape}, skipping network {net_idx}")
            patterns[net_idx] = np.zeros((len(labels), 10))
            continue
        
        mask_flat = mask.ravel()
        n_voxels = mask_flat.sum()
        if n_voxels < 10:
            patterns[net_idx] = np.zeros((len(labels), 10))
            continue
        
        # Extract voxel values for each trial
        trial_patterns = np.zeros((len(labels), n_voxels))
        for t_idx, vol_idx in enumerate(target_volumes):
            trial_patterns[t_idx, :] = bold_2d[mask_flat, vol_idx]
        
        # Z-score each voxel across trials
        means = trial_patterns.mean(axis=0, keepdims=True)
        stds = trial_patterns.std(axis=0, keepdims=True)
        stds[stds == 0] = 1
        trial_patterns = (trial_patterns - means) / stds
        
        patterns[net_idx] = trial_patterns
    
    return patterns, labels


# =========================================================================
# DECODERS
# =========================================================================

def train_network_decoders(patterns, labels, n_folds=5):
    """
    Train a decoder for each network. Uses logistic regression with
    cross-validation. Returns decoded probability vectors for each
    trial (from held-out folds).
    
    Returns:
        decoded_probs: dict {net_idx: array[n_trials, n_classes]}
        accuracies: dict {net_idx: float}
        classes: array of class labels
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    
    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = le.classes_
    n_classes = len(classes)
    n_trials = len(y)
    
    decoded_probs = {}
    accuracies = {}
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for net_idx in patterns:
        X = patterns[net_idx]
        
        # Reduce dimensionality if too many voxels
        if X.shape[1] > 500:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(100, X.shape[0] - 1), random_state=42)
            X = pca.fit_transform(X)
        
        probs = np.zeros((n_trials, n_classes))
        correct = 0
        total = 0
        
        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                                      random_state=42)
            try:
                clf.fit(X[train_idx], y[train_idx])
                probs[test_idx] = clf.predict_proba(X[test_idx])
                preds = clf.predict(X[test_idx])
                correct += (preds == y[test_idx]).sum()
                total += len(test_idx)
            except:
                # If decoding fails, assign uniform probabilities
                probs[test_idx] = 1.0 / n_classes
                total += len(test_idx)
        
        decoded_probs[net_idx] = probs
        accuracies[net_idx] = correct / total if total > 0 else 0.5
    
    return decoded_probs, accuracies, classes


# =========================================================================
# LEVEL 3 CONFLICT: JSD OF DECODED POLICIES
# =========================================================================

def compute_policy_divergence(decoded_probs, min_confidence=0.55):
    """
    Compute pairwise JSD between decoded policy distributions.
    
    For each pair of networks (i, j), average the trialwise JSD
    between their decoded probability vectors.
    
    Also computes a decode-confidence gate: if a network's decoder
    is near chance, its "policy" is noise and shouldn't count.
    
    Returns:
        C_star: 7x7 matrix of policy divergences
        G_decode: 7x7 matrix of decode confidence gates
    """
    N = len(decoded_probs)
    C_star = np.zeros((N, N))
    G_decode = np.zeros((N, N))
    
    # Per-network confidence: how far from uniform are the decoded probs?
    network_confidence = {}
    for net_idx in decoded_probs:
        probs = decoded_probs[net_idx]
        n_classes = probs.shape[1]
        uniform = np.ones(n_classes) / n_classes
        # Average entropy ratio: lower entropy = more confident
        entropies = -np.sum(probs * np.log2(probs + 1e-10), axis=1)
        max_entropy = np.log2(n_classes)
        confidence = 1.0 - np.mean(entropies) / max_entropy if max_entropy > 0 else 0
        network_confidence[net_idx] = max(confidence, 0)
    
    for i in range(N):
        for j in range(i + 1, N):
            p_i = decoded_probs[i]  # [n_trials, n_classes]
            p_j = decoded_probs[j]
            
            # Trialwise JSD
            jsds = []
            for t in range(p_i.shape[0]):
                pi = p_i[t] + 1e-10
                pj = p_j[t] + 1e-10
                pi = pi / pi.sum()
                pj = pj / pj.sum()
                m = 0.5 * (pi + pj)
                jsd = 0.5 * np.sum(pi * np.log2(pi / m)) + 0.5 * np.sum(pj * np.log2(pj / m))
                jsds.append(max(jsd, 0))
            
            C_star[i, j] = C_star[j, i] = np.mean(jsds)
            
            # Decode confidence gate: geometric mean of both networks' confidence
            g = np.sqrt(network_confidence[i] * network_confidence[j])
            G_decode[i, j] = G_decode[j, i] = g
    
    return C_star, G_decode, network_confidence


# =========================================================================
# EFFECTIVE CONNECTIVITY (TE from network timeseries)
# =========================================================================

def compute_network_timeseries(bold_file, masks):
    """Extract mean timeseries per network for TE computation."""
    from nilearn import image
    from scipy.signal import detrend
    
    bold_img = image.load_img(bold_file)
    bold_data = bold_img.get_fdata()
    n_tp = bold_data.shape[-1]
    
    timeseries = np.zeros((n_tp, 7))
    for net_idx, mask in enumerate(masks):
        if mask.sum() < 10:
            continue
        # Reshape BOLD to (n_voxels_total, n_timepoints), apply mask
        shape = bold_data.shape
        data_2d = bold_data.reshape(-1, n_tp)  # (x*y*z, t)
        mask_flat = mask.ravel()
        ts = data_2d[mask_flat, :].mean(axis=0)
        ts = detrend(ts)
        std = ts.std()
        if std > 0:
            ts = ts / std
        timeseries[:, net_idx] = ts
    
    return timeseries


def sym_te(s, t, lag=1, bins=10):
    def te(src, tgt):
        n = len(src)
        if n <= lag + 1: return 0.0
        yn, yp, xp = tgt[lag:], tgt[:-lag], src[:-lag]
        def bd(x):
            mn, mx = x.min(), x.max()
            if mx == mn: return np.zeros(len(x), dtype=int)
            return np.clip(((x-mn)/(mx-mn)*(bins-1)).astype(int), 0, bins-1)
        yb, ypb, xpb = bd(yn), bd(yp), bd(xp)
        def H(c):
            p = c[c>0]/c.sum(); return -np.sum(p*np.log2(p))
        jyy = np.zeros((bins,bins))
        for i in range(len(yb)): jyy[yb[i],ypb[i]] += 1
        jyyx = np.zeros((bins,bins,bins))
        for i in range(len(yb)): jyyx[yb[i],ypb[i],xpb[i]] += 1
        jyx = np.zeros((bins,bins))
        for i in range(len(ypb)): jyx[ypb[i],xpb[i]] += 1
        return max((H(jyy.ravel()) - H(np.bincount(ypb,minlength=bins).astype(float))) - (H(jyyx.ravel()) - H(jyx.ravel())), 0.0)
    return (te(s,t) + te(t,s)) / 2.0


def signal_coherence(ts, lag=1):
    if len(ts) < lag+2: return 0.0
    ac = np.corrcoef(ts[:-lag], ts[lag:])[0,1]
    return max(0.0, ac) if not np.isnan(ac) else 0.0


def soft_gate(x, x0, tau):
    z = np.clip((x-x0)/tau if tau>0 else 0, -20, 20)
    return 1.0/(1.0+np.exp(-z))


# =========================================================================
# LEVEL 3 J* COMPUTATION
# =========================================================================

def compute_level3_metrics(C_star, G_decode, timeseries, tau=0.05):
    """
    Compute J* = I* × C* × G_organization
    
    C* comes from decoded policy divergence (already computed)
    I* comes from TE on network timeseries
    G_org = stability × coherence × coupling × decode_confidence
    """
    N = 7
    
    # Compute TE matrix
    TE = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            te = sym_te(timeseries[:, i], timeseries[:, j])
            TE[i, j] = TE[j, i] = te
    
    # Coherence per network
    coh = np.array([signal_coherence(timeseries[:, i]) for i in range(N)])
    
    # I_min for coupling gate
    te_vals = [TE[i,j] for i in range(N) for j in range(i+1,N)]
    i_min = np.percentile(te_vals, 25)
    
    # Build J matrices
    J_level3 = np.zeros((N, N))
    J_level2a = np.zeros((N, N))  # For comparison
    pair_details = {}
    
    for i in range(N):
        for j in range(i + 1, N):
            te = TE[i, j]
            c_star = C_star[i, j]
            g_dec = G_decode[i, j]
            
            # Level 2A gates
            g_coh = np.sqrt(max(coh[i], 0) * max(coh[j], 0))
            g_coup = soft_gate(te, i_min, tau)
            
            # Level 3: J* = TE × policy_divergence × gates
            j3 = te * c_star * g_coup * g_coh * g_dec
            J_level3[i, j] = J_level3[j, i] = j3
            
            # Level 2A comparison (using amplitude JSD placeholder)
            # Will be computed separately if needed
            
            pair_name = f"{YEO7_LABELS[i]}-{YEO7_LABELS[j]}"
            pair_details[pair_name] = {
                'te': float(te), 'c_star': float(c_star),
                'g_decode': float(g_dec), 'g_coherence': float(g_coh),
                'g_coupling': float(g_coup), 'j_star': float(j3),
            }
    
    np.fill_diagonal(J_level3, 0)
    
    # Eigenvalue analysis
    Js = np.nan_to_num((J_level3 + J_level3.T) / 2.0)
    try: ev = np.linalg.eigvalsh(Js)
    except: ev = np.real(np.linalg.eigvals(Js))
    ev = np.sort(np.real(ev))[::-1]
    
    phi = sum(J_level3[i,j] for i in range(N) for j in range(i+1,N))
    n_pairs = N*(N-1)//2
    
    return {
        'phi_pd_star': float(phi / n_pairs),
        'lambda_max_star': float(ev[0]),
        'lambda_spectrum': [float(x) for x in ev],
        'pair_details': pair_details,
    }


# =========================================================================
# MAIN PIPELINE
# =========================================================================

def run_level3_poc(data_dir, subject, output_dir):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 75)
    print("PRESSURE MAKES DIAMONDS — LEVEL 3 PROOF OF CONCEPT")
    print("Decoded Policy Divergence × Effective Connectivity")
    print("=" * 75)
    print(f"\n  Subject: {subject}")
    print(f"  Dataset: {data_dir}")
    
    import pandas as pd
    
    sub_dir = data_dir / subject / "func"
    
    # =================================================================
    # TASK 1: STOP SIGNAL (GO vs STOP)
    # =================================================================
    ss_bold = sub_dir / f"{subject}_task-stopsignal_bold.nii.gz"
    ss_events = sub_dir / f"{subject}_task-stopsignal_events.tsv"
    
    if ss_bold.exists() and ss_events.exists():
        print(f"\n{'='*75}")
        print("TASK: STOP SIGNAL (GO vs STOP)")
        print(f"{'='*75}")
        print("  Prediction: STOP trials should show higher Φ*_PD than GO trials")
        print("  (STOP requires inhibitory control = inter-domain conflict)")
        
        events = pd.read_csv(ss_events, sep='\t')
        
        print(f"\n  Step 1: Loading BOLD and extracting network masks...")
        masks = get_network_masks(str(ss_bold))
        print(f"    Network voxel counts: {[m.sum() for m in masks]}")
        
        print(f"\n  Step 2: Extracting multi-voxel patterns per trial...")
        # Get TR from JSON
        ss_json = sub_dir / f"{subject}_task-stopsignal_bold.json"
        tr = 2.0
        if ss_json.exists():
            with open(ss_json) as f:
                meta = json.load(f)
                tr = meta.get('RepetitionTime', 2.0)
            print(f"    TR = {tr}s")
        
        patterns, labels = extract_multivoxel_patterns(
            str(ss_bold), masks, events,
            condition_col='trial_type', conditions=['GO', 'STOP'],
            tr=tr, hrf_delay=5.0
        )
        print(f"    Trials: {len(labels)} ({(labels=='GO').sum()} GO, {(labels=='STOP').sum()} STOP)")
        
        print(f"\n  Step 3: Training per-network decoders (GO vs STOP)...")
        decoded_probs, accuracies, classes = train_network_decoders(patterns, labels)
        
        print(f"\n    NETWORK DECODING ACCURACIES (chance = 50%):")
        for net_idx in range(7):
            acc = accuracies[net_idx]
            above = "★ above chance" if acc > 0.55 else ""
            print(f"      {YEO7_LABELS[net_idx]:<20} {acc:.1%} {above}")
        
        print(f"\n  Step 4: Computing policy divergence (JSD of decoded probabilities)...")
        C_star, G_decode, net_confidence = compute_policy_divergence(decoded_probs)
        
        print(f"\n    TOP POLICY DIVERGENCES (C*_ij):")
        pairs = []
        for i in range(7):
            for j in range(i+1, 7):
                pairs.append((C_star[i,j], f"{YEO7_LABELS[i]}-{YEO7_LABELS[j]}"))
        pairs.sort(reverse=True)
        for val, name in pairs[:5]:
            print(f"      {name:<35} C*={val:.6f}")
        
        print(f"\n  Step 5: Computing effective connectivity (TE)...")
        timeseries = compute_network_timeseries(str(ss_bold), masks)
        
        print(f"\n  Step 6: Computing J*, Φ*_PD, λ*_max...")
        metrics = compute_level3_metrics(C_star, G_decode, timeseries)
        
        print(f"\n    Φ*_PD  = {metrics['phi_pd_star']:.8f}")
        print(f"    λ*_max = {metrics['lambda_max_star']:.8f}")
        print(f"    λ spectrum: {[f'{x:.4f}' for x in metrics['lambda_spectrum']]}")
        
        # Now compute per-condition: split trials into GO and STOP
        print(f"\n  Step 7: Per-condition analysis...")
        for cond in ['GO', 'STOP']:
            mask_cond = (labels == cond)
            
            # Get decoded probs for this condition only
            cond_probs = {k: v[mask_cond] for k, v in decoded_probs.items()}
            C_cond, G_dec_cond, _ = compute_policy_divergence(cond_probs)
            cond_metrics = compute_level3_metrics(C_cond, G_dec_cond, timeseries)
            
            print(f"\n    {cond} trials (n={mask_cond.sum()}):")
            print(f"      Φ*_PD  = {cond_metrics['phi_pd_star']:.8f}")
            print(f"      λ*_max = {cond_metrics['lambda_max_star']:.8f}")
        
        # Top pair contributions
        print(f"\n    TOP J* PAIR CONTRIBUTIONS:")
        for name, details in sorted(metrics['pair_details'].items(), key=lambda x: -x[1]['j_star'])[:5]:
            d = details
            print(f"      {name:<35} J*={d['j_star']:.6f} (TE={d['te']:.3f} C*={d['c_star']:.4f} g_dec={d['g_decode']:.3f})")
    
    # =================================================================
    # TASK 2: SCAP WORKING MEMORY (Load 1/3/5)
    # =================================================================
    scap_bold = sub_dir / f"{subject}_task-scap_bold.nii.gz"
    scap_events = sub_dir / f"{subject}_task-scap_events.tsv"
    
    if scap_bold.exists() and scap_events.exists():
        print(f"\n\n{'='*75}")
        print("TASK: SCAP WORKING MEMORY (Load 1 vs 3 vs 5)")
        print(f"{'='*75}")
        print("  Prediction: Higher load should show higher Φ*_PD")
        print("  (more items = more inter-domain coordination pressure)")
        
        events = pd.read_csv(scap_events, sep='\t')
        
        # TR
        scap_json = sub_dir / f"{subject}_task-scap_bold.json"
        tr = 2.0
        if scap_json.exists():
            with open(scap_json) as f:
                tr = json.load(f).get('RepetitionTime', 2.0)
        
        print(f"\n  Extracting patterns per load condition...")
        masks_scap = get_network_masks(str(scap_bold))
        
        # Use Load as the condition
        events['load_str'] = events['Load'].astype(str)
        patterns, labels = extract_multivoxel_patterns(
            str(scap_bold), masks_scap, events,
            condition_col='load_str', conditions=['1', '3', '5'],
            tr=tr, hrf_delay=5.0
        )
        
        print(f"  Trials per load: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        print(f"\n  Training decoders (Load 1 vs 3 vs 5)...")
        decoded_probs, accuracies, classes = train_network_decoders(patterns, labels)
        
        print(f"\n    NETWORK DECODING ACCURACIES (chance = 33%):")
        for net_idx in range(7):
            acc = accuracies[net_idx]
            above = "★" if acc > 0.40 else ""
            print(f"      {YEO7_LABELS[net_idx]:<20} {acc:.1%} {above}")
        
        # Per-load analysis
        timeseries = compute_network_timeseries(str(scap_bold), masks_scap)
        
        print(f"\n  Per-load Φ*_PD:")
        for load in ['1', '3', '5']:
            mask_load = (labels == load)
            if mask_load.sum() < 5:
                continue
            cond_probs = {k: v[mask_load] for k, v in decoded_probs.items()}
            C_cond, G_dec_cond, _ = compute_policy_divergence(cond_probs)
            m = compute_level3_metrics(C_cond, G_dec_cond, timeseries)
            print(f"    Load {load}: Φ*_PD = {m['phi_pd_star']:.8f}, λ*_max = {m['lambda_max_star']:.8f}")
    
    # =================================================================
    # CROSS-TASK COMPARISON
    # =================================================================
    print(f"\n\n{'='*75}")
    print("CROSS-TASK ANALYSIS")
    print(f"{'='*75}")
    print("  Computing Level 3 for each task as a whole...")
    print("  Prediction: Tasks with more multi-domain conflict should")
    print("  show higher Φ*_PD than simpler tasks")
    
    tasks = {
        'stopsignal': {'conditions': ['GO', 'STOP'], 'col': 'trial_type'},
        'taskswitch': {'conditions': None, 'col': None},  # Use all trials
        'bart': {'conditions': ['BALOON', 'CONTROL'], 'col': 'trial_type'},
        'scap': {'conditions': ['1', '3', '5'], 'col': 'load_str'},
        'rest': {'conditions': None, 'col': None},
    }
    
    task_metrics = {}
    
    for task_name, task_info in tasks.items():
        bold_path = sub_dir / f"{subject}_task-{task_name}_bold.nii.gz"
        if not bold_path.exists():
            continue
        
        print(f"\n  {task_name}...", end="", flush=True)
        
        try:
            masks_t = get_network_masks(str(bold_path))
            ts = compute_network_timeseries(str(bold_path), masks_t)
            
            events_path = sub_dir / f"{subject}_task-{task_name}_events.tsv"
            
            if events_path.exists() and task_info['col'] is not None:
                ev = pd.read_csv(events_path, sep='\t')
                if task_name == 'scap':
                    ev['load_str'] = ev['Load'].astype(str)
                
                json_path = sub_dir / f"{subject}_task-{task_name}_bold.json"
                tr = 2.0
                if json_path.exists():
                    with open(json_path) as f:
                        tr = json.load(f).get('RepetitionTime', 2.0)
                
                pats, labs = extract_multivoxel_patterns(
                    str(bold_path), masks_t, ev,
                    condition_col=task_info['col'],
                    conditions=task_info['conditions'],
                    tr=tr
                )
                
                if len(labs) > 10:
                    dec_probs, accs, cls = train_network_decoders(pats, labs)
                    C_s, G_d, _ = compute_policy_divergence(dec_probs)
                    m = compute_level3_metrics(C_s, G_d, ts)
                    task_metrics[task_name] = m
                    mean_acc = np.mean(list(accs.values()))
                    print(f" Φ*_PD={m['phi_pd_star']:.6f}, λ*_max={m['lambda_max_star']:.6f}, mean_acc={mean_acc:.1%}")
                else:
                    print(f" too few trials")
            else:
                # Resting state — use Level 2A only
                print(f" (rest — no decoded policies, Level 2A only)")
                
        except Exception as e:
            print(f" ERROR: {e}")
    
    # Summary
    if task_metrics:
        print(f"\n  {'Task':<20} {'Φ*_PD':>12} {'λ*_max':>12}")
        print(f"  {'-'*46}")
        for tn in sorted(task_metrics, key=lambda x: -task_metrics[x]['phi_pd_star']):
            m = task_metrics[tn]
            print(f"  {tn:<20} {m['phi_pd_star']:>12.6f} {m['lambda_max_star']:>12.6f}")
    
    # Save
    def convert(obj):
        if isinstance(obj, (np.floating, float)): return float(obj)
        if isinstance(obj, (np.integer, int)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(i) for i in obj]
        return obj
    
    with open(output_dir / "level3_poc_results.json", 'w') as f:
        json.dump(convert({'subject': subject, 'task_metrics': task_metrics}), f, indent=2)
    print(f"\n  Saved: {output_dir / 'level3_poc_results.json'}")
    
    print(f"\n{'='*75}")
    print("LEVEL 3 PROOF OF CONCEPT COMPLETE")
    print(f"{'='*75}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./ds000030")
    parser.add_argument("--subject", default="sub-10159")
    parser.add_argument("--output-dir", default="./pmd_results")
    args = parser.parse_args()
    run_level3_poc(args.data_dir, args.subject, args.output_dir)
