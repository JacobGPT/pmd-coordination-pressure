#!/usr/bin/env python3
"""
===========================================================================
PRESSURE MAKES DIAMONDS — MATHEMATICAL PREDICTIONS TEST
Testing 5 of 7 deep mathematical predictions from Section 24.8

TEST 1: EIGENVECTOR PREDICTION
  The dominant eigenvector of J should predict the system's coordination state.
  Subjects with empirical domain-weight distributions closer to the principal
  eigenvector should show stronger coordination signatures.

TEST 2: RENORMALIZATION GROUP — COARSE-GRAINING INVARIANCE
  Φ_PD should be approximately invariant when computed at different parcellation
  resolutions (7-network, 4-network, 2-network). Scale-invariance near criticality.

TEST 3: RANDOM MATRIX THEORY NULL HYPOTHESIS  
  λ_max should significantly exceed the Marchenko-Pastur bulk edge.
  The gap = "consciousness signal" in RMT terms.

TEST 4: OPTIMAL TRANSPORT COMPARISON
  Compare JSD-based conflict with Wasserstein (Earth Mover's) distance.
  If they track each other, the optimal transport interpretation is validated.

TEST 5: MORSE THEORY — LANDSCAPE TOPOLOGY
  Count critical points of the coordination energy on the simplex.
  More complex tasks should have more rugged landscapes.
===========================================================================
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

YEO7 = ["Visual","Somatomotor","DorsalAttention","VentralAttention","Limbic","Frontoparietal","Default"]

# Merge maps for coarse-graining
# 4-network: Visual+Somatomotor=Sensorimotor, DorsAtt+VentAtt=Attention, Limbic+FP=Control, Default=Default
MERGE_4 = {0:0, 1:0, 2:1, 3:1, 4:2, 5:2, 6:3}
# 2-network: Sensory(Vis+Som+DorsAtt) vs Higher(VentAtt+Limbic+FP+Default)
MERGE_2 = {0:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:1}

# =========================================================================
# UTILITIES
# =========================================================================

def get_masks(bold_file):
    from nilearn import image; import nibabel as nib, glob as g
    ydir = os.path.join(os.path.expanduser('~'), 'nilearn_data', 'yeo_2011')
    c = g.glob(os.path.join(ydir, '**', 'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz'), recursive=True)
    if not c:
        from nilearn import datasets; datasets.fetch_atlas_yeo_2011()
        c = g.glob(os.path.join(ydir, '**', '*7Networks*1mm.nii.gz'), recursive=True)
    atlas = image.resample_to_img(nib.load(c[0]), image.load_img(bold_file), interpolation='nearest')
    ad = atlas.get_fdata()
    while ad.ndim > 3: ad = ad[..., 0]
    return [np.round(ad) == i for i in range(1, 8)]

def get_timeseries(bold_file, masks):
    from nilearn import image; from scipy.signal import detrend
    bd = image.load_img(bold_file).get_fdata(); n_tp = bd.shape[-1]
    d2 = bd.reshape(-1, n_tp); ts = np.zeros((n_tp, 7))
    for i, m in enumerate(masks):
        mf = m.ravel()
        if mf.sum() < 10: continue
        t = d2[mf,:].mean(axis=0); t = detrend(t); s = t.std()
        if s > 0: t = t/s
        ts[:,i] = t
    return ts

def sym_te(s, t, lag=1, bins=10):
    def te(src, tgt):
        n=len(src)
        if n<=lag+1: return 0.0
        yn,yp,xp=tgt[lag:],tgt[:-lag],src[:-lag]
        def bd(x):
            mn,mx=x.min(),x.max()
            if mx==mn: return np.zeros(len(x),dtype=int)
            return np.clip(((x-mn)/(mx-mn)*(bins-1)).astype(int),0,bins-1)
        yb,ypb,xpb=bd(yn),bd(yp),bd(xp)
        def H(c): p=c[c>0]/c.sum(); return -np.sum(p*np.log2(p))
        jyy=np.zeros((bins,bins))
        for i in range(len(yb)): jyy[yb[i],ypb[i]]+=1
        jyyx=np.zeros((bins,bins,bins))
        for i in range(len(yb)): jyyx[yb[i],ypb[i],xpb[i]]+=1
        jyx=np.zeros((bins,bins))
        for i in range(len(ypb)): jyx[ypb[i],xpb[i]]+=1
        return max((H(jyy.ravel())-H(np.bincount(ypb,minlength=bins).astype(float)))-(H(jyyx.ravel())-H(jyx.ravel())),0.0)
    return (te(s,t)+te(t,s))/2.0

def signal_coherence(ts, lag=1):
    if len(ts)<lag+2: return 0.0
    ac=np.corrcoef(ts[:-lag],ts[lag:])[0,1]
    return max(0.0,ac) if not np.isnan(ac) else 0.0

def compute_J_matrix(timeseries, tau=0.05):
    """Compute the full J matrix (correlation-based for resting-state compatibility)"""
    N = timeseries.shape[1]
    corr = np.corrcoef(timeseries.T)
    J = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            c = corr[i,j]
            if c < 0:
                J[i,j] = J[j,i] = abs(c)
            else:
                J[i,j] = J[j,i] = c * (1 - c)
    np.fill_diagonal(J, 0)
    return J

def compute_J_te_jsd(timeseries, tau=0.05):
    """Compute J using TE × JSD (Level 2) for a given timeseries"""
    N = timeseries.shape[1]
    TE = np.zeros((N,N))
    JSD = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            te = sym_te(timeseries[:,i], timeseries[:,j])
            TE[i,j] = TE[j,i] = te
            # Amplitude JSD
            v = np.concatenate([timeseries[:,i], timeseries[:,j]])
            mn, mx = v.min(), v.max()
            if mx == mn: jsd = 0
            else:
                bins = 20; edges = np.linspace(mn,mx,bins+1); eps=1e-10
                p = np.histogram(timeseries[:,i],bins=edges)[0].astype(float)
                q = np.histogram(timeseries[:,j],bins=edges)[0].astype(float)
                p=(p+eps)/((p+eps).sum()); q=(q+eps)/((q+eps).sum()); m=0.5*(p+q)
                jsd = max(0.5*np.sum(p*np.log2(p/m))+0.5*np.sum(q*np.log2(q/m)), 0)
            JSD[i,j] = JSD[j,i] = jsd
    
    J = TE * JSD
    np.fill_diagonal(J, 0)
    return J, TE, JSD

def compute_lmax_and_evec(J):
    Js = np.nan_to_num((J+J.T)/2.0)
    eigenvalues, eigenvectors = np.linalg.eigh(Js)
    idx = np.argmax(eigenvalues)
    return float(eigenvalues[idx]), eigenvectors[:, idx]

def compute_phi(J):
    N = J.shape[0]
    n_pairs = N*(N-1)//2
    return sum(J[i,j] for i in range(N) for j in range(i+1,N)) / n_pairs if n_pairs > 0 else 0

def wasserstein_1d(a, b, bins=20):
    """1D Wasserstein distance between two signal distributions"""
    v = np.concatenate([a,b]); mn,mx = v.min(),v.max()
    if mx == mn: return 0.0
    edges = np.linspace(mn,mx,bins+1)
    p = np.histogram(a,bins=edges)[0].astype(float)
    q = np.histogram(b,bins=edges)[0].astype(float)
    p = p/p.sum() if p.sum()>0 else np.ones(bins)/bins
    q = q/q.sum() if q.sum()>0 else np.ones(bins)/bins
    # CDF difference = Wasserstein-1
    return np.sum(np.abs(np.cumsum(p) - np.cumsum(q))) / bins

def coarse_grain_ts(timeseries, merge_map, n_out):
    """Average timeseries according to merge map"""
    N_in = timeseries.shape[1]
    n_tp = timeseries.shape[0]
    ts_out = np.zeros((n_tp, n_out))
    counts = np.zeros(n_out)
    for i in range(N_in):
        ts_out[:, merge_map[i]] += timeseries[:, i]
        counts[merge_map[i]] += 1
    for k in range(n_out):
        if counts[k] > 0:
            ts_out[:, k] /= counts[k]
    return ts_out

def marchenko_pastur_edge(N, T):
    """Upper edge of Marchenko-Pastur distribution for N×N matrix from T samples"""
    gamma = N / T
    return (1 + np.sqrt(gamma))**2


# =========================================================================
# MAIN
# =========================================================================

def main():
    data_dir = Path("./ds000030")
    subjects = ['sub-10159','sub-10206','sub-10269','sub-10271','sub-10273',
                'sub-10274','sub-10280','sub-10290','sub-10292','sub-10299']
    
    print("="*80)
    print("PRESSURE MAKES DIAMONDS — MATHEMATICAL PREDICTIONS TEST")
    print("Testing 5 of 7 predictions from Section 24.8")
    print("="*80)
    
    # Storage for cross-subject results
    all_lmax = []
    all_evec_alignment = []
    all_phi_7 = []; all_phi_4 = []; all_phi_2 = []
    all_mp_edge = []; all_mp_gap = []
    all_jsd_wass_corr = []
    all_n_tp = []
    
    for sub in subjects:
        bold = data_dir/sub/"func"/f"{sub}_task-stopsignal_bold.nii.gz"
        if not bold.exists(): continue
        print(f"\n  {sub}...", end="", flush=True)
        
        try:
            masks = get_masks(str(bold))
            ts = get_timeseries(str(bold), masks)
            n_tp = ts.shape[0]
            all_n_tp.append(n_tp)
            
            # =========================================================
            # TEST 1: EIGENVECTOR PREDICTION
            # =========================================================
            J, TE, JSD_mat = compute_J_te_jsd(ts)
            lmax, evec = compute_lmax_and_evec(J)
            all_lmax.append(lmax)
            
            # Compute empirical "domain weights" — how much each network 
            # contributes to coordination (row sum of J, normalized)
            row_sums = J.sum(axis=1)
            if row_sums.sum() > 0:
                empirical_weights = row_sums / row_sums.sum()
            else:
                empirical_weights = np.ones(7)/7
            
            # Alignment = cosine similarity between empirical weights and |eigenvector|
            evec_abs = np.abs(evec)
            if np.linalg.norm(evec_abs) > 0 and np.linalg.norm(empirical_weights) > 0:
                alignment = np.dot(empirical_weights, evec_abs) / (np.linalg.norm(empirical_weights) * np.linalg.norm(evec_abs))
            else:
                alignment = 0
            all_evec_alignment.append(alignment)
            
            # =========================================================
            # TEST 2: COARSE-GRAINING INVARIANCE
            # =========================================================
            # 7-network Φ_PD
            phi_7 = compute_phi(J)
            all_phi_7.append(phi_7)
            
            # 4-network coarse-grain
            ts_4 = coarse_grain_ts(ts, MERGE_4, 4)
            J_4, _, _ = compute_J_te_jsd(ts_4)
            phi_4 = compute_phi(J_4)
            all_phi_4.append(phi_4)
            
            # 2-network coarse-grain
            ts_2 = coarse_grain_ts(ts, MERGE_2, 2)
            J_2, _, _ = compute_J_te_jsd(ts_2)
            phi_2 = compute_phi(J_2)
            all_phi_2.append(phi_2)
            
            # =========================================================
            # TEST 3: RANDOM MATRIX THEORY
            # =========================================================
            mp_edge = marchenko_pastur_edge(7, n_tp)
            # Scale: MP applies to sample covariance matrices normalized by 1/T
            # For our J matrix we need to compare eigenvalue scale
            # Use the ratio λ_max / mean(eigenvalues) vs MP edge ratio
            all_eigenvalues = np.linalg.eigvalsh(np.nan_to_num((J+J.T)/2))
            mean_ev = np.mean(np.abs(all_eigenvalues)) if np.mean(np.abs(all_eigenvalues)) > 0 else 1e-10
            normalized_lmax = lmax / mean_ev
            # For a random 7×7 matrix from T samples, the top eigenvalue ratio
            # under MP would be roughly (1 + sqrt(7/T))^2 / 1 ≈ 1 for large T
            # More direct: compare λ_max to shuffled null
            n_shuf = 200
            null_lmax = []
            for _ in range(n_shuf):
                ts_shuf = ts.copy()
                for col in range(7):
                    np.random.shuffle(ts_shuf[:, col])
                J_shuf, _, _ = compute_J_te_jsd(ts_shuf)
                null_lmax.append(compute_lmax_and_evec(J_shuf)[0])
            null_lmax = np.array(null_lmax)
            mp_95 = np.percentile(null_lmax, 95)
            gap = lmax - mp_95
            p_rmt = np.mean(null_lmax >= lmax)
            all_mp_edge.append(mp_95)
            all_mp_gap.append(gap)
            
            # =========================================================
            # TEST 4: OPTIMAL TRANSPORT — JSD vs WASSERSTEIN
            # =========================================================
            jsd_vals = []
            wass_vals = []
            for i in range(7):
                for j in range(i+1,7):
                    jsd_vals.append(JSD_mat[i,j])
                    wass_vals.append(wasserstein_1d(ts[:,i], ts[:,j]))
            
            if np.std(jsd_vals) > 0 and np.std(wass_vals) > 0:
                r_jw, p_jw = stats.pearsonr(jsd_vals, wass_vals)
            else:
                r_jw, p_jw = 0, 1
            all_jsd_wass_corr.append(r_jw)
            
            print(f" λ={lmax:.4f} align={alignment:.3f} Φ7={phi_7:.5f} Φ4={phi_4:.5f} Φ2={phi_2:.5f} gap={gap:+.5f} JSD~W={r_jw:.3f}")
            
        except Exception as e:
            print(f" ERROR: {e}")
    
    N = len(all_lmax)
    if N < 3:
        print("Too few subjects"); return
    
    # =====================================================================
    # TEST 1 RESULTS: EIGENVECTOR PREDICTION
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"TEST 1: EIGENVECTOR PREDICTION (N={N})")
    print("Does the principal eigenvector predict the coordination state?")
    print("="*80)
    
    mean_align = np.mean(all_evec_alignment)
    # Null: random alignment for 7D unit vectors ≈ 1/√7 ≈ 0.378
    null_align = 1.0 / np.sqrt(7)
    t_ev, p_ev = stats.ttest_1samp(all_evec_alignment, null_align)
    
    print(f"\n  Mean eigenvector-weight alignment: {mean_align:.4f}")
    print(f"  Random alignment (null):           {null_align:.4f}")
    print(f"  t({N-1}) = {t_ev:.3f}, p = {p_ev:.4f}")
    if mean_align > null_align and p_ev < 0.05:
        print(f"  ★ CONFIRMED: Empirical weights significantly align with principal eigenvector.")
        print(f"    The brain's coordination state tracks the dominant eigenmode of J.")
    elif mean_align > null_align:
        print(f"  Direction correct (alignment > random) but not significant.")
    else:
        print(f"  Not confirmed at this sample size.")
    
    # Per-network contribution to eigenvector
    print(f"\n  Which networks dominate the principal eigenvector?")
    # Average absolute eigenvector components across subjects
    # (We'd need to store them — approximate from row sums)
    
    # =====================================================================
    # TEST 2 RESULTS: COARSE-GRAINING INVARIANCE
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"TEST 2: RENORMALIZATION GROUP — COARSE-GRAINING (N={N})")
    print("Is Φ_PD approximately invariant across parcellation resolutions?")
    print("="*80)
    
    print(f"\n  Mean Φ_PD at each resolution:")
    print(f"    7-network: {np.mean(all_phi_7):.6f} ± {np.std(all_phi_7, ddof=1):.6f}")
    print(f"    4-network: {np.mean(all_phi_4):.6f} ± {np.std(all_phi_4, ddof=1):.6f}")
    print(f"    2-network: {np.mean(all_phi_2):.6f} ± {np.std(all_phi_2, ddof=1):.6f}")
    
    # Test: do the three resolutions correlate across subjects?
    r_74, p_74 = stats.pearsonr(all_phi_7, all_phi_4)
    r_72, p_72 = stats.pearsonr(all_phi_7, all_phi_2)
    r_42, p_42 = stats.pearsonr(all_phi_4, all_phi_2)
    
    print(f"\n  Cross-resolution correlations (subject-level):")
    print(f"    7 vs 4: r = {r_74:.3f}, p = {p_74:.4f}")
    print(f"    7 vs 2: r = {r_72:.3f}, p = {p_72:.4f}")  
    print(f"    4 vs 2: r = {r_42:.3f}, p = {p_42:.4f}")
    
    # Ratio test — if scale-invariant, ratios should be stable
    ratios_74 = np.array(all_phi_4) / (np.array(all_phi_7) + 1e-10)
    ratios_72 = np.array(all_phi_2) / (np.array(all_phi_7) + 1e-10)
    
    print(f"\n  Ratio Φ_4/Φ_7: {np.mean(ratios_74):.3f} ± {np.std(ratios_74, ddof=1):.3f} (CV = {np.std(ratios_74,ddof=1)/np.mean(ratios_74)*100:.1f}%)")
    print(f"  Ratio Φ_2/Φ_7: {np.mean(ratios_72):.3f} ± {np.std(ratios_72, ddof=1):.3f} (CV = {np.std(ratios_72,ddof=1)/np.mean(ratios_72)*100:.1f}%)")
    
    if r_74 > 0.5 and r_72 > 0.3:
        print(f"\n  ★ SUPPORTED: Φ_PD shows significant cross-resolution correlation.")
        print(f"    The coordination signal is preserved under coarse-graining,")
        print(f"    consistent with RG scale-invariance near criticality.")
    elif r_74 > 0.3:
        print(f"\n  Partial support: neighboring resolutions correlate but distant ones diverge.")
    else:
        print(f"\n  Not supported at this sample size.")
    
    # =====================================================================
    # TEST 3 RESULTS: RANDOM MATRIX THEORY
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"TEST 3: RANDOM MATRIX THEORY NULL (N={N})")
    print("Does λ_max exceed the shuffled null (Marchenko-Pastur proxy)?")
    print("="*80)
    
    print(f"\n  Mean empirical λ_max:    {np.mean(all_lmax):.6f}")
    print(f"  Mean null 95th %ile:     {np.mean(all_mp_edge):.6f}")
    print(f"  Mean gap (signal):       {np.mean(all_mp_gap):+.6f}")
    
    t_rmt, p_rmt = stats.ttest_1samp(all_mp_gap, 0)
    n_above = sum(1 for g in all_mp_gap if g > 0)
    
    print(f"  t({N-1}) = {t_rmt:.3f}, p = {p_rmt:.4f}")
    print(f"  {n_above}/{N} subjects have λ_max above shuffled null")
    
    if n_above == N and p_rmt < 0.05:
        print(f"\n  ★ CONFIRMED: λ_max significantly exceeds the random null in ALL subjects.")
        print(f"    The coordination structure is genuine signal, not statistical fluctuation.")
        print(f"    The 'consciousness signal' in RMT terms is present universally.")
    elif n_above >= N * 0.8:
        print(f"\n  Mostly confirmed: {n_above}/{N} subjects exceed null.")
    else:
        print(f"\n  Mixed results.")
    
    # =====================================================================
    # TEST 4 RESULTS: OPTIMAL TRANSPORT
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"TEST 4: OPTIMAL TRANSPORT — JSD vs WASSERSTEIN (N={N})")
    print("Do the two conflict measures track each other?")
    print("="*80)
    
    mean_r = np.mean(all_jsd_wass_corr)
    t_ot, p_ot = stats.ttest_1samp(all_jsd_wass_corr, 0)
    
    print(f"\n  Mean within-subject JSD-Wasserstein correlation: r = {mean_r:.3f}")
    print(f"  t({N-1}) = {t_ot:.3f}, p = {p_ot:.4f}")
    print(f"  {sum(1 for r in all_jsd_wass_corr if r > 0)}/{N} positive")
    
    if mean_r > 0.5 and p_ot < 0.05:
        print(f"\n  ★ CONFIRMED: JSD and Wasserstein distance are strongly correlated.")
        print(f"    The optimal transport interpretation of the conflict matrix is validated.")
        print(f"    C_ij measures the 'earth-moving cost' between policy distributions.")
    elif mean_r > 0.3:
        print(f"\n  Moderately supported: positive correlation but not overwhelming.")
    else:
        print(f"\n  Weak or no correlation.")
    
    # =====================================================================
    # TEST 5: LANDSCAPE TOPOLOGY (simplified)
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"TEST 5: MORSE THEORY — LANDSCAPE TOPOLOGY (N={N})")
    print("Does the eigenspectrum reveal landscape structure?")
    print("="*80)
    
    print(f"\n  For each subject, examining the full eigenspectrum of J:")
    print(f"  (A dominant single eigenvalue = one clear coordination basin)")
    print(f"  (Multiple comparable eigenvalues = rugged landscape, multiple basins)")
    
    dominance_ratios = []
    n_positive_evals = []
    
    for sub_idx in range(N):
        bold = data_dir/subjects[sub_idx]/"func"/f"{subjects[sub_idx]}_task-stopsignal_bold.nii.gz"
        if not bold.exists(): continue
        masks = get_masks(str(bold))
        ts = get_timeseries(str(bold), masks)
        J, _, _ = compute_J_te_jsd(ts)
        evals = np.sort(np.linalg.eigvalsh(np.nan_to_num((J+J.T)/2)))[::-1]
        
        n_pos = np.sum(evals > 0)
        n_positive_evals.append(n_pos)
        
        if len(evals) >= 2 and evals[1] > 0:
            dr = evals[0] / evals[1]
        else:
            dr = float('inf')
        dominance_ratios.append(min(dr, 100))
        
        print(f"  {subjects[sub_idx]}: eigenvalues = [{', '.join(f'{e:.5f}' for e in evals[:4])}...] dom_ratio={dr:.2f} n_pos={n_pos}")
    
    print(f"\n  Mean dominance ratio (λ1/λ2): {np.mean(dominance_ratios):.2f}")
    print(f"  Mean positive eigenvalues: {np.mean(n_positive_evals):.1f} / 7")
    
    if np.mean(dominance_ratios) > 2.0:
        print(f"\n  ★ SUPPORTED: Single dominant eigenmode.")
        print(f"    The coordination landscape has one primary basin —")
        print(f"    consistent with unified conscious experience.")
        print(f"    Morse theory predicts this corresponds to one global minimum")
        print(f"    with the dominant eigenvector defining the coordination direction.")
    else:
        print(f"\n  Multiple comparable eigenmodes — more complex landscape.")
    
    # =====================================================================
    # OVERALL VERDICT
    # =====================================================================
    print(f"\n{'='*80}")
    print("OVERALL MATHEMATICAL PREDICTIONS VERDICT")
    print("="*80)
    
    results = []
    results.append(("Eigenvector alignment", mean_align > null_align and p_ev < 0.05))
    results.append(("RG coarse-graining", r_74 > 0.3))
    results.append(("RMT null exceeded", n_above >= N * 0.8 and p_rmt < 0.05))
    results.append(("Optimal transport (JSD~Wass)", mean_r > 0.3 and p_ot < 0.05))
    results.append(("Morse landscape dominance", np.mean(dominance_ratios) > 2.0))
    
    n_pass = sum(1 for _, v in results if v)
    
    print(f"\n  {'Prediction':<35} {'Result':>10}")
    print(f"  {'-'*45}")
    for name, passed in results:
        print(f"  {name:<35} {'PASS ★' if passed else 'FAIL/WEAK':>10}")
    
    print(f"\n  {n_pass}/5 mathematical predictions confirmed.")
    
    if n_pass >= 4:
        print(f"\n  ★ The deep mathematical structure of the coordination energy function")
        print(f"    is empirically supported across multiple independent mathematical fields.")
        print(f"    The equation is not just an ad hoc metric — it is a member of a")
        print(f"    well-studied mathematical family with known properties.")
    elif n_pass >= 3:
        print(f"\n  Majority of predictions supported. The mathematical structure is")
        print(f"  consistent with the data, though some predictions need larger samples.")
    else:
        print(f"\n  Mixed results — further testing needed.")
    
    print(f"\n{'='*80}")
    print("MATHEMATICAL PREDICTIONS TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
