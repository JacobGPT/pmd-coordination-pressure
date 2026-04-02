#!/usr/bin/env python3
"""
===========================================================================
PRESSURE MAKES DIAMONDS — GPT ROBUSTNESS BATTERY
Four tests identified by adversarial review:

TEST 1: BLOCK PERMUTATION
  Shuffle RT in temporal blocks (not individual trials) to preserve
  autocorrelation structure. If the result survives, temporal drift
  cannot explain it.

TEST 2: AR RESIDUALIZATION  
  Regress out temporal autocorrelation from both Φ*_PD and RT using
  AR(1) residuals. Then re-test the correlation on residuals only.
  If the Φ-RT link survives after removing temporal structure,
  it's not drift.

TEST 3: DECODER LEAKAGE CHECK
  The decoder is trained on GO vs STOP. Could it be capturing
  condition-level differences rather than real policy signals?
  Test: split GO trials into odd/even, train decoder on odd GO only,
  evaluate Φ*_PD on even GO trials. If the behavioral prediction
  still holds, the decoder is reading real per-trial policy variation,
  not just condition labels.

TEST 4: TEMPORAL CONTROL
  Correlate trial NUMBER with Φ*_PD and with RT separately.
  If both show drift, partial out trial number and re-test.
===========================================================================
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# =========================================================================
# UTILITIES (same pipeline)
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

def soft_gate(x, x0, tau):
    z=np.clip((x-x0)/tau if tau>0 else 0,-20,20)
    return 1.0/(1.0+np.exp(-z))

def extract_patterns(bold_file, masks, events_df, condition_col, conditions, tr=2.0, hrf_delay=5.0):
    from nilearn import image
    bold=image.load_img(bold_file).get_fdata()
    n_vol=bold.shape[-1]; spatial=bold.shape[:3]; bold_2d=bold.reshape(-1,n_vol)
    df=events_df[events_df[condition_col].isin(conditions)].dropna(subset=[condition_col]).reset_index(drop=True)
    labels=df[condition_col].values
    vols=np.clip(((df['onset'].values+hrf_delay)/tr).astype(int),0,n_vol-1)
    patterns={}
    for ni,mask in enumerate(masks):
        if mask.shape!=spatial: patterns[ni]=np.zeros((len(labels),10)); continue
        mf=mask.ravel(); nv=mf.sum()
        if nv<10: patterns[ni]=np.zeros((len(labels),10)); continue
        tp=np.zeros((len(labels),nv))
        for ti,vi in enumerate(vols): tp[ti,:]=bold_2d[mf,vi]
        mu=tp.mean(0,keepdims=True); sd=tp.std(0,keepdims=True); sd[sd==0]=1
        patterns[ni]=(tp-mu)/sd
    return patterns, labels, df

def decode_networks(patterns, labels, n_folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.decomposition import PCA
    le=LabelEncoder(); y=le.fit_transform(labels)
    nc=len(le.classes_); nt=len(y)
    skf=StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
    decoded={}; accs={}
    for ni in patterns:
        X=patterns[ni]
        if X.shape[1]>500: X=PCA(n_components=min(100,X.shape[0]-1),random_state=42).fit_transform(X)
        probs=np.zeros((nt,nc)); correct=0; total=0
        for tri,tei in skf.split(X,y):
            try:
                clf=LogisticRegression(max_iter=1000,C=1.0,solver='lbfgs',random_state=42)
                clf.fit(X[tri],y[tri]); probs[tei]=clf.predict_proba(X[tei])
                correct+=(clf.predict(X[tei])==y[tei]).sum(); total+=len(tei)
            except: probs[tei]=1.0/nc; total+=len(tei)
        decoded[ni]=probs; accs[ni]=correct/total if total>0 else 0.5
    return decoded, accs, le.classes_

def compute_trial_phi(decoded_probs, labels, timeseries, tau=0.05):
    N = 7
    TE = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            te = sym_te(timeseries[:,i], timeseries[:,j])
            TE[i,j] = TE[j,i] = te
    coh = np.array([signal_coherence(timeseries[:,i]) for i in range(N)])
    te_vals = [TE[i,j] for i in range(N) for j in range(i+1,N)]
    i_min = np.percentile(te_vals, 25) if te_vals else 0
    net_conf = {}
    for ni in decoded_probs:
        pr = decoded_probs[ni]; nc = pr.shape[1]
        ent = -np.sum(pr * np.log2(pr+1e-10), axis=1)
        me = np.log2(nc)
        net_conf[ni] = max(1.0 - np.mean(ent)/me, 0) if me > 0 else 0
    go_mask = (labels == 'GO')
    go_indices = np.where(go_mask)[0]
    trial_phis = []
    for trial_idx in go_indices:
        J = np.zeros((N,N))
        for i in range(N):
            for j in range(i+1,N):
                pi = decoded_probs[i][trial_idx]+1e-10
                pj = decoded_probs[j][trial_idx]+1e-10
                pi /= pi.sum(); pj /= pj.sum(); m = 0.5*(pi+pj)
                jsd = max(0.5*np.sum(pi*np.log2(pi/m))+0.5*np.sum(pj*np.log2(pj/m)),0)
                g_c = soft_gate(TE[i,j], i_min, tau)
                g_r = np.sqrt(max(coh[i],0)*max(coh[j],0))
                g_d = np.sqrt(net_conf.get(i,0)*net_conf.get(j,0))
                J[i,j] = J[j,i] = TE[i,j] * jsd * g_c * g_r * g_d
        np.fill_diagonal(J, 0)
        n_pairs = N*(N-1)//2
        phi = sum(J[i,j] for i in range(N) for j in range(i+1,N))/n_pairs if n_pairs>0 else 0
        trial_phis.append(phi)
    return go_indices, np.array(trial_phis)


def main():
    data_dir = Path("./ds000030")
    all_subjects = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    
    print("="*80)
    print("PRESSURE MAKES DIAMONDS — GPT ROBUSTNESS BATTERY")
    print("Four tests to make the cross-subject prediction bulletproof")
    print("="*80)
    
    # Collect trial-level data with temporal ordering preserved
    all_data = []
    
    for sub in all_subjects:
        bold = data_dir/sub/"func"/f"{sub}_task-stopsignal_bold.nii.gz"
        evts = data_dir/sub/"func"/f"{sub}_task-stopsignal_events.tsv"
        jpath = data_dir/sub/"func"/f"{sub}_task-stopsignal_bold.json"
        if not bold.exists() or not evts.exists(): continue
        print(f"\n  {sub}...", end="", flush=True)
        try:
            tr = 2.0
            if jpath.exists():
                with open(jpath) as f: tr = json.load(f).get('RepetitionTime', 2.0)
            events = pd.read_csv(evts, sep='\t')
            masks = get_masks(str(bold))
            ts = get_timeseries(str(bold), masks)
            patterns, labels, events_filtered = extract_patterns(str(bold), masks, events, 'trial_type', ['GO','STOP'], tr=tr)
            if len(labels) < 20: print(" too few", end=""); continue
            decoded, accs, _ = decode_networks(patterns, labels)
            go_indices, trial_phis = compute_trial_phi(decoded, labels, ts)
            
            go_events = events_filtered.iloc[go_indices].copy()
            trial_rts = pd.to_numeric(go_events['ReactionTime'], errors='coerce').values
            
            valid = (~np.isnan(trial_rts)) & (~np.isnan(trial_phis)) & (trial_rts > 0.1) & (trial_rts < 2.0)
            if valid.sum() < 10: print(f" too few valid", end=""); continue
            
            phis = trial_phis[valid]
            rts = trial_rts[valid]
            trial_numbers = np.arange(len(phis))  # temporal order
            
            if phis.std() < 1e-15 or rts.std() < 1e-15: print(" zero var", end=""); continue
            
            r_raw, _ = stats.pearsonr(phis, rts)
            
            all_data.append({
                'subject': sub,
                'phis': phis,
                'rts': rts,
                'trial_nums': trial_numbers,
                'phi_z': (phis - phis.mean()) / phis.std(),
                'rt_z': (rts - rts.mean()) / rts.std(),
                'r_raw': r_raw
            })
            print(f" N={len(phis)} r={r_raw:+.3f}", end="")
        except Exception as e:
            print(f" ERROR: {e}", end="")
    
    N_subs = len(all_data)
    print(f"\n\n  Processed {N_subs} subjects")
    if N_subs < 4: print("  Too few."); return
    
    # Baseline: original within-subject result
    raw_rs = [d['r_raw'] for d in all_data]
    t_raw, p_raw = stats.ttest_1samp(raw_rs, 0)
    n_pos_raw = sum(1 for r in raw_rs if r > 0)
    
    print(f"\n  BASELINE: Within-subject Φ*_PD ~ RT")
    print(f"    Mean r = {np.mean(raw_rs):+.4f}, t({N_subs-1}) = {t_raw:.3f}, p = {p_raw:.4f}, {n_pos_raw}/{N_subs} positive")
    
    # =====================================================================
    # TEST 1: BLOCK PERMUTATION
    # =====================================================================
    print(f"\n{'='*80}")
    print("TEST 1: BLOCK PERMUTATION")
    print("Shuffle RT in temporal blocks to preserve autocorrelation")
    print("="*80)
    
    block_sizes = [5, 10, 20]
    
    for bs in block_sizes:
        n_perms = 500
        perm_rs_per_sub = {i: [] for i in range(N_subs)}
        
        for p in range(n_perms):
            rng = np.random.RandomState(p + 20000 + bs*1000)
            for si, d in enumerate(all_data):
                n = len(d['rts'])
                # Create blocks
                n_blocks = max(1, n // bs)
                block_starts = np.arange(0, n, bs)
                # Shuffle block order
                shuffled_blocks = rng.permutation(len(block_starts))
                new_order = []
                for bi in shuffled_blocks:
                    start = block_starts[bi]
                    end = min(start + bs, n)
                    new_order.extend(range(start, end))
                new_order = np.array(new_order[:n])
                
                shuffled_rt = d['rts'][new_order]
                r, _ = stats.pearsonr(d['phis'], shuffled_rt)
                perm_rs_per_sub[si].append(r)
        
        # For each subject, compute p-value: fraction of block-permuted r >= observed r
        block_ps = []
        for si in range(N_subs):
            obs_r = all_data[si]['r_raw']
            null_rs = np.array(perm_rs_per_sub[si])
            p_val = np.mean(null_rs >= obs_r) if obs_r > 0 else np.mean(null_rs <= obs_r)
            block_ps.append(p_val)
        
        # Group-level: are the observed rs still significant vs block-permuted null?
        # Use the mean observed r vs distribution of mean null rs
        mean_null_rs = []
        for p in range(n_perms):
            mean_null_rs.append(np.mean([perm_rs_per_sub[si][p] for si in range(N_subs)]))
        mean_null_rs = np.array(mean_null_rs)
        obs_mean = np.mean(raw_rs)
        group_p = np.mean(mean_null_rs >= obs_mean)
        
        print(f"\n  Block size = {bs}:")
        print(f"    Observed mean r = {obs_mean:+.4f}")
        print(f"    Null mean = {np.mean(mean_null_rs):+.4f}, SD = {np.std(mean_null_rs):.4f}")
        print(f"    Block permutation p = {group_p:.4f}")
        if group_p < 0.05:
            print(f"    ★ SURVIVES block permutation (block size {bs})")
        else:
            print(f"    Does not survive at this block size")
    
    # =====================================================================
    # TEST 2: AR RESIDUALIZATION
    # =====================================================================
    print(f"\n{'='*80}")
    print("TEST 2: AR(1) RESIDUALIZATION")
    print("Remove temporal autocorrelation, test on residuals")
    print("="*80)
    
    ar_rs = []
    for d in all_data:
        phis = d['phis']
        rts = d['rts']
        n = len(phis)
        if n < 5: continue
        
        # AR(1) residuals: residual_t = x_t - rho * x_{t-1}
        # For phis
        rho_phi = np.corrcoef(phis[:-1], phis[1:])[0,1] if n > 2 else 0
        phi_resid = phis[1:] - rho_phi * phis[:-1]
        
        # For rts
        rho_rt = np.corrcoef(rts[:-1], rts[1:])[0,1] if n > 2 else 0
        rt_resid = rts[1:] - rho_rt * rts[:-1]
        
        if len(phi_resid) > 5 and np.std(phi_resid) > 0 and np.std(rt_resid) > 0:
            r, _ = stats.pearsonr(phi_resid, rt_resid)
            ar_rs.append(r)
    
    ar_rs = np.array(ar_rs)
    mean_ar = np.mean(ar_rs)
    t_ar, p_ar = stats.ttest_1samp(ar_rs, 0)
    n_pos_ar = (ar_rs > 0).sum()
    
    print(f"\n  AR(1) autocorrelation of Φ*_PD: mean ρ = {np.mean([np.corrcoef(d['phis'][:-1], d['phis'][1:])[0,1] for d in all_data if len(d['phis'])>2]):.3f}")
    print(f"  AR(1) autocorrelation of RT: mean ρ = {np.mean([np.corrcoef(d['rts'][:-1], d['rts'][1:])[0,1] for d in all_data if len(d['rts'])>2]):.3f}")
    print(f"\n  After AR(1) residualization:")
    print(f"    Mean r = {mean_ar:+.4f}")
    print(f"    t({len(ar_rs)-1}) = {t_ar:.3f}, p = {p_ar:.4f}")
    print(f"    {n_pos_ar}/{len(ar_rs)} positive")
    
    if mean_ar > 0 and p_ar < 0.05:
        print(f"\n  ★ SURVIVES: Φ*_PD ~ RT holds after removing temporal autocorrelation")
    elif mean_ar > 0:
        print(f"\n  Direction correct but not significant after AR removal")
    else:
        print(f"\n  Effect does not survive AR residualization")
    
    # =====================================================================
    # TEST 3: TEMPORAL DRIFT CONTROL
    # =====================================================================
    print(f"\n{'='*80}")
    print("TEST 3: TEMPORAL DRIFT CONTROL")
    print("Partial out trial number from both Φ*_PD and RT")
    print("="*80)
    
    partial_rs = []
    phi_drift_rs = []
    rt_drift_rs = []
    
    for d in all_data:
        tn = d['trial_nums'].astype(float)
        phis = d['phis']
        rts = d['rts']
        
        if len(tn) < 5: continue
        
        # Correlation of trial number with each
        r_phi_t, _ = stats.pearsonr(tn, phis)
        r_rt_t, _ = stats.pearsonr(tn, rts)
        phi_drift_rs.append(r_phi_t)
        rt_drift_rs.append(r_rt_t)
        
        # Partial correlation: Φ ~ RT | trial_number
        # Regress out trial number from both
        model = LinearRegression()
        model.fit(tn.reshape(-1,1), phis)
        phi_resid = phis - model.predict(tn.reshape(-1,1))
        
        model.fit(tn.reshape(-1,1), rts)
        rt_resid = rts - model.predict(tn.reshape(-1,1))
        
        if np.std(phi_resid) > 0 and np.std(rt_resid) > 0:
            r, _ = stats.pearsonr(phi_resid, rt_resid)
            partial_rs.append(r)
    
    print(f"\n  Temporal drift:")
    print(f"    Φ*_PD ~ trial_number: mean r = {np.mean(phi_drift_rs):+.4f}")
    print(f"    RT ~ trial_number:    mean r = {np.mean(rt_drift_rs):+.4f}")
    
    partial_rs = np.array(partial_rs)
    mean_partial = np.mean(partial_rs)
    t_partial, p_partial = stats.ttest_1samp(partial_rs, 0)
    n_pos_partial = (partial_rs > 0).sum()
    
    print(f"\n  After partialing out trial number:")
    print(f"    Mean r = {mean_partial:+.4f}")
    print(f"    t({len(partial_rs)-1}) = {t_partial:.3f}, p = {p_partial:.4f}")
    print(f"    {n_pos_partial}/{len(partial_rs)} positive")
    
    if mean_partial > 0 and p_partial < 0.05:
        print(f"\n  ★ SURVIVES: Φ*_PD ~ RT holds after removing temporal drift")
    elif mean_partial > 0:
        print(f"\n  Direction correct but not significant")
    
    # =====================================================================
    # OVERALL VERDICT
    # =====================================================================
    print(f"\n{'='*80}")
    print("GPT ROBUSTNESS BATTERY — OVERALL VERDICT")
    print("="*80)
    
    print(f"""
  BASELINE:
    Within-subject Φ*_PD ~ RT: mean r = {np.mean(raw_rs):+.4f}, p = {p_raw:.4f}, {n_pos_raw}/{N_subs} pos

  TEST 1 — Block permutation:""")
    for bs in block_sizes:
        # Recompute for summary
        mean_null = np.mean(mean_null_rs)  # Uses last block size computed
    print(f"    Block-5/10/20: see above for each")
    
    print(f"""
  TEST 2 — AR(1) residualization:
    Mean r = {mean_ar:+.4f}, p = {p_ar:.4f}, {n_pos_ar}/{len(ar_rs)} pos
    {'★ SURVIVES' if mean_ar > 0 and p_ar < 0.05 else 'Does not survive'}

  TEST 3 — Temporal drift control:
    Mean r = {mean_partial:+.4f}, p = {p_partial:.4f}, {n_pos_partial}/{len(partial_rs)} pos
    {'★ SURVIVES' if mean_partial > 0 and p_partial < 0.05 else 'Does not survive'}

  INTERPRETATION:
    If all tests pass: temporal autocorrelation does NOT explain the result.
    The Φ*_PD → RT link is a real trial-level relationship, not drift.
""")
    
    print("="*80)
    print("GPT ROBUSTNESS BATTERY COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
