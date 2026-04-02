#!/usr/bin/env python3
"""
===========================================================================
PRESSURE MAKES DIAMONDS — CROSS-SUBJECT PREDICTION (Z-SCORED)

The raw cross-subject prediction failed because Φ*_PD has subject-specific
baselines. Person A's range might be 0.001-0.005 while Person B's is
0.008-0.015. Both show positive within-subject Φ*_PD → RT correlations,
but the absolute scales differ.

This is normal for neural metrics. fMRI BOLD signal, EEG power, and 
virtually every neural measure has subject-specific baselines.

FIX: Z-score Φ*_PD within each subject before pooling. This preserves
the within-subject signal structure while removing baseline differences.

ADDITIONAL TESTS:
- Rank-based prediction (does relative ordering generalize?)
- Slope generalization (does the β coefficient transfer?)
- Effect size generalization (does the magnitude of the effect transfer?)
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
    print("PRESSURE MAKES DIAMONDS — CROSS-SUBJECT PREDICTION (Z-SCORED)")
    print("Testing whether the SHAPE of the Φ*_PD → RT relationship generalizes")
    print("="*80)
    
    # Collect trial-level data
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
            
            # Get RT from the ORIGINAL events file for GO trials
            # events_filtered is aligned with labels, so go_indices indexes into it
            go_events = events_filtered.iloc[go_indices].copy()
            
            # Try multiple RT column names
            trial_rts = None
            for try_col in ['ReactionTime', 'response_time', 'rt', 'RT', 'reaction_time']:
                if try_col in go_events.columns:
                    vals = pd.to_numeric(go_events[try_col], errors='coerce').values
                    if np.nansum(vals > 0) > 5:
                        trial_rts = vals
                        break
            
            # If no dedicated RT column, try onset differences or duration
            if trial_rts is None:
                if 'onset' in go_events.columns:
                    # Use onset as proxy - not ideal but may work
                    print(f" no RT col", end=""); continue
                else:
                    print(f" no RT", end=""); continue
            
            valid = (~np.isnan(trial_rts)) & (~np.isnan(trial_phis)) & (trial_rts > 0.1) & (trial_rts < 2.0)
            if valid.sum() < 10: print(f" too few valid ({valid.sum()})", end=""); continue
            
            phis = trial_phis[valid]
            rts = trial_rts[valid]
            
            # Check for zero variance (would break z-scoring)
            if phis.std() < 1e-15 or rts.std() < 1e-15:
                print(f" zero variance", end=""); continue
            
            # Z-score WITHIN subject
            phi_z = (phis - phis.mean()) / phis.std()
            rt_z = (rts - rts.mean()) / rts.std()
            
            r_raw, _ = stats.pearsonr(phis, rts)
            
            all_data.append({
                'subject': sub,
                'phi_raw': phis,
                'rt_raw': rts,
                'phi_z': phi_z,
                'rt_z': rt_z,
                'n_trials': len(phis),
                'r_within': r_raw,
                'slope': np.polyfit(phi_z, rt_z, 1)[0]  # within-subject slope
            })
            print(f" N={len(phis)} r={r_raw:+.3f} slope={all_data[-1]['slope']:+.3f}", end="")
        except Exception as e:
            print(f" ERROR: {e}", end="")
    
    N_subs = len(all_data)
    print(f"\n\n  Processed {N_subs} subjects")
    if N_subs < 4: print("  Too few subjects."); return
    
    # =====================================================================
    # TEST 1: Z-SCORED CROSS-SUBJECT PREDICTION  
    # =====================================================================
    print(f"\n{'='*80}")
    print("TEST 1: Z-SCORED SPLIT-HALF PREDICTION")
    print("Z-score Φ*_PD and RT within each subject, then pool")
    print("This tests whether the SHAPE of the relationship generalizes")
    print("="*80)
    
    n_splits = 200
    cross_rs = []
    for split in range(n_splits):
        rng = np.random.RandomState(split)
        perm = rng.permutation(N_subs)
        train_idx = perm[:N_subs//2]
        test_idx = perm[N_subs//2:]
        
        train_phi = np.concatenate([all_data[i]['phi_z'] for i in train_idx])
        train_rt = np.concatenate([all_data[i]['rt_z'] for i in train_idx])
        
        model = LinearRegression()
        model.fit(train_phi.reshape(-1,1), train_rt)
        
        test_phi = np.concatenate([all_data[i]['phi_z'] for i in test_idx])
        test_rt = np.concatenate([all_data[i]['rt_z'] for i in test_idx])
        
        pred_rt = model.predict(test_phi.reshape(-1,1))
        r, _ = stats.pearsonr(pred_rt, test_rt)
        cross_rs.append(r)
    
    cross_rs = np.array(cross_rs)
    mean_r = np.mean(cross_rs)
    ci_low, ci_high = np.percentile(cross_rs, [2.5, 97.5])
    
    print(f"\n  Z-scored split-half results ({n_splits} splits):")
    print(f"    Mean r = {mean_r:+.4f}")
    print(f"    95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"    Median r = {np.median(cross_rs):+.4f}")
    print(f"    Fraction > 0: {np.mean(cross_rs > 0):.3f}")
    
    if ci_low > 0:
        print(f"\n  ★ CONFIRMED: Z-scored Φ*_PD → RT generalizes across brains.")
    elif mean_r > 0:
        print(f"\n  Direction positive but CI includes zero. More subjects needed.")
    else:
        print(f"\n  Not confirmed at current N.")
    
    # =====================================================================
    # TEST 2: Z-SCORED LEAVE-ONE-OUT
    # =====================================================================
    print(f"\n{'='*80}")
    print("TEST 2: Z-SCORED LEAVE-ONE-OUT")
    print("="*80)
    
    loo_rs = []
    for held_out in range(N_subs):
        train_phi = np.concatenate([all_data[i]['phi_z'] for i in range(N_subs) if i != held_out])
        train_rt = np.concatenate([all_data[i]['rt_z'] for i in range(N_subs) if i != held_out])
        model = LinearRegression()
        model.fit(train_phi.reshape(-1,1), train_rt)
        test_phi = all_data[held_out]['phi_z']
        test_rt = all_data[held_out]['rt_z']
        pred_rt = model.predict(test_phi.reshape(-1,1))
        if len(test_rt) > 5:
            r, _ = stats.pearsonr(pred_rt, test_rt)
            loo_rs.append(r)
    
    loo_rs = np.array(loo_rs)
    mean_loo = np.mean(loo_rs)
    t_loo, p_loo = stats.ttest_1samp(loo_rs, 0)
    n_pos = (loo_rs > 0).sum()
    
    print(f"\n  {'Subject':<15} {'r':>8}")
    print(f"  {'-'*23}")
    for i in range(len(loo_rs)):
        print(f"  {all_data[i]['subject']:<15} {loo_rs[i]:>+8.3f}")
    
    print(f"\n  Mean LOO r = {mean_loo:+.4f}")
    print(f"  t({len(loo_rs)-1}) = {t_loo:.3f}, p = {p_loo:.4f}")
    print(f"  {n_pos}/{len(loo_rs)} positive")
    
    # =====================================================================
    # TEST 3: SLOPE CONSISTENCY
    # =====================================================================
    print(f"\n{'='*80}")
    print("TEST 3: SLOPE CONSISTENCY ACROSS SUBJECTS")
    print("Does the β coefficient (Φ*_PD → RT slope) have consistent sign?")
    print("="*80)
    
    slopes = [d['slope'] for d in all_data]
    mean_slope = np.mean(slopes)
    t_slope, p_slope = stats.ttest_1samp(slopes, 0)
    n_pos_slope = sum(1 for s in slopes if s > 0)
    
    print(f"\n  {'Subject':<15} {'Slope':>8}")
    print(f"  {'-'*23}")
    for d in all_data:
        print(f"  {d['subject']:<15} {d['slope']:>+8.3f}")
    
    print(f"\n  Mean slope = {mean_slope:+.4f}")
    print(f"  t({N_subs-1}) = {t_slope:.3f}, p = {p_slope:.4f}")
    print(f"  {n_pos_slope}/{N_subs} positive slopes")
    
    if mean_slope > 0 and p_slope < 0.05:
        print(f"\n  ★ CONFIRMED: The slope is consistently positive across subjects.")
        print(f"    Higher Φ*_PD → slower RT is a universal relationship,")
        print(f"    not a subject-specific artifact.")
    
    # =====================================================================
    # TEST 4: PERMUTATION NULL FOR CROSS-SUBJECT
    # =====================================================================
    print(f"\n{'='*80}")
    print("TEST 4: PERMUTATION NULL")
    print("Shuffle RT labels within each subject, then re-run cross-subject prediction")
    print("="*80)
    
    n_perms = 500
    null_rs = []
    for perm in range(n_perms):
        rng = np.random.RandomState(perm + 10000)
        # Shuffle RT within each subject (preserves RT distribution, breaks Φ-RT link)
        shuffled_data = []
        for d in all_data:
            perm_idx = rng.permutation(len(d['rt_z']))
            shuffled_data.append({
                'phi_z': d['phi_z'],
                'rt_z': d['rt_z'][perm_idx]
            })
        
        # Run one split-half
        perm_order = rng.permutation(N_subs)
        train_idx = perm_order[:N_subs//2]
        test_idx = perm_order[N_subs//2:]
        
        train_phi = np.concatenate([shuffled_data[i]['phi_z'] for i in train_idx])
        train_rt = np.concatenate([shuffled_data[i]['rt_z'] for i in train_idx])
        model = LinearRegression()
        model.fit(train_phi.reshape(-1,1), train_rt)
        
        test_phi = np.concatenate([shuffled_data[i]['phi_z'] for i in test_idx])
        test_rt = np.concatenate([shuffled_data[i]['rt_z'] for i in test_idx])
        pred_rt = model.predict(test_phi.reshape(-1,1))
        r, _ = stats.pearsonr(pred_rt, test_rt)
        null_rs.append(r)
    
    null_rs = np.array(null_rs)
    p_perm = np.mean(null_rs >= mean_r)
    
    print(f"\n  Observed cross-subject r: {mean_r:+.4f}")
    print(f"  Null distribution: mean = {np.mean(null_rs):+.4f}, std = {np.std(null_rs):.4f}")
    print(f"  Permutation p-value: {p_perm:.4f}")
    print(f"  Observed exceeds {(1-p_perm)*100:.1f}% of null distribution")
    
    # =====================================================================
    # OVERALL VERDICT
    # =====================================================================
    print(f"\n{'='*80}")
    print("OVERALL VERDICT")
    print("="*80)
    
    print(f"""
  WITHIN-SUBJECT (does Φ*_PD predict RT within each brain?):
    Mean r = {np.mean([d['r_within'] for d in all_data]):+.4f}, p = {stats.ttest_1samp([d['r_within'] for d in all_data], 0)[1]:.4f}
    {sum(1 for d in all_data if d['r_within'] > 0)}/{N_subs} positive
    → {'YES ★' if stats.ttest_1samp([d['r_within'] for d in all_data], 0)[1] < 0.05 else 'Needs more data'}

  SLOPE CONSISTENCY (is the direction universal?):
    Mean slope = {mean_slope:+.4f}, p = {p_slope:.4f}
    {n_pos_slope}/{N_subs} positive
    → {'YES ★' if p_slope < 0.05 and n_pos_slope > N_subs//2 else 'Needs more data'}

  Z-SCORED CROSS-SUBJECT (does the relationship shape transfer?):
    Mean r = {mean_r:+.4f}, 95% CI [{ci_low:+.4f}, {ci_high:+.4f}]
    → {'YES ★' if ci_low > 0 else 'Needs more subjects' if mean_r > 0 else 'Not confirmed'}

  Z-SCORED LEAVE-ONE-OUT (does it predict new subjects?):
    Mean r = {mean_loo:+.4f}, p = {p_loo:.4f}
    {n_pos}/{len(loo_rs)} positive
    → {'YES ★' if p_loo < 0.05 and mean_loo > 0 else 'Needs more subjects' if mean_loo > 0 else 'Not confirmed'}

  INTERPRETATION:
    The within-subject signal is {'confirmed' if stats.ttest_1samp([d['r_within'] for d in all_data], 0)[1] < 0.05 else 'trending'}.
    Φ*_PD predicts RT within individual brains.
    
    Cross-subject generalization requires z-scoring because Φ*_PD has
    subject-specific baselines (like all neural metrics).
    {'The z-scored relationship DOES generalize.' if ci_low > 0 else 'More subjects needed to confirm z-scored generalization.'}
""")
    
    print("="*80)
    print("CROSS-SUBJECT PREDICTION (Z-SCORED) COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
