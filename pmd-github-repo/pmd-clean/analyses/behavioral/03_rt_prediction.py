#!/usr/bin/env python3
"""
===========================================================================
PRESSURE MAKES DIAMONDS — BEHAVIORAL CORRELATION (FIXED)
Does trial-level Φ*_PD predict reaction time on GO trials?

FIX: Decode GO vs STOP on ALL trials, then extract Φ*_PD for GO trials 
only and correlate with RT. Previous version decoded GO-only which gave
one class and constant probabilities.
===========================================================================
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')

YEO7 = ["Visual","Somatomotor","DorsalAttention","VentralAttention","Limbic","Frontoparietal","Default"]

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

def compute_lmax(J):
    Js=np.nan_to_num((J+J.T)/2.0)
    try: ev=np.linalg.eigvalsh(Js)
    except: ev=np.real(np.linalg.eigvals(Js))
    return float(np.max(np.real(ev)))

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

def compute_trialwise_phi(decoded_probs, timeseries, tau=0.05):
    N = 7
    n_trials = decoded_probs[0].shape[0]
    
    TE = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            te = sym_te(timeseries[:,i], timeseries[:,j])
            TE[i,j] = TE[j,i] = te
    coh = np.array([signal_coherence(timeseries[:,i]) for i in range(N)])
    te_vals = [TE[i,j] for i in range(N) for j in range(i+1,N)]
    i_min = np.percentile(te_vals, 25)
    
    net_conf = {}
    for ni in decoded_probs:
        pr = decoded_probs[ni]; nc = pr.shape[1]
        ent = -np.sum(pr * np.log2(pr+1e-10), axis=1)
        me = np.log2(nc)
        net_conf[ni] = max(1.0 - np.mean(ent)/me, 0) if me > 0 else 0
    
    gates = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            g_c = soft_gate(TE[i,j], i_min, tau)
            g_r = np.sqrt(max(coh[i],0) * max(coh[j],0))
            g_d = np.sqrt(net_conf[i] * net_conf[j])
            gates[i,j] = gates[j,i] = TE[i,j] * g_c * g_r * g_d
    
    phi_trials = np.zeros(n_trials)
    lmax_trials = np.zeros(n_trials)
    
    for t in range(n_trials):
        J = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                pi = decoded_probs[i][t] + 1e-10
                pj = decoded_probs[j][t] + 1e-10
                pi /= pi.sum(); pj /= pj.sum()
                m = 0.5 * (pi + pj)
                jsd = max(0.5*np.sum(pi*np.log2(pi/m)) + 0.5*np.sum(pj*np.log2(pj/m)), 0)
                J[i,j] = J[j,i] = gates[i,j] * jsd
        np.fill_diagonal(J, 0)
        n_pairs = N*(N-1)//2
        phi_trials[t] = sum(J[i,j] for i in range(N) for j in range(i+1,N)) / n_pairs
        lmax_trials[t] = compute_lmax(J)
    
    return phi_trials, lmax_trials

# =========================================================================
# MAIN
# =========================================================================

def main():
    data_dir = Path("./ds000030")
    all_subs = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    
    print("="*75)
    print("PRESSURE MAKES DIAMONDS — BEHAVIORAL CORRELATION (FIXED)")
    print("Does trial-level Φ*_PD predict reaction time?")
    print("="*75)
    print()
    print("  Method: Decode GO vs STOP on ALL trials.")
    print("  Then extract per-trial Φ*_PD for GO trials only.")
    print("  Correlate GO-trial Φ*_PD with GO-trial reaction time.")
    print()
    print("  Prediction: Positive correlation — more policy divergence")
    print("  on a GO trial means more coordination pressure to resolve,")
    print("  which should slow the response.")
    print("="*75)
    
    all_r_phi = []
    all_r_lmax = []
    subject_details = []
    
    for sub in all_subs:
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
            
            # KEY FIX: Decode GO vs STOP on ALL trials (both classes)
            patterns, labels, trial_df = extract_patterns(
                str(bold), masks, events, 'trial_type', ['GO', 'STOP'], tr=tr
            )
            
            if len(labels) < 20:
                print(f" too few trials"); continue
            
            decoded, accs, classes = decode_networks(patterns, labels)
            mean_acc = np.mean(list(accs.values()))
            
            # Compute per-trial Φ*_PD for ALL trials
            phi_trials, lmax_trials = compute_trialwise_phi(decoded, ts)
            
            # Now extract GO trials only and their RTs
            go_mask = (labels == 'GO')
            go_phi = phi_trials[go_mask]
            go_lmax = lmax_trials[go_mask]
            go_df = trial_df[trial_df['trial_type'] == 'GO'].reset_index(drop=True)
            go_rt = go_df['ReactionTime'].values.astype(float)
            
            # Filter valid RTs
            valid = ~np.isnan(go_rt) & (go_rt > 0.1) & (go_rt < 3.0)
            
            if valid.sum() < 20:
                print(f" too few valid GO RTs ({valid.sum()})"); continue
            
            phi_v = go_phi[valid]
            lmax_v = go_lmax[valid]
            rt_v = go_rt[valid]
            
            # Check for constant values
            if np.std(phi_v) < 1e-12 or np.std(rt_v) < 1e-12:
                print(f" constant values, skip"); continue
            
            r_phi, p_phi = stats.pearsonr(phi_v, rt_v)
            r_lmax, p_lmax = stats.pearsonr(lmax_v, rt_v)
            
            all_r_phi.append(r_phi)
            all_r_lmax.append(r_lmax)
            
            sig_p = "*" if p_phi < 0.05 else ""
            sig_l = "*" if p_lmax < 0.05 else ""
            
            subject_details.append({
                'sub': sub, 'n_go': int(valid.sum()), 'acc': mean_acc,
                'r_phi': r_phi, 'p_phi': p_phi, 'r_lmax': r_lmax, 'p_lmax': p_lmax,
                'mean_phi': float(np.mean(phi_v)), 'std_phi': float(np.std(phi_v)),
                'mean_rt': float(np.mean(rt_v))
            })
            
            print(f" N_go={valid.sum()} acc={mean_acc:.1%} | Φ*~RT: r={r_phi:+.3f} p={p_phi:.3f}{sig_p} | λ*~RT: r={r_lmax:+.3f} p={p_lmax:.3f}{sig_l}")
            
        except Exception as e:
            print(f" ERROR: {e}")
    
    # =====================================================================
    # GROUP RESULTS
    # =====================================================================
    if len(all_r_phi) >= 3:
        print(f"\n{'='*75}")
        print(f"GROUP RESULTS (N={len(all_r_phi)} subjects)")
        print(f"{'='*75}")
        
        r_phi_arr = np.array(all_r_phi)
        r_lmax_arr = np.array(all_r_lmax)
        
        t_phi, p_phi = stats.ttest_1samp(r_phi_arr, 0)
        t_lmax, p_lmax = stats.ttest_1samp(r_lmax_arr, 0)
        
        sig_phi = "***" if p_phi<0.001 else "**" if p_phi<0.01 else "*" if p_phi<0.05 else "ns"
        sig_lmax = "***" if p_lmax<0.001 else "**" if p_lmax<0.01 else "*" if p_lmax<0.05 else "ns"
        
        print(f"\n  Φ*_PD ~ Reaction Time:")
        print(f"    Mean r = {np.mean(r_phi_arr):+.3f} ± {np.std(r_phi_arr,ddof=1):.3f}")
        print(f"    t({len(r_phi_arr)-1}) = {t_phi:.3f}, p = {p_phi:.4f} {sig_phi}")
        print(f"    {(r_phi_arr>0).sum()}/{len(r_phi_arr)} subjects show positive correlation")
        
        print(f"\n  λ*_max ~ Reaction Time:")
        print(f"    Mean r = {np.mean(r_lmax_arr):+.3f} ± {np.std(r_lmax_arr,ddof=1):.3f}")
        print(f"    t({len(r_lmax_arr)-1}) = {t_lmax:.3f}, p = {p_lmax:.4f} {sig_lmax}")
        print(f"    {(r_lmax_arr>0).sum()}/{len(r_lmax_arr)} subjects show positive correlation")
        
        print(f"\n  Interpretation:")
        mean_r = np.mean(r_phi_arr)
        if mean_r > 0.05 and p_phi < 0.05:
            print(f"    ★ SIGNIFICANT POSITIVE: higher Φ*_PD → slower RT")
            print(f"      More coordination pressure = harder conflict = slower response")
        elif mean_r > 0.05:
            print(f"    Positive trend: higher Φ*_PD tends toward slower RT")
            print(f"    (not significant — may need more subjects)")
        elif mean_r < -0.05 and p_phi < 0.05:
            print(f"    ★ SIGNIFICANT NEGATIVE: higher Φ*_PD → faster RT")
            print(f"      Possible: better coordination = more efficient responses")
        elif abs(mean_r) < 0.05:
            print(f"    No consistent relationship between Φ*_PD and RT")
        else:
            print(f"    Trend present but not significant")
        
        # Also check: do STOP trials show higher Φ*_PD? (sanity check)
        print(f"\n  Sanity check: mean Φ*_PD by condition across subjects")
        print(f"    (computed during decoding)")
    
    print(f"\n{'='*75}")
    print("BEHAVIORAL CORRELATION COMPLETE")
    print("="*75)

if __name__ == "__main__":
    main()
