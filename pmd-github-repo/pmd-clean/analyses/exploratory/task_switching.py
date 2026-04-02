#!/usr/bin/env python3
"""
===========================================================================
PRESSURE MAKES DIAMONDS — LEVEL 3: TASK SWITCHING
Step 2 on the testing ladder: Rule-switch conflict
===========================================================================

PREDICTIONS:
  1. SWITCH > NOSWITCH in Φ*_PD (rule change = inter-domain conflict)
  2. INCONGRUENT > CONGRUENT in Φ*_PD (competing response mappings)
  
Both are policy disagreement manipulations that should increase
coordination pressure between networks.
===========================================================================
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')

YEO7 = ["Visual","Somatomotor","DorsalAttention","VentralAttention","Limbic","Frontoparietal","Default"]

# =========================================================================
# UTILITIES (same as Level 3 POC)
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
    return patterns, labels

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

def compute_level3(decoded_probs, timeseries, tau=0.05):
    N=7
    TE=np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            te=sym_te(timeseries[:,i],timeseries[:,j]); TE[i,j]=TE[j,i]=te
    coh=np.array([signal_coherence(timeseries[:,i]) for i in range(N)])
    te_vals=[TE[i,j] for i in range(N) for j in range(i+1,N)]
    i_min=np.percentile(te_vals,25)
    net_conf={}
    for ni in decoded_probs:
        pr=decoded_probs[ni]; nc=pr.shape[1]
        ent=-np.sum(pr*np.log2(pr+1e-10),axis=1); me=np.log2(nc)
        net_conf[ni]=max(1.0-np.mean(ent)/me,0) if me>0 else 0
    C=np.zeros((N,N)); G_dec=np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            jsds=[]
            for t in range(decoded_probs[i].shape[0]):
                pi=decoded_probs[i][t]+1e-10; pj=decoded_probs[j][t]+1e-10
                pi/=pi.sum(); pj/=pj.sum(); m=0.5*(pi+pj)
                jsds.append(max(0.5*np.sum(pi*np.log2(pi/m))+0.5*np.sum(pj*np.log2(pj/m)),0))
            C[i,j]=C[j,i]=np.mean(jsds)
            G_dec[i,j]=G_dec[j,i]=np.sqrt(net_conf[i]*net_conf[j])
    J=np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            g_c=soft_gate(TE[i,j],i_min,tau); g_r=np.sqrt(max(coh[i],0)*max(coh[j],0))
            J[i,j]=J[j,i]=TE[i,j]*C[i,j]*g_c*g_r*G_dec[i,j]
    np.fill_diagonal(J,0)
    phi=sum(J[i,j] for i in range(N) for j in range(i+1,N))/(N*(N-1)//2)
    return {'phi':float(phi),'lmax':compute_lmax(J)}

def compute_percond(decoded_probs, labels, cond, timeseries, tau=0.05):
    mask=(labels==cond)
    if mask.sum()<5: return None
    return compute_level3({k:v[mask] for k,v in decoded_probs.items()}, timeseries, tau)

# =========================================================================
# MAIN
# =========================================================================

def main():
    data_dir = Path("./ds000030")
    subjects = ['sub-10159','sub-10206','sub-10269','sub-10271','sub-10273',
                'sub-10274','sub-10280','sub-10290','sub-10292']  # sub-10299 has no taskswitch
    
    print("="*75)
    print("PRESSURE MAKES DIAMONDS — LEVEL 3: TASK SWITCHING")
    print("Step 2: Rule-switch conflict (N=9)")
    print("="*75)
    print()
    print("  PREDICTIONS:")
    print("    1. SWITCH > NOSWITCH (rule change = policy conflict)")
    print("    2. INCONGRUENT > CONGRUENT (competing response mappings)")
    print("="*75)
    
    # =====================================================================
    # ANALYSIS 1: SWITCH vs NOSWITCH
    # =====================================================================
    print(f"\n{'='*75}")
    print("ANALYSIS 1: SWITCH vs NOSWITCH")
    print("="*75)
    
    switch_results = []
    
    for sub in subjects:
        bold = data_dir/sub/"func"/f"{sub}_task-taskswitch_bold.nii.gz"
        evts = data_dir/sub/"func"/f"{sub}_task-taskswitch_events.tsv"
        jpath = data_dir/sub/"func"/f"{sub}_task-taskswitch_bold.json"
        
        if not bold.exists(): continue
        print(f"\n  {sub}...", end="", flush=True)
        
        tr = 2.0
        if jpath.exists():
            with open(jpath) as f: tr = json.load(f).get('RepetitionTime', 2.0)
        
        events = pd.read_csv(evts, sep='\t')
        masks = get_masks(str(bold))
        ts = get_timeseries(str(bold), masks)
        
        # Decode SWITCH vs NOSWITCH
        patterns, labels = extract_patterns(str(bold), masks, events, 'Switching', ['SWITCH','NOSWITCH'], tr=tr)
        
        if len(labels) < 20:
            print(f" too few trials ({len(labels)})"); continue
        
        decoded, accs, classes = decode_networks(patterns, labels)
        mean_acc = np.mean(list(accs.values()))
        
        sw = compute_percond(decoded, labels, 'SWITCH', ts)
        ns = compute_percond(decoded, labels, 'NOSWITCH', ts)
        
        if sw and ns:
            switch_results.append({
                'sub': sub, 'sw_phi': sw['phi'], 'sw_lmax': sw['lmax'],
                'ns_phi': ns['phi'], 'ns_lmax': ns['lmax'], 'acc': mean_acc
            })
            d = sw['lmax'] - ns['lmax']
            print(f" SW={sw['lmax']:.4f} NS={ns['lmax']:.4f} Δ={d:+.4f} acc={mean_acc:.1%} {'✓' if d>0 else '✗'}")
        else:
            print(f" computation failed")
    
    if len(switch_results) >= 3:
        sw_lmax = [r['sw_lmax'] for r in switch_results]
        ns_lmax = [r['ns_lmax'] for r in switch_results]
        diff = np.array(sw_lmax) - np.array(ns_lmax)
        t, p = stats.ttest_rel(sw_lmax, ns_lmax)
        d_val = np.mean(diff)/(np.std(diff,ddof=1)+1e-10)
        n_pos = (diff>0).sum()
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        
        print(f"\n  GROUP RESULT (N={len(switch_results)}):")
        print(f"    SWITCH λ*_max:    {np.mean(sw_lmax):.6f} ± {np.std(sw_lmax,ddof=1):.6f}")
        print(f"    NOSWITCH λ*_max:  {np.mean(ns_lmax):.6f} ± {np.std(ns_lmax,ddof=1):.6f}")
        print(f"    Difference:       {np.mean(diff):+.6f}")
        print(f"    t({len(diff)-1}) = {t:.3f}, p = {p:.4f} {sig}")
        print(f"    Cohen's d = {d_val:+.3f}")
        print(f"    Direction: {n_pos}/{len(diff)} subjects show SWITCH > NOSWITCH")
    
    # =====================================================================
    # ANALYSIS 2: INCONGRUENT vs CONGRUENT
    # =====================================================================
    print(f"\n{'='*75}")
    print("ANALYSIS 2: INCONGRUENT vs CONGRUENT")
    print("="*75)
    
    cong_results = []
    
    for sub in subjects:
        bold = data_dir/sub/"func"/f"{sub}_task-taskswitch_bold.nii.gz"
        evts = data_dir/sub/"func"/f"{sub}_task-taskswitch_events.tsv"
        jpath = data_dir/sub/"func"/f"{sub}_task-taskswitch_bold.json"
        
        if not bold.exists(): continue
        print(f"\n  {sub}...", end="", flush=True)
        
        tr = 2.0
        if jpath.exists():
            with open(jpath) as f: tr = json.load(f).get('RepetitionTime', 2.0)
        
        events = pd.read_csv(evts, sep='\t')
        masks = get_masks(str(bold))
        ts = get_timeseries(str(bold), masks)
        
        # Decode INCONGRUENT vs CONGRUENT
        patterns, labels = extract_patterns(str(bold), masks, events, 'Congruency', ['INCONGRUENT','CONGRUENT'], tr=tr)
        
        if len(labels) < 20:
            print(f" too few trials ({len(labels)})"); continue
        
        decoded, accs, classes = decode_networks(patterns, labels)
        mean_acc = np.mean(list(accs.values()))
        
        inc = compute_percond(decoded, labels, 'INCONGRUENT', ts)
        con = compute_percond(decoded, labels, 'CONGRUENT', ts)
        
        if inc and con:
            cong_results.append({
                'sub': sub, 'inc_phi': inc['phi'], 'inc_lmax': inc['lmax'],
                'con_phi': con['phi'], 'con_lmax': con['lmax'], 'acc': mean_acc
            })
            d = inc['lmax'] - con['lmax']
            print(f" INC={inc['lmax']:.4f} CON={con['lmax']:.4f} Δ={d:+.4f} acc={mean_acc:.1%} {'✓' if d>0 else '✗'}")
        else:
            print(f" computation failed")
    
    if len(cong_results) >= 3:
        inc_lmax = [r['inc_lmax'] for r in cong_results]
        con_lmax = [r['con_lmax'] for r in cong_results]
        diff = np.array(inc_lmax) - np.array(con_lmax)
        t, p = stats.ttest_rel(inc_lmax, con_lmax)
        d_val = np.mean(diff)/(np.std(diff,ddof=1)+1e-10)
        n_pos = (diff>0).sum()
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        
        print(f"\n  GROUP RESULT (N={len(cong_results)}):")
        print(f"    INCONGRUENT λ*_max:  {np.mean(inc_lmax):.6f} ± {np.std(inc_lmax,ddof=1):.6f}")
        print(f"    CONGRUENT λ*_max:    {np.mean(con_lmax):.6f} ± {np.std(con_lmax,ddof=1):.6f}")
        print(f"    Difference:          {np.mean(diff):+.6f}")
        print(f"    t({len(diff)-1}) = {t:.3f}, p = {p:.4f} {sig}")
        print(f"    Cohen's d = {d_val:+.3f}")
        print(f"    Direction: {n_pos}/{len(diff)} subjects show INCONGRUENT > CONGRUENT")
    
    # =====================================================================
    # SCORECARD
    # =====================================================================
    print(f"\n{'='*75}")
    print("LEVEL 3 SCORECARD — TASK SWITCHING")
    print("="*75)
    print(f"""
  Combined with Stop Signal results:
  
  TEST                                    RESULT
  ──────────────────────────────────────────────────
  Stop Signal: STOP > GO                  p=0.014, d=0.96, 9/10 ★
  Stop Signal > SCAP cross-task           p=0.0001, d=2.22, 10/10 ★★★
  BART > SCAP cross-task                  p=0.0001, d=2.51, 9/9 ★★★
  Task Switch: SWITCH > NOSWITCH          {f'p={stats.ttest_rel(sw_lmax, ns_lmax)[1]:.4f}' if len(switch_results)>=3 else 'N/A'}
  Task Switch: INCONGRUENT > CONGRUENT    {f'p={stats.ttest_rel(inc_lmax, con_lmax)[1]:.4f}' if len(cong_results)>=3 else 'N/A'}
""")
    
    print("="*75)
    print("TASK SWITCHING ANALYSIS COMPLETE")
    print("="*75)

if __name__ == "__main__":
    main()
