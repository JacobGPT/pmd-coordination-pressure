#!/usr/bin/env python3
"""
===========================================================================
GPT FOLLOW-UP ANALYSES

TEST 1: ARBITRATION REGIME
  GPT noticed that Φ*_PD adds the most when baseline predictors fail.
  Test: split trials into low/high Φ*_PD, evaluate baseline R² in each.
  Prediction: baseline R² drops in high-Φ*_PD trials.

TEST 2: NEGATIVE-SLOPE SSRT CHECK
  GPT suggested negative-slope subjects might have faster SSRT
  (better inhibitory control), producing "rapid override" instead of
  "slow deliberation" under high coordination pressure.

TEST 3: CROSS-SUBJECT EIGENVECTOR SIMILARITY
  GPT suggested computing eigenvectors separately per subject and
  measuring cosine similarity between them. If the coordination axis
  is universal, similarity should be high across brains.

TEST 4: BASELINE R² vs ΔR² CORRELATION
  Formal test of GPT's observation: is there a negative correlation
  between baseline model strength and Φ*_PD contribution?
===========================================================================
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# Same pipeline utilities (abbreviated — imports same functions)
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


def compute_J_matrix(decoded_probs, go_indices, timeseries, tau=0.05):
    """Compute mean J matrix for given GO trial indices."""
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
    
    J = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            jsds = []
            for t in go_indices:
                pi = decoded_probs[i][t]+1e-10; pj = decoded_probs[j][t]+1e-10
                pi /= pi.sum(); pj /= pj.sum(); m_ = 0.5*(pi+pj)
                jsds.append(max(0.5*np.sum(pi*np.log2(pi/m_))+0.5*np.sum(pj*np.log2(pj/m_)),0))
            mean_jsd = np.mean(jsds) if jsds else 0
            g_c = soft_gate(TE[i,j], i_min, tau)
            g_r = np.sqrt(max(coh[i],0)*max(coh[j],0))
            g_d = np.sqrt(net_conf.get(i,0)*net_conf.get(j,0))
            J[i,j] = J[j,i] = TE[i,j] * mean_jsd * g_c * g_r * g_d
    np.fill_diagonal(J, 0)
    return J


def main():
    data_dir = Path("./ds000030")
    all_subjects = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    
    print("="*80)
    print("GPT FOLLOW-UP ANALYSES")
    print("="*80)
    
    # Collect data
    all_data = []
    subject_eigvecs = []
    
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
            if len(labels) < 20: continue
            decoded, accs, _ = decode_networks(patterns, labels)
            
            go_mask = labels == 'GO'
            go_indices = np.where(go_mask)[0]
            N = 7
            
            # Session-level quantities
            TE = np.zeros((N,N))
            for i in range(N):
                for j in range(i+1,N):
                    te = sym_te(ts[:,i], ts[:,j])
                    TE[i,j] = TE[j,i] = te
            coh = np.array([signal_coherence(ts[:,i]) for i in range(N)])
            te_vals = [TE[i,j] for i in range(N) for j in range(i+1,N)]
            i_min = np.percentile(te_vals, 25) if te_vals else 0
            net_conf = {}
            for ni in decoded:
                pr = decoded[ni]; nc = pr.shape[1]
                ent = -np.sum(pr * np.log2(pr+1e-10), axis=1)
                me = np.log2(nc)
                net_conf[ni] = max(1.0 - np.mean(ent)/me, 0) if me > 0 else 0
            
            # Per-trial metrics
            from nilearn import image
            bold_data = image.load_img(str(bold)).get_fdata()
            n_vol = bold_data.shape[-1]; bold_2d = bold_data.reshape(-1, n_vol)
            hrf_delay = 5.0
            
            trials = []
            for ti, trial_idx in enumerate(go_indices):
                # Φ*_PD
                J = np.zeros((N,N))
                for i in range(N):
                    for j in range(i+1,N):
                        pi = decoded[i][trial_idx]+1e-10; pj = decoded[j][trial_idx]+1e-10
                        pi /= pi.sum(); pj /= pj.sum(); m_ = 0.5*(pi+pj)
                        jsd = max(0.5*np.sum(pi*np.log2(pi/m_))+0.5*np.sum(pj*np.log2(pj/m_)),0)
                        g_c = soft_gate(TE[i,j], i_min, 0.05)
                        g_r = np.sqrt(max(coh[i],0)*max(coh[j],0))
                        g_d = np.sqrt(net_conf.get(i,0)*net_conf.get(j,0))
                        J[i,j] = J[j,i] = TE[i,j] * jsd * g_c * g_r * g_d
                np.fill_diagonal(J, 0)
                n_pairs = N*(N-1)//2
                phi = sum(J[i,j] for i in range(N) for j in range(i+1,N))/n_pairs
                
                # Baseline predictors
                entropies = []
                for ni in decoded:
                    pr = decoded[ni][trial_idx]+1e-10; pr = pr/pr.sum()
                    entropies.append(-np.sum(pr*np.log2(pr)))
                
                all_probs = []
                for ni in decoded: all_probs.extend(decoded[ni][trial_idx].tolist())
                
                onset = events_filtered.iloc[trial_idx]['onset'] if trial_idx < len(events_filtered) else 0
                vol = min(int((onset + hrf_delay)/tr), n_vol-1)
                
                net_acts = []
                for mi, mask in enumerate(masks):
                    mf = mask.ravel()
                    if mf.sum() > 10 and vol < n_vol:
                        net_acts.append(bold_2d[mf, vol].mean())
                
                trials.append({
                    'phi': phi,
                    'entropy': np.mean(entropies),
                    'prob_var': np.var(all_probs),
                    'act_mean': np.mean(net_acts) if net_acts else 0,
                    'act_var': np.var(net_acts) if len(net_acts) > 1 else 0,
                    'rt': pd.to_numeric(events_filtered.iloc[go_indices[ti]]['ReactionTime'] if ti < len(go_indices) else np.nan, errors='coerce')
                })
            
            df = pd.DataFrame(trials)
            df = df.dropna(subset=['rt'])
            df = df[(df['rt'] > 0.1) & (df['rt'] < 2.0)]
            if len(df) < 15: continue
            
            # Compute SSRT estimate
            stop_indices = np.where(labels == 'STOP')[0]
            go_rts = df['rt'].values
            go_rts_sorted = np.sort(go_rts)
            n_stop = len(stop_indices)
            n_go = len(go_rts)
            stop_accuracy = 0.5  # default
            if n_stop > 0:
                # Get SSD from events
                ssds = []
                for si in stop_indices:
                    if si < len(events_filtered):
                        ssd_val = events_filtered.iloc[si].get('StopSignalDelay', np.nan)
                        if not pd.isna(ssd_val):
                            ssd = pd.to_numeric(ssd_val, errors='coerce')
                            if not np.isnan(ssd): ssds.append(ssd)
                mean_ssd = np.mean(ssds) if ssds else 0.25
                # Simple SSRT estimate: nth RT - mean SSD
                p_respond = 0.5  # approximate
                nth = max(1, min(int(p_respond * n_go), n_go-1))
                ssrt = go_rts_sorted[nth] - mean_ssd
            else:
                ssrt = np.nan
            
            # Phi-RT slope
            r_phi_rt = stats.pearsonr(df['phi'].values, df['rt'].values)[0] if df['phi'].std() > 0 else 0
            
            # J matrix for eigenvector
            J_full = compute_J_matrix(decoded, go_indices, ts)
            Js = np.nan_to_num((J_full + J_full.T)/2)
            evals, evecs = np.linalg.eigh(Js)
            eigvec = np.abs(evecs[:, np.argmax(evals)])
            subject_eigvecs.append(eigvec)
            
            all_data.append({
                'subject': sub, 'df': df, 'r_phi_rt': r_phi_rt,
                'ssrt': ssrt, 'eigvec': eigvec
            })
            print(f" N={len(df)} r={r_phi_rt:+.3f} SSRT={ssrt:.3f}" if not np.isnan(ssrt) else f" N={len(df)} r={r_phi_rt:+.3f}", end="")
            
        except Exception as e:
            print(f" ERROR: {e}", end="")
    
    N_subs = len(all_data)
    print(f"\n\n  Processed {N_subs} subjects")
    if N_subs < 4: return
    
    # =====================================================================
    # TEST 1: ARBITRATION REGIME
    # =====================================================================
    print(f"\n{'='*80}")
    print("TEST 1: ARBITRATION REGIME")
    print("Do baseline predictors fail when Φ*_PD is high?")
    print("="*80)
    
    baseline_cols = ['entropy', 'prob_var', 'act_mean', 'act_var']
    
    r2_low_list = []
    r2_high_list = []
    
    for d in all_data:
        df = d['df'].copy()
        if df['phi'].std() < 1e-15: continue
        
        median_phi = df['phi'].median()
        low = df[df['phi'] <= median_phi]
        high = df[df['phi'] > median_phi]
        
        if len(low) < 10 or len(high) < 10: continue
        
        for split_name, split_df, r2_list in [("low", low, r2_low_list), ("high", high, r2_high_list)]:
            X = split_df[baseline_cols].values
            y = split_df['rt'].values
            if y.std() < 1e-15: r2_list.append(0); continue
            model = LinearRegression().fit(X, y)
            pred = model.predict(X)
            ss_res = np.sum((y - pred)**2)
            ss_tot = np.sum((y - y.mean())**2)
            r2 = max(1 - ss_res/ss_tot, 0) if ss_tot > 0 else 0
            r2_list.append(r2)
    
    if r2_low_list and r2_high_list:
        mean_low = np.mean(r2_low_list)
        mean_high = np.mean(r2_high_list)
        t_regime, p_regime = stats.ttest_rel(r2_low_list, r2_high_list)
        
        print(f"\n  Baseline R² on LOW Φ*_PD trials:  {mean_low:.4f}")
        print(f"  Baseline R² on HIGH Φ*_PD trials: {mean_high:.4f}")
        print(f"  Paired t-test: t = {t_regime:.3f}, p = {p_regime:.4f}")
        
        if mean_low > mean_high and p_regime < 0.05:
            print(f"\n  ★ CONFIRMED: Baseline predictors fail when Φ*_PD is high.")
            print(f"    Φ*_PD captures arbitration-specific information.")
        elif mean_low > mean_high:
            print(f"\n  Direction correct (baseline weaker at high Φ*_PD) but not significant.")
        else:
            print(f"\n  No clear arbitration regime pattern at this N.")
    
    # =====================================================================
    # TEST 2: BASELINE R² vs ΔR² CORRELATION
    # =====================================================================
    print(f"\n{'='*80}")
    print("TEST 2: BASELINE R² vs ΔR² (GPT's observation)")
    print("Does Φ*_PD add more when baseline is weak?")
    print("="*80)
    
    # Recompute from stored data
    r2_bases = []
    delta_r2s = []
    for d in all_data:
        df = d['df'].copy()
        y = df['rt'].values
        if y.std() < 1e-15: continue
        X_base = df[baseline_cols].values
        X_full = df[baseline_cols + ['phi']].values
        
        ss_tot = np.sum((y - y.mean())**2)
        pred_b = LinearRegression().fit(X_base, y).predict(X_base)
        r2_b = max(1 - np.sum((y - pred_b)**2)/ss_tot, 0)
        pred_f = LinearRegression().fit(X_full, y).predict(X_full)
        r2_f = max(1 - np.sum((y - pred_f)**2)/ss_tot, 0)
        
        r2_bases.append(r2_b)
        delta_r2s.append(max(r2_f - r2_b, 0))
    
    if len(r2_bases) >= 5:
        r_obs, p_obs = stats.pearsonr(r2_bases, delta_r2s)
        print(f"\n  Correlation: baseline_R² ~ ΔR²")
        print(f"    r = {r_obs:+.3f}, p = {p_obs:.4f}")
        if r_obs < -0.3 and p_obs < 0.05:
            print(f"\n  ★ CONFIRMED: Φ*_PD adds most when baseline fails.")
            print(f"    This is the signature of an arbitration-specific signal.")
        elif r_obs < 0:
            print(f"\n  Direction correct (negative correlation) but {'significant' if p_obs < 0.05 else 'not significant'}.")
    
    # =====================================================================
    # TEST 3: NEGATIVE-SLOPE SSRT CHECK
    # =====================================================================
    print(f"\n{'='*80}")
    print("TEST 3: NEGATIVE-SLOPE SSRT CHECK")
    print("Do negative-slope subjects have better inhibitory control?")
    print("="*80)
    
    slopes = [d['r_phi_rt'] for d in all_data]
    ssrts = [d['ssrt'] for d in all_data]
    valid = [(s, ssrt) for s, ssrt in zip(slopes, ssrts) if not np.isnan(ssrt)]
    
    if len(valid) >= 5:
        slopes_v = [v[0] for v in valid]
        ssrts_v = [v[1] for v in valid]
        r_ssrt, p_ssrt = stats.pearsonr(slopes_v, ssrts_v)
        
        print(f"\n  Correlation: Φ-RT_slope ~ SSRT")
        print(f"    r = {r_ssrt:+.3f}, p = {p_ssrt:.4f}")
        print(f"    (Positive r = subjects with positive slopes have SLOWER SSRT)")
        print(f"    (Negative r = subjects with positive slopes have FASTER SSRT)")
        
        if r_ssrt > 0.3 and p_ssrt < 0.1:
            print(f"\n  Subjects with positive Φ-RT slopes have slower SSRT (poorer inhibition).")
            print(f"  Negative-slope subjects may resolve conflicts via rapid override.")
        elif r_ssrt < -0.3:
            print(f"\n  Opposite pattern: positive slopes go with faster SSRT.")
    else:
        print(f"\n  Too few valid SSRT estimates ({len(valid)})")
    
    # =====================================================================
    # TEST 4: CROSS-SUBJECT EIGENVECTOR SIMILARITY
    # =====================================================================
    print(f"\n{'='*80}")
    print("TEST 4: CROSS-SUBJECT EIGENVECTOR SIMILARITY")
    print("Is the coordination axis universal across brains?")
    print("="*80)
    
    if len(subject_eigvecs) >= 5:
        # Compute all pairwise cosine similarities
        n_ev = len(subject_eigvecs)
        sims = []
        for i in range(n_ev):
            for j in range(i+1, n_ev):
                cos = np.dot(subject_eigvecs[i], subject_eigvecs[j]) / (
                    np.linalg.norm(subject_eigvecs[i]) * np.linalg.norm(subject_eigvecs[j]))
                sims.append(cos)
        
        sims = np.array(sims)
        null_cos = 1.0 / np.sqrt(7)  # expected for random unit vectors
        
        print(f"\n  Pairwise cosine similarity between subject eigenvectors:")
        print(f"    Mean = {np.mean(sims):.4f}")
        print(f"    Min  = {np.min(sims):.4f}")
        print(f"    Max  = {np.max(sims):.4f}")
        print(f"    Random null = {null_cos:.4f}")
        
        t_sim, p_sim = stats.ttest_1samp(sims, null_cos)
        print(f"    t = {t_sim:.3f}, p = {p_sim:.6f}")
        
        if np.mean(sims) > 0.9:
            print(f"\n  ★ CONFIRMED: The coordination axis is nearly identical across brains.")
            print(f"    Different brains operate along the same principal coordination mode.")
        elif np.mean(sims) > 0.7:
            print(f"\n  Moderate similarity: coordination axes are related but not identical.")
        elif np.mean(sims) > null_cos and p_sim < 0.05:
            print(f"\n  Significant above chance but modest: partial universality.")
        else:
            print(f"\n  Eigenvectors are not consistent across brains.")
        
        # Also compute mean eigenvector
        mean_ev = np.mean(subject_eigvecs, axis=0)
        mean_ev /= np.linalg.norm(mean_ev)
        
        network_names = ['Visual', 'Somatomotor', 'DorsalAtt', 'VentralAtt', 
                         'Limbic', 'FrontoParietal', 'Default']
        
        print(f"\n  Mean eigenvector (coordination axis weights):")
        sorted_idx = np.argsort(-mean_ev)
        for idx in sorted_idx:
            bar = '█' * int(mean_ev[idx] * 50)
            print(f"    {network_names[idx]:<15} {mean_ev[idx]:.4f} {bar}")
    
    # =====================================================================
    # OVERALL
    # =====================================================================
    print(f"\n{'='*80}")
    print("GPT FOLLOW-UP — OVERALL RESULTS")
    print("="*80)
    print(f"""
  Arbitration regime:        {'CONFIRMED' if r2_low_list and np.mean(r2_low_list) > np.mean(r2_high_list) and p_regime < 0.05 else 'Direction correct' if r2_low_list and np.mean(r2_low_list) > np.mean(r2_high_list) else 'Not confirmed'}
  Baseline R² vs ΔR²:       r = {r_obs:+.3f}, p = {p_obs:.4f}
  Cross-subject eigenvector: Mean similarity = {np.mean(sims):.4f}
""")
    
    print("="*80)
    print("GPT FOLLOW-UP ANALYSES COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
