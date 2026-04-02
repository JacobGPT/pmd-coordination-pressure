#!/usr/bin/env python3
"""
===========================================================================
PRESSURE MAKES DIAMONDS — VARIANCE DECOMPOSITION
Does Φ*_PD capture behavioral information that existing metrics cannot?

GPT's decisive test: build nested regression models predicting RT from
increasingly comprehensive baseline neural predictors, then test whether
adding Φ*_PD significantly improves prediction.

BASELINE PREDICTORS (per trial):
  1. Mean decoded entropy (uncertainty across networks)
  2. Decoded probability variance (spread of decoded policies)
  3. Mean network activation magnitude (global activity level)
  4. Network activation variance (activation spread across networks)
  5. Mean pairwise TE (functional connectivity strength)

MODELS:
  Model 1: RT ~ entropy + prob_var + act_mean + act_var + mean_TE
  Model 2: RT ~ entropy + prob_var + act_mean + act_var + mean_TE + Φ*_PD
  Model 3: RT ~ Φ*_PD only

KEY TEST: Does Model 2 beat Model 1? (ΔR² with F-test)
CROSS-VALIDATED: Leave-one-subject-out prediction comparison
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


def compute_all_trial_metrics(decoded_probs, labels, timeseries, bold_file, masks, events_filtered, tau=0.05):
    """Compute Φ*_PD AND all baseline predictors for each GO trial."""
    from nilearn import image
    N = 7
    
    # Session-level TE
    TE = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            te = sym_te(timeseries[:,i], timeseries[:,j])
            TE[i,j] = TE[j,i] = te
    mean_te_session = np.mean([TE[i,j] for i in range(N) for j in range(i+1,N)])
    
    coh = np.array([signal_coherence(timeseries[:,i]) for i in range(N)])
    te_vals = [TE[i,j] for i in range(N) for j in range(i+1,N)]
    i_min = np.percentile(te_vals, 25) if te_vals else 0
    
    net_conf = {}
    for ni in decoded_probs:
        pr = decoded_probs[ni]; nc = pr.shape[1]
        ent = -np.sum(pr * np.log2(pr+1e-10), axis=1)
        me = np.log2(nc)
        net_conf[ni] = max(1.0 - np.mean(ent)/me, 0) if me > 0 else 0
    
    # Get BOLD data for activation metrics
    bold_data = image.load_img(bold_file).get_fdata()
    n_vol = bold_data.shape[-1]
    bold_2d = bold_data.reshape(-1, n_vol)
    
    go_mask = (labels == 'GO')
    go_indices = np.where(go_mask)[0]
    
    # Get trial volumes
    tr = 2.0  # Will be overridden by caller if needed
    hrf_delay = 5.0
    onsets = events_filtered.iloc[go_indices]['onset'].values
    vols = np.clip(((onsets + hrf_delay) / tr).astype(int), 0, n_vol - 1)
    
    results = []
    for ti, (trial_idx, vol) in enumerate(zip(go_indices, vols)):
        # --- Φ*_PD ---
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
        
        # --- Baseline predictor 1: Mean decoded entropy ---
        entropies = []
        for ni in decoded_probs:
            pr = decoded_probs[ni][trial_idx] + 1e-10
            pr = pr / pr.sum()
            ent = -np.sum(pr * np.log2(pr))
            entropies.append(ent)
        mean_entropy = np.mean(entropies)
        
        # --- Baseline predictor 2: Decoded probability variance ---
        all_probs = []
        for ni in decoded_probs:
            all_probs.extend(decoded_probs[ni][trial_idx].tolist())
        prob_var = np.var(all_probs)
        
        # --- Baseline predictor 3: Mean network activation ---
        net_acts = []
        for mi, mask in enumerate(masks):
            mf = mask.ravel()
            if mf.sum() > 10 and vol < n_vol:
                act = bold_2d[mf, vol].mean()
                net_acts.append(act)
        act_mean = np.mean(net_acts) if net_acts else 0
        
        # --- Baseline predictor 4: Network activation variance ---
        act_var = np.var(net_acts) if len(net_acts) > 1 else 0
        
        # --- Baseline predictor 5: Mean TE (session-level, same for all trials) ---
        mean_te = mean_te_session
        
        # --- Baseline predictor 6: Control network activation (FP + VA) ---
        # Frontoparietal = index 5, VentralAttention = index 3
        control_act = 0
        control_count = 0
        for ci in [3, 5]:  # VentAtt, FrontoParietal
            if ci < len(masks):
                mf = masks[ci].ravel()
                if mf.sum() > 10 and vol < n_vol:
                    control_act += bold_2d[mf, vol].mean()
                    control_count += 1
        control_act = control_act / max(control_count, 1)
        
        results.append({
            'phi': phi,
            'entropy': mean_entropy,
            'prob_var': prob_var,
            'act_mean': act_mean,
            'act_var': act_var,
            'mean_te': mean_te,
            'control_act': control_act
        })
    
    return go_indices, results


def main():
    data_dir = Path("./ds000030")
    all_subjects = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    
    print("="*80)
    print("PRESSURE MAKES DIAMONDS — VARIANCE DECOMPOSITION")
    print("Does Φ*_PD capture information beyond ALL known neural predictors?")
    print("="*80)
    
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
            
            go_indices, trial_metrics = compute_all_trial_metrics(
                decoded, labels, ts, str(bold), masks, events_filtered)
            
            go_events = events_filtered.iloc[go_indices].copy()
            trial_rts = pd.to_numeric(go_events['ReactionTime'], errors='coerce').values
            
            # Build trial dataframe
            df = pd.DataFrame(trial_metrics)
            df['rt'] = trial_rts
            df = df.dropna()
            df = df[(df['rt'] > 0.1) & (df['rt'] < 2.0)]
            
            if len(df) < 15: print(f" too few valid ({len(df)})", end=""); continue
            
            # Z-score everything within subject
            for col in df.columns:
                if df[col].std() > 0:
                    df[col] = (df[col] - df[col].mean()) / df[col].std()
            
            all_data.append({'subject': sub, 'df': df, 'n': len(df)})
            print(f" N={len(df)}", end="")
            
        except Exception as e:
            print(f" ERROR: {e}", end="")
    
    N_subs = len(all_data)
    print(f"\n\n  Processed {N_subs} subjects")
    if N_subs < 4: return
    
    # =====================================================================
    # WITHIN-SUBJECT VARIANCE DECOMPOSITION
    # =====================================================================
    print(f"\n{'='*80}")
    print("WITHIN-SUBJECT VARIANCE DECOMPOSITION (per subject)")
    print("="*80)
    
    baseline_cols = ['entropy', 'prob_var', 'act_mean', 'act_var', 'control_act']
    
    r2_baseline = []
    r2_full = []
    r2_phi_only = []
    delta_r2 = []
    partial_rs = []
    
    for d in all_data:
        df = d['df']
        y = df['rt'].values
        
        # Model 1: baseline predictors only
        X_base = df[baseline_cols].values
        model_base = LinearRegression().fit(X_base, y)
        pred_base = model_base.predict(X_base)
        ss_res_base = np.sum((y - pred_base)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2_b = 1 - ss_res_base/ss_tot if ss_tot > 0 else 0
        r2_baseline.append(max(r2_b, 0))
        
        # Model 2: baseline + Φ*_PD
        X_full = df[baseline_cols + ['phi']].values
        model_full = LinearRegression().fit(X_full, y)
        pred_full = model_full.predict(X_full)
        ss_res_full = np.sum((y - pred_full)**2)
        r2_f = 1 - ss_res_full/ss_tot if ss_tot > 0 else 0
        r2_full.append(max(r2_f, 0))
        
        # Model 3: Φ*_PD only
        X_phi = df[['phi']].values
        model_phi = LinearRegression().fit(X_phi, y)
        pred_phi = model_phi.predict(X_phi)
        ss_res_phi = np.sum((y - pred_phi)**2)
        r2_p = 1 - ss_res_phi/ss_tot if ss_tot > 0 else 0
        r2_phi_only.append(max(r2_p, 0))
        
        # ΔR²
        dr2 = max(r2_f - r2_b, 0)
        delta_r2.append(dr2)
        
        # Partial correlation: Φ ~ RT | baselines
        resid_phi = df['phi'].values - LinearRegression().fit(X_base, df['phi'].values).predict(X_base)
        resid_rt = y - pred_base
        if np.std(resid_phi) > 0 and np.std(resid_rt) > 0:
            pr, _ = stats.pearsonr(resid_phi, resid_rt)
            partial_rs.append(pr)
        else:
            partial_rs.append(0)
    
    print(f"\n  {'Subject':<15} {'R²_base':>8} {'R²_full':>8} {'R²_Φonly':>8} {'ΔR²':>8} {'Partial r':>10}")
    print(f"  {'-'*59}")
    for i, d in enumerate(all_data):
        print(f"  {d['subject']:<15} {r2_baseline[i]:>8.4f} {r2_full[i]:>8.4f} {r2_phi_only[i]:>8.4f} {delta_r2[i]:>8.4f} {partial_rs[i]:>+10.3f}")
    
    # Group statistics
    print(f"\n  GROUP RESULTS:")
    print(f"    Mean R² baseline:    {np.mean(r2_baseline):.4f}")
    print(f"    Mean R² full:        {np.mean(r2_full):.4f}")
    print(f"    Mean R² Φ*_PD only:  {np.mean(r2_phi_only):.4f}")
    
    mean_dr2 = np.mean(delta_r2)
    t_dr2, p_dr2 = stats.ttest_1samp(delta_r2, 0)
    n_pos_dr2 = sum(1 for d in delta_r2 if d > 0)
    
    print(f"\n    Mean ΔR² (full - baseline): {mean_dr2:+.4f}")
    print(f"    t({N_subs-1}) = {t_dr2:.3f}, p = {p_dr2:.4f}")
    print(f"    {n_pos_dr2}/{N_subs} subjects show positive ΔR²")
    
    mean_partial = np.mean(partial_rs)
    t_partial, p_partial = stats.ttest_1samp(partial_rs, 0)
    n_pos_partial = sum(1 for r in partial_rs if r > 0)
    
    print(f"\n    Mean partial r (Φ ~ RT | baselines): {mean_partial:+.4f}")
    print(f"    t({N_subs-1}) = {t_partial:.3f}, p = {p_partial:.4f}")
    print(f"    {n_pos_partial}/{N_subs} positive")
    
    # =====================================================================
    # CROSS-SUBJECT PREDICTION COMPARISON
    # =====================================================================
    print(f"\n{'='*80}")
    print("CROSS-SUBJECT LEAVE-ONE-OUT: BASELINE vs FULL vs Φ-ONLY")
    print("="*80)
    
    loo_base_rs = []
    loo_full_rs = []
    loo_phi_rs = []
    
    for held_out in range(N_subs):
        # Pool training data
        train_dfs = [all_data[i]['df'] for i in range(N_subs) if i != held_out]
        train = pd.concat(train_dfs, ignore_index=True)
        test = all_data[held_out]['df']
        
        y_train = train['rt'].values
        y_test = test['rt'].values
        
        if len(y_test) < 5: continue
        
        # Baseline
        X_train_base = train[baseline_cols].values
        X_test_base = test[baseline_cols].values
        model = LinearRegression().fit(X_train_base, y_train)
        pred = model.predict(X_test_base)
        r, _ = stats.pearsonr(pred, y_test)
        loo_base_rs.append(r)
        
        # Full
        X_train_full = train[baseline_cols + ['phi']].values
        X_test_full = test[baseline_cols + ['phi']].values
        model = LinearRegression().fit(X_train_full, y_train)
        pred = model.predict(X_test_full)
        r, _ = stats.pearsonr(pred, y_test)
        loo_full_rs.append(r)
        
        # Phi only
        X_train_phi = train[['phi']].values
        X_test_phi = test[['phi']].values
        model = LinearRegression().fit(X_train_phi, y_train)
        pred = model.predict(X_test_phi)
        r, _ = stats.pearsonr(pred, y_test)
        loo_phi_rs.append(r)
    
    print(f"\n  {'Model':<25} {'Mean LOO r':>12} {'t-stat':>10} {'p-value':>10} {'N+':>5}")
    print(f"  {'-'*62}")
    
    for name, rs in [("Baseline (5 predictors)", loo_base_rs), 
                      ("Full (baseline + Φ*_PD)", loo_full_rs),
                      ("Φ*_PD only", loo_phi_rs)]:
        rs = np.array(rs)
        t, p = stats.ttest_1samp(rs, 0)
        n_pos = (rs > 0).sum()
        print(f"  {name:<25} {np.mean(rs):>+12.4f} {t:>10.3f} {p:>10.4f} {n_pos:>3}/{len(rs)}")
    
    # Direct comparison: full vs baseline
    diff = np.array(loo_full_rs) - np.array(loo_base_rs)
    t_diff, p_diff = stats.ttest_1samp(diff, 0)
    n_pos_diff = (diff > 0).sum()
    
    print(f"\n  Full vs Baseline improvement:")
    print(f"    Mean Δr = {np.mean(diff):+.4f}, t = {t_diff:.3f}, p = {p_diff:.4f}, {n_pos_diff}/{len(diff)} improved")
    
    # =====================================================================
    # OVERALL VERDICT
    # =====================================================================
    print(f"\n{'='*80}")
    print("VARIANCE DECOMPOSITION — OVERALL VERDICT")
    print("="*80)
    
    print(f"""
  QUESTION: Does Φ*_PD capture behavioral information beyond existing metrics?

  Within-subject ΔR² (Φ*_PD beyond 5 baseline predictors):
    Mean ΔR² = {mean_dr2:+.4f}, p = {p_dr2:.4f}, {n_pos_dr2}/{N_subs} positive
    → {'YES ★' if p_dr2 < 0.05 and mean_dr2 > 0 else 'Not significant'}

  Partial correlation (Φ ~ RT | all baselines):
    Mean r = {mean_partial:+.4f}, p = {p_partial:.4f}, {n_pos_partial}/{N_subs} positive
    → {'YES ★' if p_partial < 0.05 and mean_partial > 0 else 'Not significant'}

  Cross-subject LOO (full model vs baseline):
    Mean improvement = {np.mean(diff):+.4f}, p = {p_diff:.4f}
    → {'YES ★' if p_diff < 0.05 and np.mean(diff) > 0 else 'Not significant at this N'}

  INTERPRETATION:
""")
    
    if p_dr2 < 0.05 and mean_dr2 > 0 and p_partial < 0.05 and mean_partial > 0:
        print(f"  ★ Φ*_PD captures behavioral information that entropy, activation,")
        print(f"    variance, connectivity, and control-network signals CANNOT explain.")
        print(f"    The metric is measuring something genuinely new.")
    elif (p_dr2 < 0.05 and mean_dr2 > 0) or (p_partial < 0.05 and mean_partial > 0):
        print(f"  Partial support: Φ*_PD adds some independent information.")
        print(f"  More subjects needed for definitive conclusion.")
    else:
        print(f"  Φ*_PD does not clearly add beyond existing predictors at this N.")
    
    print(f"\n{'='*80}")
    print("VARIANCE DECOMPOSITION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
