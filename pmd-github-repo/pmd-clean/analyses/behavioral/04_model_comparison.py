#!/usr/bin/env python3
"""
===========================================================================
PRESSURE MAKES DIAMONDS — MODEL COMPARISON & BINNED VISUALIZATION
1. Does Φ*_PD predict RT beyond simpler neural metrics?
2. Binned visualization: RT as a function of Φ*_PD quintiles
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


# =========================================================================
# COMPUTE ALL TRIAL-LEVEL METRICS
# =========================================================================

def compute_trial_metrics(decoded_probs, timeseries, labels, tau=0.05):
    """
    Compute per-trial: Φ*_PD, λ*_max, mean_entropy, mean_variance, mean_connectivity
    Returns arrays of length n_trials.
    """
    N = 7
    n_trials = decoded_probs[0].shape[0]
    
    # TE matrix (whole scan)
    TE = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            te = sym_te(timeseries[:,i], timeseries[:,j])
            TE[i,j] = TE[j,i] = te
    
    coh = np.array([signal_coherence(timeseries[:,i]) for i in range(N)])
    te_vals = [TE[i,j] for i in range(N) for j in range(i+1,N)]
    i_min = np.percentile(te_vals, 25)
    
    # Decode confidence
    net_conf = {}
    for ni in decoded_probs:
        pr = decoded_probs[ni]; nc = pr.shape[1]
        ent = -np.sum(pr * np.log2(pr+1e-10), axis=1)
        me = np.log2(nc)
        net_conf[ni] = max(1.0 - np.mean(ent)/me, 0) if me > 0 else 0
    
    # Precompute gates
    gates = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            g_c = soft_gate(TE[i,j], i_min, tau)
            g_r = np.sqrt(max(coh[i],0) * max(coh[j],0))
            g_d = np.sqrt(net_conf[i] * net_conf[j])
            gates[i,j] = gates[j,i] = TE[i,j] * g_c * g_r * g_d
    
    # Mean connectivity (scalar per scan, same for all trials)
    mean_conn = np.mean(te_vals)
    
    # Per-trial metrics
    phi_trials = np.zeros(n_trials)
    lmax_trials = np.zeros(n_trials)
    entropy_trials = np.zeros(n_trials)
    variance_trials = np.zeros(n_trials)
    
    for t in range(n_trials):
        J = np.zeros((N, N))
        trial_entropies = []
        
        for i in range(N):
            # Per-network entropy on this trial
            pi = decoded_probs[i][t] + 1e-10
            pi = pi / pi.sum()
            h = -np.sum(pi * np.log2(pi))
            trial_entropies.append(h)
            
            for j in range(i+1, N):
                pj = decoded_probs[j][t] + 1e-10
                pj = pj / pj.sum()
                m = 0.5 * (pi + pj)
                jsd = max(0.5*np.sum(pi*np.log2(pi/m)) + 0.5*np.sum(pj*np.log2(pj/m)), 0)
                J[i,j] = J[j,i] = gates[i,j] * jsd
        
        np.fill_diagonal(J, 0)
        n_pairs = N*(N-1)//2
        phi_trials[t] = sum(J[i,j] for i in range(N) for j in range(i+1,N)) / n_pairs
        lmax_trials[t] = compute_lmax(J)
        entropy_trials[t] = np.mean(trial_entropies)
    
    # Variance: use the decoded probability variance across networks per trial
    for t in range(n_trials):
        probs_all = np.array([decoded_probs[i][t] for i in range(N)])  # (7, n_classes)
        # Variance of the "GO probability" across networks
        variance_trials[t] = np.var(probs_all[:, 0])  # variance of P(GO) across networks
    
    return {
        'phi': phi_trials,
        'lmax': lmax_trials,
        'entropy': entropy_trials,
        'variance': variance_trials,
        'connectivity': np.full(n_trials, mean_conn),
    }


# =========================================================================
# MAIN
# =========================================================================

def main():
    data_dir = Path("./ds000030")
    all_subs = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    
    print("="*75)
    print("PRESSURE MAKES DIAMONDS — MODEL COMPARISON & BINNED VISUALIZATION")
    print("="*75)
    print()
    print("  PART 1: Does Φ*_PD predict RT beyond simpler metrics?")
    print("  PART 2: Binned visualization — RT by Φ*_PD quintile")
    print("="*75)
    
    # Collect all trial data across subjects
    all_subject_bins = {1:[], 2:[], 3:[], 4:[], 5:[]}  # quintile -> RT values
    
    # Per-subject regression results
    model_results = []
    
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
            
            # Decode GO vs STOP on all trials
            patterns, labels, trial_df = extract_patterns(str(bold), masks, events, 'trial_type', ['GO','STOP'], tr=tr)
            
            if len(labels) < 20: print(" too few trials"); continue
            
            decoded, accs, classes = decode_networks(patterns, labels)
            
            # Compute all trial-level metrics
            metrics = compute_trial_metrics(decoded, ts, labels)
            
            # Extract GO trials with valid RT
            go_mask = (labels == 'GO')
            go_df = trial_df[trial_df['trial_type']=='GO'].reset_index(drop=True)
            go_rt = go_df['ReactionTime'].values.astype(float)
            valid = go_mask.copy()
            valid_indices = np.where(go_mask)[0]
            
            # Filter for valid RTs among GO trials
            rt_valid = []
            phi_valid = []
            ent_valid = []
            var_valid = []
            conn_valid = []
            lmax_valid = []
            
            go_count = 0
            for idx in range(len(labels)):
                if labels[idx] == 'GO':
                    rt = go_rt[go_count]
                    go_count += 1
                    if not np.isnan(rt) and 0.1 < rt < 3.0:
                        rt_valid.append(rt)
                        phi_valid.append(metrics['phi'][idx])
                        ent_valid.append(metrics['entropy'][idx])
                        var_valid.append(metrics['variance'][idx])
                        conn_valid.append(metrics['connectivity'][idx])
                        lmax_valid.append(metrics['lmax'][idx])
            
            rt_valid = np.array(rt_valid)
            phi_valid = np.array(phi_valid)
            ent_valid = np.array(ent_valid)
            var_valid = np.array(var_valid)
            lmax_valid = np.array(lmax_valid)
            
            if len(rt_valid) < 20: print(" too few valid"); continue
            
            # =========================================================
            # PART 1: Model comparison per subject
            # =========================================================
            
            # Simple correlations with RT
            def safe_corr(x, y):
                if np.std(x) < 1e-12 or np.std(y) < 1e-12: return 0.0, 1.0
                return stats.pearsonr(x, y)
            
            r_phi, p_phi = safe_corr(phi_valid, rt_valid)
            r_ent, p_ent = safe_corr(ent_valid, rt_valid)
            r_var, p_var = safe_corr(var_valid, rt_valid)
            
            # Incremental R² — does Φ*_PD add to entropy + variance?
            from numpy.linalg import lstsq
            
            n = len(rt_valid)
            
            # Baseline model: entropy + variance
            X_base = np.column_stack([ent_valid, var_valid, np.ones(n)])
            beta_base, _, _, _ = lstsq(X_base, rt_valid, rcond=None)
            pred_base = X_base @ beta_base
            ss_res_base = np.sum((rt_valid - pred_base)**2)
            ss_tot = np.sum((rt_valid - rt_valid.mean())**2)
            r2_base = 1 - ss_res_base/ss_tot if ss_tot > 0 else 0
            
            # Extended model: entropy + variance + Φ*_PD
            X_ext = np.column_stack([ent_valid, var_valid, phi_valid, np.ones(n)])
            beta_ext, _, _, _ = lstsq(X_ext, rt_valid, rcond=None)
            pred_ext = X_ext @ beta_ext
            ss_res_ext = np.sum((rt_valid - pred_ext)**2)
            r2_ext = 1 - ss_res_ext/ss_tot if ss_tot > 0 else 0
            
            r2_increment = r2_ext - r2_base
            
            # F-test for the increment
            df1 = 1  # one additional predictor
            df2 = n - 4  # residual df for extended model
            if ss_res_ext > 0 and df2 > 0:
                f_stat = ((ss_res_base - ss_res_ext) / df1) / (ss_res_ext / df2)
                p_increment = 1 - stats.f.cdf(f_stat, df1, df2)
            else:
                f_stat = 0; p_increment = 1.0
            
            model_results.append({
                'sub': sub, 'n': n,
                'r_phi': r_phi, 'p_phi': p_phi,
                'r_ent': r_ent, 'p_ent': p_ent,
                'r_var': r_var, 'p_var': p_var,
                'r2_base': r2_base, 'r2_ext': r2_ext,
                'r2_increment': r2_increment,
                'f_stat': f_stat, 'p_increment': p_increment,
            })
            
            sig_inc = "*" if p_increment < 0.05 else ""
            print(f" N={n} | r_phi={r_phi:+.3f} r_ent={r_ent:+.3f} r_var={r_var:+.3f} | ΔR²={r2_increment:+.4f}{sig_inc}")
            
            # =========================================================
            # PART 2: Binned visualization (within-subject quintiles)
            # =========================================================
            
            # Sort by Φ*_PD, divide into 5 bins
            sort_idx = np.argsort(phi_valid)
            bin_size = len(sort_idx) // 5
            
            for q in range(5):
                start = q * bin_size
                end = (q+1) * bin_size if q < 4 else len(sort_idx)
                bin_idx = sort_idx[start:end]
                mean_rt = np.mean(rt_valid[bin_idx])
                all_subject_bins[q+1].append(mean_rt)
            
        except Exception as e:
            print(f" ERROR: {e}")
    
    # =====================================================================
    # PART 1: GROUP MODEL COMPARISON
    # =====================================================================
    if len(model_results) >= 3:
        print(f"\n{'='*75}")
        print(f"PART 1: MODEL COMPARISON (N={len(model_results)})")
        print("Does Φ*_PD predict RT beyond entropy + variance?")
        print("="*75)
        
        # Correlations with RT
        print(f"\n  Mean correlations with RT:")
        for metric, key in [("Φ*_PD","r_phi"), ("Entropy","r_ent"), ("Variance","r_var")]:
            vals = [r[key] for r in model_results]
            t, p = stats.ttest_1samp(vals, 0)
            sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
            n_pos = sum(1 for v in vals if v > 0)
            print(f"    {metric:<15} mean r = {np.mean(vals):+.3f} ± {np.std(vals,ddof=1):.3f}  t={t:.2f} p={p:.4f} {sig}  ({n_pos}/{len(vals)} positive)")
        
        # Incremental R²
        print(f"\n  Incremental R² (Φ*_PD added to entropy + variance):")
        increments = [r['r2_increment'] for r in model_results]
        p_increments = [r['p_increment'] for r in model_results]
        n_sig = sum(1 for p in p_increments if p < 0.05)
        n_pos = sum(1 for i in increments if i > 0)
        
        t_inc, p_inc = stats.ttest_1samp(increments, 0)
        sig_inc = "***" if p_inc<0.001 else "**" if p_inc<0.01 else "*" if p_inc<0.05 else "ns"
        
        print(f"    Mean ΔR² = {np.mean(increments):+.4f} ± {np.std(increments,ddof=1):.4f}")
        print(f"    t({len(increments)-1}) = {t_inc:.3f}, p = {p_inc:.4f} {sig_inc}")
        print(f"    {n_pos}/{len(increments)} subjects show positive increment")
        print(f"    {n_sig}/{len(increments)} subjects show significant increment (p<0.05)")
        
        print(f"\n  Interpretation:")
        if np.mean(increments) > 0 and p_inc < 0.05:
            print(f"    ★ Φ*_PD adds SIGNIFICANT explanatory power beyond entropy + variance")
            print(f"    The coordination pressure metric captures information about RT")
            print(f"    that simpler neural measures miss.")
        elif np.mean(increments) > 0:
            print(f"    Positive trend but not significant.")
            print(f"    Φ*_PD may add small incremental value.")
        else:
            print(f"    Φ*_PD does not add beyond entropy + variance.")
        
        # Per-subject detail
        print(f"\n  Per-subject detail:")
        print(f"    {'Subject':<12} {'r_Φ':>8} {'r_ent':>8} {'r_var':>8} {'R²_base':>8} {'R²_ext':>8} {'ΔR²':>8} {'p_inc':>8}")
        for r in model_results:
            sig = "*" if r['p_increment'] < 0.05 else ""
            print(f"    {r['sub']:<12} {r['r_phi']:>+8.3f} {r['r_ent']:>+8.3f} {r['r_var']:>+8.3f} {r['r2_base']:>8.4f} {r['r2_ext']:>8.4f} {r['r2_increment']:>+8.4f} {r['p_increment']:>8.4f}{sig}")
    
    # =====================================================================
    # PART 2: BINNED VISUALIZATION
    # =====================================================================
    if all(len(all_subject_bins[q]) >= 3 for q in range(1,6)):
        print(f"\n{'='*75}")
        print("PART 2: BINNED VISUALIZATION")
        print("Mean RT by Φ*_PD quintile (averaged across subjects)")
        print("="*75)
        
        print(f"\n  Quintile    Mean Φ*_PD rank    Mean RT (s)    SEM")
        print(f"  " + "-"*55)
        
        bin_means = []
        bin_sems = []
        for q in range(1, 6):
            vals = all_subject_bins[q]
            m = np.mean(vals)
            sem = np.std(vals, ddof=1) / np.sqrt(len(vals))
            bin_means.append(m)
            bin_sems.append(sem)
            label = "lowest Φ*_PD" if q == 1 else "highest Φ*_PD" if q == 5 else ""
            print(f"  Q{q} {label:<18}                {m:.4f}        ±{sem:.4f}")
        
        # Test for monotonic trend
        # Spearman correlation between quintile number and mean RT
        quintiles = np.array([1,2,3,4,5])
        bin_means_arr = np.array(bin_means)
        
        # Per-subject slope
        subject_slopes = []
        for si in range(len(all_subject_bins[1])):
            sub_rts = [all_subject_bins[q][si] for q in range(1,6)]
            slope, _, _, _, _ = stats.linregress(quintiles, sub_rts)
            subject_slopes.append(slope)
        
        t_slope, p_slope = stats.ttest_1samp(subject_slopes, 0)
        mean_slope = np.mean(subject_slopes)
        sig_slope = "***" if p_slope<0.001 else "**" if p_slope<0.01 else "*" if p_slope<0.05 else "ns"
        
        print(f"\n  Monotonic trend test:")
        print(f"    Mean slope (RT per quintile): {mean_slope:+.4f} s/quintile")
        print(f"    t({len(subject_slopes)-1}) = {t_slope:.3f}, p = {p_slope:.4f} {sig_slope}")
        
        if mean_slope > 0 and p_slope < 0.05:
            print(f"    ★ SIGNIFICANT MONOTONIC INCREASE: higher Φ*_PD → slower RT")
        elif mean_slope > 0:
            print(f"    Positive trend but not significant")
        else:
            print(f"    No monotonic trend")
        
        # ASCII visualization
        print(f"\n  Visual pattern:")
        max_rt = max(bin_means)
        min_rt = min(bin_means)
        rt_range = max_rt - min_rt if max_rt > min_rt else 0.001
        
        for q in range(5):
            bar_len = int(((bin_means[q] - min_rt) / rt_range) * 40) + 1
            bar = "█" * bar_len
            print(f"    Q{q+1}: {bar} {bin_means[q]:.4f}s")
        
        print(f"\n    Q1 = lowest coordination pressure")
        print(f"    Q5 = highest coordination pressure")
        if bin_means[4] > bin_means[0]:
            pct_diff = (bin_means[4] - bin_means[0]) / bin_means[0] * 100
            print(f"    Q5-Q1 difference: {bin_means[4]-bin_means[0]:.4f}s ({pct_diff:.1f}% slower)")
    
    print(f"\n{'='*75}")
    print("MODEL COMPARISON & VISUALIZATION COMPLETE")
    print("="*75)

if __name__ == "__main__":
    main()
