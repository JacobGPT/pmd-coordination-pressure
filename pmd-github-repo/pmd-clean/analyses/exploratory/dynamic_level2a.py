#!/usr/bin/env python3
"""
===========================================================================
PRESSURE MAKES DIAMONDS — DYNAMIC LEVEL 2A
Temporal Dynamics of Effective Coordination Pressure
===========================================================================

Static Level 2A showed the correct direction on both datasets but with
small effect sizes. Psychedelics may change the TEMPORAL STRUCTURE of
coordination more than the static mean.

This script computes Level 2A metrics in sliding windows and measures:
  - Mean λ_max_eff(t): average coordination pressure (same as static)
  - Std λ_max_eff(t):  variability of coordination pressure over time
  - Range:             max - min of λ_max_eff over time
  - Metastability:     how much the coordination regime fluctuates
  - Dwell time above threshold: fraction of windows in high-pressure state

LSD prediction: increased VARIABILITY of coordination (more dynamic,
  more state-switching, wider range of conscious intensity)
Propofol prediction: DECREASED variability (fragmented, less dynamic,
  stuck in low-coordination state)
===========================================================================
"""

import os, sys, json, argparse, warnings
import numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')

YEO7_LABELS = ["Visual","Somatomotor","DorsalAttention","VentralAttention","Limbic","Frontoparietal","Default"]

# =========================================================================
# TIMESERIES + MEASURES (same as Level 2A)
# =========================================================================

def extract_network_timeseries(bold_file):
    from nilearn import maskers
    import glob
    yeo_dir = os.path.join(os.path.expanduser('~'), 'nilearn_data', 'yeo_2011')
    cands = glob.glob(os.path.join(yeo_dir, '**', 'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz'), recursive=True)
    if not cands: cands = glob.glob(os.path.join(yeo_dir, '**', '*7Networks*1mm.nii.gz'), recursive=True)
    if not cands: raise RuntimeError("No Yeo atlas found")
    masker = maskers.NiftiLabelsMasker(labels_img=cands[0], standardize='zscore_sample', detrend=True, memory='nilearn_cache', memory_level=1)
    ts = masker.fit_transform(bold_file)
    if ts.shape[0] < 30: raise ValueError(f"Too few timepoints: {ts.shape[0]}")
    if ts.shape[1] != 7:
        pad = np.zeros((ts.shape[0], 7)); pad[:, :min(ts.shape[1],7)] = ts[:, :min(ts.shape[1],7)]; ts = pad
    return ts

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

def signal_jsd(a, b, bins=20):
    v = np.concatenate([a,b]); mn,mx = v.min(),v.max()
    if mx==mn: return 0.0
    e = np.linspace(mn,mx,bins+1); eps=1e-10
    p = np.histogram(a,bins=e)[0].astype(float); q = np.histogram(b,bins=e)[0].astype(float)
    p=(p+eps)/((p+eps).sum()); q=(q+eps)/((q+eps).sum()); m=0.5*(p+q)
    return max(0.5*np.sum(p*np.log2(p/m)) + 0.5*np.sum(q*np.log2(q/m)), 0.0)

def windowed_jsd(a, b, win=60, step=30, bins=20):
    n = len(a)
    if n < win: return signal_jsd(a,b,bins), 0.0
    vals = [signal_jsd(a[s:s+win], b[s:s+win], bins) for s in range(0, n-win+1, step)]
    return (np.mean(vals), np.std(vals)) if vals else (signal_jsd(a,b,bins), 0.0)

def signal_coherence(ts, lag=1):
    if len(ts) < lag+2: return 0.0
    ac = np.corrcoef(ts[:-lag], ts[lag:])[0,1]
    return max(0.0, ac) if not np.isnan(ac) else 0.0

def soft_gate(x, x0, tau):
    z = np.clip((x-x0)/tau if tau>0 else 0, -20, 20)
    return 1.0/(1.0+np.exp(-z))

# =========================================================================
# WINDOWED LEVEL 2A COMPUTATION
# =========================================================================

def compute_dynamic_level2a(timeseries, window=80, step=20, tau=0.05):
    """
    Compute Level 2A metrics in sliding windows across the timeseries.
    Returns a timeseries of λ_max_eff values plus summary statistics.
    """
    N = timeseries.shape[1]  # 7
    n_tp = timeseries.shape[0]
    
    if n_tp < window:
        raise ValueError(f"Timeseries ({n_tp}) shorter than window ({window})")
    
    # Compute global coherence (used for all windows)
    coh = np.array([signal_coherence(timeseries[:, i]) for i in range(N)])
    
    # Compute global I_min for coupling gate
    global_te = []
    for i in range(N):
        for j in range(i+1, N):
            global_te.append(sym_te(timeseries[:, i], timeseries[:, j]))
    i_min = np.percentile(global_te, 25)
    
    # Sliding window analysis
    lambda_eff_series = []
    phi_eff_series = []
    window_times = []
    
    for start in range(0, n_tp - window + 1, step):
        end = start + window
        win_ts = timeseries[start:end, :]
        
        # Compute pairwise measures in this window
        J_eff = np.zeros((N, N))
        
        for i in range(N):
            for j in range(i+1, N):
                te = sym_te(win_ts[:, i], win_ts[:, j])
                jsd = signal_jsd(win_ts[:, i], win_ts[:, j])
                
                # Level 2A gates
                # For windowed JSD stability, use sub-windows within this window
                sub_win = window // 3
                if sub_win > 10:
                    sub_jsds = [signal_jsd(win_ts[s:s+sub_win, i], win_ts[s:s+sub_win, j], 15) 
                               for s in range(0, window-sub_win+1, sub_win//2)]
                    jsd_std = np.std(sub_jsds) if sub_jsds else 0
                else:
                    jsd_std = 0
                
                g_stab = 1.0 / (1.0 + jsd_std)
                g_coup = soft_gate(te, i_min, tau)
                g_coh = np.sqrt(max(coh[i], 0) * max(coh[j], 0))
                
                c_eff = jsd * g_stab * g_coup * g_coh
                j_eff = te * c_eff
                
                J_eff[i,j] = J_eff[j,i] = j_eff
        
        np.fill_diagonal(J_eff, 0)
        
        # Eigenvalue
        Js = np.nan_to_num((J_eff + J_eff.T) / 2.0)
        try: ev = np.linalg.eigvalsh(Js)
        except: ev = np.real(np.linalg.eigvals(Js))
        ev = np.sort(np.real(ev))[::-1]
        
        # Phi
        phi = sum(J_eff[i,j] for i in range(N) for j in range(i+1,N))
        phi_norm = phi / (N*(N-1)//2)
        
        lambda_eff_series.append(float(ev[0]))
        phi_eff_series.append(float(phi_norm))
        window_times.append(float(start + window/2))
    
    lam = np.array(lambda_eff_series)
    phi = np.array(phi_eff_series)
    
    # Also compute static (whole-scan) Level 2A for comparison
    J_static = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            te = sym_te(timeseries[:, i], timeseries[:, j])
            jm, js = windowed_jsd(timeseries[:, i], timeseries[:, j])
            g_stab = 1.0 / (1.0 + js)
            g_coup = soft_gate(te, i_min, tau)
            g_coh = np.sqrt(max(coh[i], 0) * max(coh[j], 0))
            J_static[i,j] = J_static[j,i] = te * jm * g_stab * g_coup * g_coh
    np.fill_diagonal(J_static, 0)
    Jss = np.nan_to_num((J_static + J_static.T) / 2.0)
    try: sev = np.linalg.eigvalsh(Jss)
    except: sev = np.real(np.linalg.eigvals(Jss))
    sev = np.sort(np.real(sev))[::-1]
    static_lmax = float(sev[0])
    
    # Compute dynamic summary statistics
    if len(lam) > 1:
        # Metastability: std of the order parameter over time
        metastability = float(np.std(lam))
        # Range
        dyn_range = float(np.max(lam) - np.min(lam))
        # Coefficient of variation
        cv = float(np.std(lam) / np.mean(lam)) if np.mean(lam) > 0 else 0
        # Fraction of windows in "high pressure" state (above median)
        median_lam = np.median(lam)
        high_dwell = float(np.mean(lam > median_lam))
    else:
        metastability = 0; dyn_range = 0; cv = 0; high_dwell = 0.5
    
    return {
        'static_lmax': static_lmax,
        'dynamic_mean': float(np.mean(lam)),
        'dynamic_std': float(np.std(lam)),
        'dynamic_range': dyn_range,
        'dynamic_cv': cv,
        'metastability': metastability,
        'dynamic_max': float(np.max(lam)),
        'dynamic_min': float(np.min(lam)),
        'n_windows': len(lam),
        'coherence': float(np.mean(coh)),
        'lambda_series': [float(x) for x in lam],
    }


# =========================================================================
# FILE FINDERS
# =========================================================================

def find_anesthesia(d):
    d = Path(d); r = {}
    tm = {'restawake':'Awake','restlight':'Mild Sedation','restdeep':'Deep Sedation','restrecovery':'Recovery'}
    for f in sorted(d.glob("sub-*/func/*task-rest*bold.nii.gz")):
        p = str(f).replace('\\','/')
        sub = next((x for x in p.split('/') if x.startswith('sub-')), None)
        cond = next((v for k,v in tm.items() if f"task-{k}" in f.name), None)
        if sub and cond: r.setdefault(sub,{})[cond] = str(f)
    return r

def find_psychedelic(d):
    d = Path(d); r = {}
    for f in sorted(d.glob("sub-*/ses-*/*task-rest*bold.nii.gz")):
        p = str(f).replace('\\','/')
        sub = next((x for x in p.split('/') if x.startswith('sub-')), None)
        cond = None
        for x in p.split('/'):
            if 'ses-' in x:
                if 'LSD' in x.upper(): cond='LSD'
                elif 'PLCB' in x.upper(): cond='Placebo'
        if sub and cond: r.setdefault(sub,{}).setdefault(cond,[]).append(str(f))
    return r


# =========================================================================
# MAIN ANALYSIS
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anesthesia-dir", default="./ds003171")
    parser.add_argument("--psychedelic-dir", default="./psychedelic_data")
    parser.add_argument("--output-dir", default="./pmd_results")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--anesthesia-only", action="store_true")
    parser.add_argument("--psychedelic-only", action="store_true")
    args = parser.parse_args()
    
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    
    print("=" * 75)
    print("PRESSURE MAKES DIAMONDS — DYNAMIC LEVEL 2A")
    print("Temporal Dynamics of Effective Coordination Pressure")
    print("=" * 75)
    print()
    print("  Measures λ_max_eff in sliding windows to capture:")
    print("  - Mean: average coordination pressure (same as static)")
    print("  - Std/Metastability: temporal variability of coordination")  
    print("  - Range: max-min spread of coordination pressure")
    print("  - CV: coefficient of variation (normalized variability)")
    print()
    print("  LSD prediction:      increased variability (more dynamic)")
    print("  Propofol prediction:  decreased variability (fragmented/stuck)")
    print("=" * 75)
    
    from scipy import stats
    
    # =====================================================================
    # ANESTHESIA
    # =====================================================================
    if not args.psychedelic_only:
        print("\n" + "=" * 75)
        print("ANESTHESIA: Dynamic Level 2A")
        print("=" * 75)
        
        af = find_anesthesia(args.anesthesia_dir)
        if af:
            print(f"  Subjects: {len(af)}")
            anes_data = {}
            
            for sub in sorted(af):
                print(f"\n  {sub}:")
                for cond, fp in sorted(af[sub].items()):
                    print(f"    {cond}...", end="", flush=True)
                    try:
                        ts = extract_network_timeseries(fp)
                        m = compute_dynamic_level2a(ts)
                        anes_data.setdefault(cond, []).append(m)
                        print(f" static={m['static_lmax']:.4f} dyn_mean={m['dynamic_mean']:.4f} dyn_std={m['dynamic_std']:.4f} meta={m['metastability']:.4f} wins={m['n_windows']}")
                    except Exception as e:
                        print(f" ERROR: {e}")
            
            order = ["Awake","Mild Sedation","Deep Sedation","Recovery"]
            print(f"\n  {'Condition':<20} {'Static':>8} {'Dyn Mean':>10} {'Dyn Std':>10} {'Range':>8} {'CV':>8} {'Meta':>8}")
            print("  " + "-" * 78)
            
            for c in order:
                if c not in anes_data: continue
                ml = anes_data[c]
                st = np.mean([m['static_lmax'] for m in ml])
                dm = np.mean([m['dynamic_mean'] for m in ml])
                ds = np.mean([m['dynamic_std'] for m in ml])
                dr = np.mean([m['dynamic_range'] for m in ml])
                cv = np.mean([m['dynamic_cv'] for m in ml])
                mt = np.mean([m['metastability'] for m in ml])
                print(f"  {c:<20} {st:>8.4f} {dm:>10.4f} {ds:>10.4f} {dr:>8.4f} {cv:>8.3f} {mt:>8.4f}")
            
            # Key tests
            print(f"\n  KEY DYNAMIC TESTS:")
            if "Awake" in anes_data and "Deep Sedation" in anes_data:
                for metric, label in [('dynamic_mean','Mean'), ('dynamic_std','Variability'), ('metastability','Metastability'), ('dynamic_range','Range')]:
                    aw = [m[metric] for m in anes_data["Awake"]]
                    dp = [m[metric] for m in anes_data["Deep Sedation"]]
                    t, p = stats.ttest_ind(aw, dp)
                    d = (np.mean(aw)-np.mean(dp)) / np.sqrt((np.std(aw)**2+np.std(dp)**2)/2) if (np.std(aw)+np.std(dp))>0 else 0
                    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
                    direction = "Awake > Deep" if np.mean(aw) > np.mean(dp) else "Deep > Awake"
                    print(f"    {label:<15} {direction:<20} d={d:+.3f} p={p:.4f} {sig}")
    
    # =====================================================================
    # PSYCHEDELICS
    # =====================================================================
    if not args.anesthesia_only:
        print("\n" + "=" * 75)
        print("PSYCHEDELICS: Dynamic Level 2A")
        print("=" * 75)
        
        pf = find_psychedelic(args.psychedelic_dir)
        paired = {s:f for s,f in pf.items() if 'LSD' in f and 'Placebo' in f}
        
        if paired:
            print(f"  Paired subjects: {len(paired)}")
            lsd_all, plcb_all = [], []
            
            for sub in sorted(paired):
                print(f"\n  {sub}:")
                for cond in ['LSD','Placebo']:
                    run_metrics = []
                    for fp in paired[sub][cond]:
                        print(f"    {cond}: {os.path.basename(fp)}...", end="", flush=True)
                        try:
                            ts = extract_network_timeseries(fp)
                            m = compute_dynamic_level2a(ts)
                            run_metrics.append(m)
                            print(f" mean={m['dynamic_mean']:.4f} std={m['dynamic_std']:.4f} meta={m['metastability']:.4f}")
                        except Exception as e:
                            print(f" ERROR: {e}")
                    
                    if run_metrics:
                        avg = {k: float(np.mean([r[k] for r in run_metrics])) 
                               for k in ['static_lmax','dynamic_mean','dynamic_std','dynamic_range','dynamic_cv','metastability','dynamic_max','dynamic_min','coherence']}
                        if cond == 'LSD': lsd_all.append(avg)
                        else: plcb_all.append(avg)
            
            n = min(len(lsd_all), len(plcb_all))
            if n > 0:
                print(f"\n  {'Metric':<20} {'LSD':>12} {'Placebo':>12} {'Dir':>8} {'d':>8} {'p':>8}")
                print("  " + "-" * 70)
                
                for label, key in [("Static λ_eff","static_lmax"), ("Dynamic Mean","dynamic_mean"),
                                    ("Dynamic Std","dynamic_std"), ("Metastability","metastability"),
                                    ("Range","dynamic_range"), ("CV","dynamic_cv"),
                                    ("Dynamic Max","dynamic_max"), ("Coherence","coherence")]:
                    lv = [m[key] for m in lsd_all[:n]]
                    pv = [m[key] for m in plcb_all[:n]]
                    t, p = stats.ttest_rel(lv, pv)
                    diff = np.array(lv) - np.array(pv)
                    d = np.mean(diff) / (np.std(diff)+1e-10)
                    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
                    dr = "LSD ↑" if np.mean(lv) > np.mean(pv) else "LSD ↓"
                    print(f"  {label:<20} {np.mean(lv):>12.6f} {np.mean(pv):>12.6f} {dr:>8} {d:>+8.3f} {p:>8.4f} {sig}")
                
                print(f"\n  KEY QUESTION: Does LSD increase dynamic variability?")
                lsd_std = [m['dynamic_std'] for m in lsd_all[:n]]
                plcb_std = [m['dynamic_std'] for m in plcb_all[:n]]
                t, p = stats.ttest_rel(lsd_std, plcb_std)
                diff = np.array(lsd_std) - np.array(plcb_std)
                d = np.mean(diff) / (np.std(diff)+1e-10)
                if np.mean(lsd_std) > np.mean(plcb_std):
                    print(f"    YES — LSD shows higher coordination variability (d={d:+.3f}, p={p:.4f})")
                else:
                    print(f"    NO — LSD shows lower coordination variability (d={d:+.3f}, p={p:.4f})")
    
    # Save
    print("\n" + "=" * 75)
    print("DYNAMIC ANALYSIS COMPLETE")
    print("=" * 75)

if __name__ == "__main__":
    main()
