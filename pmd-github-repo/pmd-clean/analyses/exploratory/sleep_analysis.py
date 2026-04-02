#!/usr/bin/env python3
"""
PMD SLEEP CONSCIOUSNESS ANALYSIS v3 (fixed)
Handles variable TSV formats across subjects.
"""

import os, sys, warnings, gc
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.signal import detrend

warnings.filterwarnings('ignore')

N_NET = 7
DS_DIR = Path("ds003768")
SOURCEDATA = DS_DIR / "sourcedata"
if not SOURCEDATA.exists():
    SOURCEDATA = Path("ds003768_staging") / "sourcedata"

def get_yeo7_masks(bold_file):
    from nilearn import image, datasets
    import nibabel as nib
    import glob as g
    ydir = os.path.join(os.path.expanduser('~'), 'nilearn_data', 'yeo_2011')
    c = g.glob(os.path.join(ydir, '**', 'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz'), recursive=True)
    if not c:
        datasets.fetch_atlas_yeo_2011()
        c = g.glob(os.path.join(ydir, '**', '*7Networks*1mm.nii.gz'), recursive=True)
    atlas = image.resample_to_img(nib.load(c[0]), image.load_img(bold_file), interpolation='nearest')
    ad = atlas.get_fdata()
    while ad.ndim > 3: ad = ad[..., 0]
    return [np.round(ad) == i for i in range(1, 8)]

def get_timeseries(bold_file, masks):
    from nilearn import image
    import nibabel as nib
    # Memory-safe: use float32 instead of float64
    img = nib.load(bold_file)
    bd = np.asarray(img.dataobj, dtype=np.float32)
    n_tp = bd.shape[-1]
    d2 = bd.reshape(-1, n_tp)
    ts = np.zeros((n_tp, N_NET), dtype=np.float32)
    for i, m in enumerate(masks):
        mf = m.ravel()
        if mf.sum() < 10: continue
        t = d2[mf, :].mean(axis=0)
        t = detrend(t)
        s = t.std()
        if s > 0: t = t / s
        ts[:, i] = t
    del bd, d2
    gc.collect()
    return ts

def sym_te(s, t, lag=1, bins=10):
    def te(src, tgt):
        n = len(src)
        if n <= lag + 1: return 0.0
        yn, yp, xp = tgt[lag:], tgt[:-lag], src[:-lag]
        def bd(x):
            mn, mx = x.min(), x.max()
            if mx == mn: return np.zeros(len(x), dtype=int)
            return np.clip(((x - mn) / (mx - mn) * (bins - 1)).astype(int), 0, bins - 1)
        yb, ypb, xpb = bd(yn), bd(yp), bd(xp)
        def H(c): p = c[c > 0] / c.sum(); return -np.sum(p * np.log2(p))
        jyy = np.zeros((bins, bins))
        for i in range(len(yb)): jyy[yb[i], ypb[i]] += 1
        jyyx = np.zeros((bins, bins, bins))
        for i in range(len(yb)): jyyx[yb[i], ypb[i], xpb[i]] += 1
        jyx = np.zeros((bins, bins))
        for i in range(len(ypb)): jyx[ypb[i], xpb[i]] += 1
        return max((H(jyy.ravel()) - H(np.bincount(ypb, minlength=bins).astype(float))) -
                   (H(jyyx.ravel()) - H(jyx.ravel())), 0.0)
    return (te(s, t) + te(t, s)) / 2.0

def signal_coherence(ts, lag=1):
    if len(ts) < lag + 2: return 0.0
    ac = np.corrcoef(ts[:-lag], ts[lag:])[0, 1]
    return max(0.0, ac) if not np.isnan(ac) else 0.0

def soft_gate(x, x0, tau):
    z = np.clip((x - x0) / tau if tau > 0 else 0, -20, 20)
    return 1.0 / (1.0 + np.exp(-z))

def amplitude_jsd(ts_i, ts_j, bins=10):
    def hist_prob(x):
        h, _ = np.histogram(x, bins=bins, density=True)
        h = h + 1e-10
        return h / h.sum()
    pi = hist_prob(ts_i)
    pj = hist_prob(ts_j)
    m = 0.5 * (pi + pj)
    return max(0.5 * np.sum(pi * np.log2(pi / m)) + 0.5 * np.sum(pj * np.log2(pj / m)), 0)

def compute_phi(ts_seg):
    if ts_seg.shape[0] < 20: return None
    TE = np.zeros((N_NET, N_NET))
    for i in range(N_NET):
        for j in range(i + 1, N_NET):
            te = sym_te(ts_seg[:, i], ts_seg[:, j])
            TE[i, j] = TE[j, i] = te
    coh = np.array([signal_coherence(ts_seg[:, i]) for i in range(N_NET)])
    te_vals = [TE[i, j] for i in range(N_NET) for j in range(i + 1, N_NET)]
    i_min = np.percentile(te_vals, 25) if te_vals else 0
    J = np.zeros((N_NET, N_NET))
    for i in range(N_NET):
        for j in range(i + 1, N_NET):
            jsd = amplitude_jsd(ts_seg[:, i], ts_seg[:, j])
            g_c = soft_gate(TE[i, j], i_min, 0.05)
            g_r = np.sqrt(max(coh[i], 0) * max(coh[j], 0))
            J[i, j] = J[j, i] = TE[i, j] * jsd * g_c * g_r
    n_pairs = N_NET * (N_NET - 1) // 2
    return sum(J[i, j] for i in range(N_NET) for j in range(i + 1, N_NET)) / n_pairs


def load_staging(sub_id):
    """Load staging TSV, handling variable column formats."""
    num = sub_id.replace('sub-', '')
    
    for pattern in [f"sub-{num}-sleep-stage.tsv", f"sub-{int(num):02d}-sleep-stage.tsv"]:
        fpath = SOURCEDATA / pattern
        if fpath.exists():
            df = pd.read_csv(fpath, sep='\t')
            
            # Normalize column names — find the session, time, and stage columns
            cols = df.columns.tolist()
            
            # Find stage column (contains W, 1, 2, 3)
            stage_col = None
            session_col = None
            time_col = None
            
            for c in cols:
                if 'stage' in c.lower() or 'sleep' in c.lower():
                    stage_col = c
                elif 'session' in c.lower() or 'task' in c.lower():
                    session_col = c
                elif 'time' in c.lower() or 'epoch' in c.lower() or 'start' in c.lower():
                    time_col = c
            
            # Fallback: last column is usually stage, first is session or subject
            if stage_col is None:
                stage_col = cols[-1]
            if session_col is None:
                # First column that contains 'task-' values
                for c in cols:
                    vals = df[c].astype(str)
                    if vals.str.contains('task-').any():
                        session_col = c
                        break
                if session_col is None:
                    session_col = cols[0] if 'sub' not in cols[0].lower() else cols[1]
            if time_col is None:
                # Find numeric column
                for c in cols:
                    if c != stage_col and c != session_col:
                        try:
                            pd.to_numeric(df[c])
                            time_col = c
                            break
                        except:
                            continue
            
            # Standardize
            result = []
            for _, row in df.iterrows():
                session = str(row[session_col]).strip()
                try:
                    time_sec = float(row[time_col])
                except:
                    time_sec = 0
                stage = str(row[stage_col]).strip().upper()
                result.append({'session': session, 'time': time_sec, 'stage': stage})
            
            return result
    
    return None


def main():
    print("=" * 70)
    print("PMD SLEEP CONSCIOUSNESS ANALYSIS v3")
    print("Does Φ*_PD drop during natural sleep?")
    print("=" * 70)
    
    # Find subjects
    available = []
    for sub_dir in sorted(DS_DIR.iterdir()):
        if sub_dir.is_dir() and sub_dir.name.startswith('sub-'):
            func = sub_dir / "func"
            if func.exists() and list(func.glob("*bold.nii.gz")):
                available.append(sub_dir.name)
    
    print(f"\n  Subjects: {len(available)}")
    print(f"  Staging dir: {SOURCEDATA}")
    
    # Load atlas
    first_func = DS_DIR / available[0] / "func"
    first_bold = sorted(first_func.glob("*bold.nii.gz"))[0]
    print(f"  Loading Yeo-7 atlas...", end="", flush=True)
    masks = get_yeo7_masks(str(first_bold))
    print(" done\n")
    
    all_results = []
    
    for sub in available:
        print(f"  {sub}:", end="", flush=True)
        func_dir = DS_DIR / sub / "func"
        
        staging = load_staging(sub)
        if staging is None:
            print(f" no staging — skip")
            continue
        
        print(f" {len(staging)} epochs", end="", flush=True)
        
        # Get all BOLD files
        bold_files = sorted(func_dir.glob("*bold.nii.gz"))
        rest_bolds = [f for f in bold_files if 'task-rest' in f.name]
        sleep_bolds = [f for f in bold_files if 'task-sleep' in f.name]
        
        sub_data = {'subject': sub}
        tr = 2.1
        trs_per_epoch = int(30.0 / tr)  # ~14 TRs per 30-sec epoch
        
        # Process ALL bold files (rest and sleep), segmenting by staging
        all_bolds = rest_bolds + sleep_bolds
        
        stage_trs_global = {'W': {}, 'N1': {}, 'N2': {}, 'N3': {}}
        # Map: stage -> {bold_file_index: [tr_indices]}
        
        for bi, bf in enumerate(all_bolds):
            # Figure out which session this file is
            # Extract task-XXX_run-Y from filename
            parts = bf.stem.split('_')
            task_part = [p for p in parts if p.startswith('task-')]
            run_part = [p for p in parts if p.startswith('run-')]
            
            if task_part and run_part:
                session_key = f"{task_part[0]}_{run_part[0]}"
            elif task_part:
                session_key = task_part[0]
            else:
                continue
            
            # Find staging entries for this session
            session_epochs = [s for s in staging if session_key in s['session']]
            
            if not session_epochs:
                # Try looser match
                for tk in [t for t in task_part]:
                    session_epochs = [s for s in staging if tk in s['session']]
                    if session_epochs: break
            
            if not session_epochs:
                continue
            
            # Get timeseries
            try:
                ts = get_timeseries(str(bf), masks)
                n_tp = ts.shape[0]
            except Exception as e:
                print(f" ERR:{str(e)[:30]}", end="")
                continue
            
            # Segment by stage
            for ep in session_epochs:
                stage = ep['stage']
                if stage == '1': stage = 'N1'
                elif stage == '2': stage = 'N2'
                elif stage == '3': stage = 'N3'
                
                if stage not in stage_trs_global:
                    continue
                
                start_tr = int(ep['time'] / tr)
                end_tr = min(start_tr + trs_per_epoch, n_tp)
                
                if start_tr < n_tp:
                    if bi not in stage_trs_global[stage]:
                        stage_trs_global[stage][bi] = {'ts': ts, 'trs': []}
                    stage_trs_global[stage][bi]['trs'].extend(range(start_tr, end_tr))
        
        # Compute Φ for each stage by concatenating all TRs across runs
        for stage_name in ['W', 'N1', 'N2', 'N3']:
            all_ts_segments = []
            for bi_data in stage_trs_global[stage_name].values():
                ts = bi_data['ts']
                trs = sorted(set(bi_data['trs']))
                valid_trs = [t for t in trs if t < ts.shape[0]]
                if valid_trs:
                    all_ts_segments.append(ts[valid_trs, :])
            
            if all_ts_segments:
                combined = np.concatenate(all_ts_segments, axis=0)
                if combined.shape[0] >= 30:
                    phi = compute_phi(combined)
                    if phi is not None:
                        sub_data[stage_name] = phi
                        print(f" {stage_name}={phi:.6f}", end="")
        
        # Clean up memory
        del stage_trs_global
        gc.collect()
        
        if len(sub_data) > 1:  # Has at least one stage
            all_results.append(sub_data)
        
        print()
    
    # =====================================================================
    # RESULTS
    # =====================================================================
    print(f"\n{'='*70}")
    print("RESULTS")
    print("="*70)
    
    print(f"\n  {'Subject':<10} {'Wake':>10} {'NREM1':>10} {'NREM2':>10} {'NREM3':>10}")
    print(f"  {'-'*52}")
    for d in all_results:
        w = f"{d.get('W', 0):.6f}" if 'W' in d else "   ---"
        n1 = f"{d.get('N1', 0):.6f}" if 'N1' in d else "   ---"
        n2 = f"{d.get('N2', 0):.6f}" if 'N2' in d else "   ---"
        n3 = f"{d.get('N3', 0):.6f}" if 'N3' in d else "   ---"
        print(f"  {d['subject']:<10} {w:>10} {n1:>10} {n2:>10} {n3:>10}")
    
    # Paired tests
    print(f"\n{'='*70}")
    print("STATISTICAL TESTS")
    print("="*70)
    
    for compare_stage in ['N1', 'N2', 'N3']:
        pairs = [(d['W'], d[compare_stage]) for d in all_results if 'W' in d and compare_stage in d]
        if len(pairs) >= 3:
            w_arr = np.array([p[0] for p in pairs])
            s_arr = np.array([p[1] for p in pairs])
            diff = w_arr - s_arr
            n_correct = (diff > 0).sum()
            
            if len(pairs) >= 5:
                t, p = stats.ttest_rel(w_arr, s_arr)
                d_cohen = diff.mean() / (diff.std() + 1e-15)
            else:
                # Too few for t-test, use sign test
                t = np.nan
                from math import comb
                k = int(n_correct)
                n = len(pairs)
                p = sum(comb(n, i) for i in range(k, n+1)) / 2**n
                d_cohen = diff.mean() / (diff.std() + 1e-15)
            
            direction = "CORRECT ✓" if w_arr.mean() > s_arr.mean() else "WRONG ✗"
            
            print(f"\n  ★ Wake > {compare_stage}: {direction}")
            print(f"    Wake: {w_arr.mean():.6f}, {compare_stage}: {s_arr.mean():.6f}")
            print(f"    Ratio: {w_arr.mean() / (s_arr.mean() + 1e-15):.2f}x")
            print(f"    {n_correct}/{len(pairs)} subjects correct direction")
            if not np.isnan(t):
                print(f"    t({len(pairs)-1}) = {t:.3f}, p = {p:.4f}, d = {d_cohen:.3f}")
            else:
                print(f"    Sign test p = {p:.4f}, d = {d_cohen:.3f}")
    
    # =====================================================================
    # COMBINED VERDICT
    # =====================================================================
    print(f"\n{'='*70}")
    print("COMBINED CONSCIOUSNESS EVIDENCE")
    print("="*70)
    print(f"""
  MANIPULATION 1 — Propofol (ds006623, N=11):
    Pre-LOR > Post-LOR: p = 0.0053, d = 1.12, 9/11 correct
    Mechanism: GABAergic inhibition
    
  MANIPULATION 2 — Natural Sleep (ds003768):""")
    
    for compare_stage in ['N2', 'N3']:
        pairs = [(d['W'], d[compare_stage]) for d in all_results if 'W' in d and compare_stage in d]
        if len(pairs) >= 3:
            w_arr = np.array([p[0] for p in pairs])
            s_arr = np.array([p[1] for p in pairs])
            t, p = stats.ttest_rel(w_arr, s_arr) if len(pairs) >= 5 else (np.nan, np.nan)
            print(f"    Wake > {compare_stage}: N={len(pairs)}, p = {p:.4f}" if not np.isnan(p) else f"    Wake > {compare_stage}: N={len(pairs)}")
    
    print(f"""
  If BOTH show the same direction:
    → Propofol (drug) and Sleep (natural) = different mechanisms
    → Same metric drops when consciousness fades
    → "Tracks propofol pharmacology" eliminated as explanation
    → Φ*_PD tracks consciousness
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
