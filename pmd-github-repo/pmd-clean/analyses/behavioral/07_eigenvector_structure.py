#!/usr/bin/env python3
"""
===========================================================================
EIGENVECTOR INDEPENDENCE TEST
Addresses GPT's concern: is the 0.996 alignment structural or empirical?

The concern: if the "coordination state" (row sums of J) is computed from
the same matrix whose eigenvector we're testing, high alignment is
mathematically expected rather than empirically discovered.

THE FIX: Split trials in half.
  - Compute J_A from odd-numbered GO trials
  - Compute J_B from even-numbered GO trials  
  - Extract principal eigenvector from J_A
  - Extract coordination state (row sums) from J_B
  - Measure alignment between them

If alignment stays high with independent data, it's a real property.
If it drops, the original result was partly definitional.

ALSO: Decoder accuracy vs Φ-RT slope diagnostic
===========================================================================
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')

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


def compute_J_from_trials(decoded_probs, trial_indices, timeseries, tau=0.05):
    """Compute J matrix using only specified trial indices for the C term."""
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
            for t in trial_indices:
                pi = decoded_probs[i][t]+1e-10; pj = decoded_probs[j][t]+1e-10
                pi /= pi.sum(); pj /= pj.sum(); m = 0.5*(pi+pj)
                jsds.append(max(0.5*np.sum(pi*np.log2(pi/m))+0.5*np.sum(pj*np.log2(pj/m)),0))
            mean_jsd = np.mean(jsds) if jsds else 0
            g_c = soft_gate(TE[i,j], i_min, tau)
            g_r = np.sqrt(max(coh[i],0)*max(coh[j],0))
            g_d = np.sqrt(net_conf.get(i,0)*net_conf.get(j,0))
            J[i,j] = J[j,i] = TE[i,j] * mean_jsd * g_c * g_r * g_d
    np.fill_diagonal(J, 0)
    return J


def main():
    data_dir = Path("./ds000030")
    subjects = ['sub-10159','sub-10206','sub-10269','sub-10271','sub-10273',
                'sub-10274','sub-10280','sub-10290','sub-10292','sub-10299']
    
    print("="*80)
    print("EIGENVECTOR INDEPENDENCE TEST")
    print("Is the 0.996 alignment structural or empirical?")
    print("="*80)
    
    split_alignments = []
    full_alignments = []
    decoder_accs = []
    phi_rt_slopes = []
    
    for sub in subjects:
        bold = data_dir/sub/"func"/f"{sub}_task-stopsignal_bold.nii.gz"
        evts = data_dir/sub/"func"/f"{sub}_task-stopsignal_events.tsv"
        jpath = data_dir/sub/"func"/f"{sub}_task-stopsignal_bold.json"
        if not bold.exists(): continue
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
            
            mean_acc = np.mean(list(accs.values()))
            decoder_accs.append(mean_acc)
            
            # Get GO trial indices
            go_mask = (labels == 'GO')
            go_indices = np.where(go_mask)[0]
            n_go = len(go_indices)
            
            # Split into odd and even
            odd_trials = go_indices[0::2]   # odd-indexed GO trials
            even_trials = go_indices[1::2]  # even-indexed GO trials
            
            # Compute J from each half
            J_odd = compute_J_from_trials(decoded, odd_trials, ts)
            J_even = compute_J_from_trials(decoded, even_trials, ts)
            
            # Full J (all GO trials)
            J_full = compute_J_from_trials(decoded, go_indices, ts)
            
            # Eigenvector from odd-half
            Js_odd = np.nan_to_num((J_odd + J_odd.T)/2)
            evals_odd, evecs_odd = np.linalg.eigh(Js_odd)
            evec_odd = np.abs(evecs_odd[:, np.argmax(evals_odd)])
            
            # Coordination state (row sums) from even-half
            row_sums_even = J_even.sum(axis=1)
            if row_sums_even.sum() > 0:
                weights_even = row_sums_even / row_sums_even.sum()
            else:
                weights_even = np.ones(7)/7
            
            # Split-half alignment
            if np.linalg.norm(evec_odd) > 0 and np.linalg.norm(weights_even) > 0:
                split_align = np.dot(evec_odd, weights_even) / (np.linalg.norm(evec_odd) * np.linalg.norm(weights_even))
            else:
                split_align = 0
            split_alignments.append(split_align)
            
            # Full alignment (for comparison — this is what was 0.996 before)
            Js_full = np.nan_to_num((J_full + J_full.T)/2)
            evals_full, evecs_full = np.linalg.eigh(Js_full)
            evec_full = np.abs(evecs_full[:, np.argmax(evals_full)])
            row_sums_full = J_full.sum(axis=1)
            if row_sums_full.sum() > 0:
                weights_full = row_sums_full / row_sums_full.sum()
            else:
                weights_full = np.ones(7)/7
            if np.linalg.norm(evec_full) > 0 and np.linalg.norm(weights_full) > 0:
                full_align = np.dot(evec_full, weights_full) / (np.linalg.norm(evec_full) * np.linalg.norm(weights_full))
            else:
                full_align = 0
            full_alignments.append(full_align)
            
            # Φ-RT slope for decoder diagnostic
            go_events = events_filtered.iloc[go_indices].copy()
            trial_rts = pd.to_numeric(go_events['ReactionTime'], errors='coerce').values
            
            # Quick per-trial phi
            trial_phis = []
            for trial_idx in go_indices:
                J_t = np.zeros((7,7))
                for i in range(7):
                    for j in range(i+1,7):
                        pi = decoded[i][trial_idx]+1e-10; pj = decoded[j][trial_idx]+1e-10
                        pi /= pi.sum(); pj /= pj.sum(); m_ = 0.5*(pi+pj)
                        jsd = max(0.5*np.sum(pi*np.log2(pi/m_))+0.5*np.sum(pj*np.log2(pj/m_)),0)
                        J_t[i,j] = jsd
                trial_phis.append(np.mean([J_t[i,j] for i in range(7) for j in range(i+1,7)]))
            trial_phis = np.array(trial_phis)
            
            valid = (~np.isnan(trial_rts)) & (trial_rts > 0.1) & (trial_rts < 2.0)
            if valid.sum() > 10:
                r, _ = stats.pearsonr(trial_phis[valid], trial_rts[valid])
                phi_rt_slopes.append(r)
            else:
                phi_rt_slopes.append(np.nan)
            
            print(f" full={full_align:.4f} split={split_align:.4f} acc={mean_acc:.3f}", end="")
            
        except Exception as e:
            print(f" ERROR: {e}", end="")
    
    N = len(split_alignments)
    
    # =====================================================================
    # RESULTS
    # =====================================================================
    print(f"\n\n{'='*80}")
    print(f"EIGENVECTOR INDEPENDENCE TEST RESULTS (N={N})")
    print("="*80)
    
    null_align = 1.0 / np.sqrt(7)
    
    print(f"\n  Full alignment (eigvec & weights from same data):")
    print(f"    Mean = {np.mean(full_alignments):.4f}")
    print(f"    This is the original 0.996 result")
    
    print(f"\n  Split-half alignment (eigvec from odd, weights from even):")
    print(f"    Mean = {np.mean(split_alignments):.4f}")
    print(f"    Random null = {null_align:.4f}")
    
    t_split, p_split = stats.ttest_1samp(split_alignments, null_align)
    print(f"    t({N-1}) = {t_split:.3f}, p = {p_split:.4f}")
    
    drop = np.mean(full_alignments) - np.mean(split_alignments)
    print(f"\n  Drop from full to split-half: {drop:.4f}")
    
    if np.mean(split_alignments) > 0.9:
        print(f"\n  ★ Split-half alignment remains very high (>{0.9:.1f}).")
        print(f"    The eigenvector structure is a REAL property of brain coordination,")
        print(f"    not an artifact of computing eigenvec and weights from the same data.")
    elif np.mean(split_alignments) > 0.7:
        print(f"\n  Split-half alignment is moderate. The eigenvector structure is partly")
        print(f"  real but the original 0.996 was inflated by shared construction.")
    elif np.mean(split_alignments) > null_align and p_split < 0.05:
        print(f"\n  Split-half alignment exceeds random but is much lower than full.")
        print(f"  The original 0.996 was substantially structural.")
    else:
        print(f"\n  Split-half alignment near random. The original result was definitional.")
    
    # =====================================================================
    # DECODER ACCURACY vs SLOPE DIAGNOSTIC
    # =====================================================================
    print(f"\n{'='*80}")
    print("DECODER ACCURACY vs Φ-RT SLOPE")
    print("Do negative slopes come from bad decoders?")
    print("="*80)
    
    valid_both = [(a, s) for a, s in zip(decoder_accs, phi_rt_slopes) if not np.isnan(s)]
    if len(valid_both) >= 5:
        accs_v = [x[0] for x in valid_both]
        slopes_v = [x[1] for x in valid_both]
        r_diag, p_diag = stats.pearsonr(accs_v, slopes_v)
        print(f"\n  Correlation: decoder_accuracy ~ Φ-RT_slope")
        print(f"    r = {r_diag:+.3f}, p = {p_diag:.4f}")
        
        if r_diag > 0.3 and p_diag < 0.05:
            print(f"\n  Positive correlation: better decoders → stronger Φ-RT link.")
            print(f"  Negative slopes likely reflect measurement noise, not real reversal.")
        elif abs(r_diag) < 0.2:
            print(f"\n  No relationship: decoder quality does not drive slope differences.")
            print(f"  Negative slopes may reflect genuine individual differences.")
        else:
            print(f"\n  Weak relationship. Inconclusive.")
    
    print(f"\n{'='*80}")
    print("EIGENVECTOR INDEPENDENCE TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
