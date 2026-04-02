#!/usr/bin/env python3
"""
==========================================================================
PRESSURE MAKES DIAMONDS — HCP Analysis Pipeline
==========================================================================

Tests the framework's ARCHITECTURAL predictions using the Human
Connectome Project dataset. Unlike the psychedelic analysis (which
tests a pharmacological perturbation), this tests what the framework
actually describes: how different configurations of multi-domain
engagement produce different levels of coordination pressure.

Two analysis modes:

MODE A: PTN Netmats (preferred — fast, 1003 subjects)
  Uses pre-computed connectivity matrices from HCP's Parcellation +
  Timeseries + Netmats release. Each subject has a 100x100 connectivity
  matrix already computed. We map ICA components to Yeo-7 networks,
  compute Φ_PD, and correlate with behavioral measures.

  Required files:
    - netmats1.txt (correlation matrices, 1003 rows)
    - netmats2.txt (partial correlation matrices, 1003 rows)
    - subjectIDs.txt (subject ID per row)
    - behavioral data CSV

MODE B: Single Subject Preprocessed (for pipeline validation)
  Uses one subject's full preprocessed resting-state data to validate
  the extraction and computation pipeline before scaling.

  Required files:
    - Subject's rfMRI preprocessed directory from BALSA

Predictions tested:
  1. Resting Φ_PD varies across subjects (individual differences exist)
  2. Higher Φ_PD correlates with better cognitive performance on tasks
     requiring multi-domain integration
  3. Φ_PD from partial correlations (which better capture unique
     contributions) outperforms full correlations

Usage:
  # Mode A: PTN netmats
  python pmd_hcp_analysis.py --mode ptn --ptn-dir ./HCP_PTN1200 --behavioral ./behavioral.csv

  # Mode B: Single subject validation
  python pmd_hcp_analysis.py --mode single --subject-dir ./100307 --data-dir ./psychedelic_data

==========================================================================
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
from scipy import stats
from scipy.linalg import eigvalsh
from pathlib import Path

warnings.filterwarnings('ignore')


# =========================================================================
# ICA COMPONENT TO YEO-7 NETWORK MAPPING
# =========================================================================

# The HCP PTN uses group-ICA at d=100 (100 components).
# Each ICA component needs to be mapped to a Yeo-7 network.
# This mapping is typically done by spatial correlation between
# ICA maps and the Yeo atlas. For the HCP d100 decomposition,
# the mapping has been published. We use the standard assignment.
#
# If using the actual ICA spatial maps, we can compute this mapping
# automatically. For now, we provide a function that does it from
# the melodic_IC file if available, or uses a published mapping.

def load_ica_to_yeo_mapping(melodic_ic_path=None, n_components=100):
    """
    Map ICA components to Yeo-7 networks.
    
    If melodic_IC file is available, compute spatial overlap.
    Otherwise, return None and we'll use a data-driven approach
    from the correlation structure itself.
    """
    if melodic_ic_path and os.path.exists(melodic_ic_path):
        try:
            from nilearn import image, datasets
            import nibabel as nib
            
            # Load Yeo atlas
            yeo = datasets.fetch_atlas_yeo_2011(n_networks=7)
            if hasattr(yeo, 'maps'):
                yeo_img = image.load_img(yeo.maps)
            else:
                yeo_img = image.load_img(yeo['thick_7'])
            
            yeo_data = yeo_img.get_fdata()
            
            # Load ICA maps
            ica_img = nib.load(melodic_ic_path)
            ica_data = ica_img.get_fdata()
            
            mapping = {}
            for comp in range(min(n_components, ica_data.shape[-1])):
                comp_map = ica_data[..., comp]
                
                # For each Yeo network, compute spatial overlap
                best_network = 0
                best_overlap = 0
                for net_id in range(1, 8):
                    net_mask = (yeo_data == net_id)
                    overlap = np.sum(np.abs(comp_map[net_mask]))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_network = net_id
                
                mapping[comp] = best_network
            
            return mapping
        except Exception as e:
            print(f"  Warning: Could not compute ICA-Yeo mapping: {e}")
            return None
    
    return None


def map_netmat_to_yeo7(netmat_100x100, ica_yeo_mapping=None):
    """
    Aggregate a 100x100 ICA connectivity matrix to 7x7 Yeo network matrix.
    
    If no ICA-Yeo mapping is available, use a data-driven clustering approach:
    group the 100 components into 7 clusters based on their connectivity profiles.
    """
    if ica_yeo_mapping is not None:
        # Use the spatial mapping
        yeo_labels = ['Visual', 'Somatomotor', 'DorsalAttention',
                      'VentralAttention', 'Limbic', 'Frontoparietal', 'Default']
        
        yeo7_matrix = np.zeros((7, 7))
        yeo7_counts = np.zeros((7, 7))
        
        for i in range(100):
            for j in range(100):
                if i == j:
                    continue
                net_i = ica_yeo_mapping.get(i, 0) - 1  # 0-indexed
                net_j = ica_yeo_mapping.get(j, 0) - 1
                if 0 <= net_i < 7 and 0 <= net_j < 7:
                    yeo7_matrix[net_i, net_j] += netmat_100x100[i, j]
                    yeo7_counts[net_i, net_j] += 1
        
        # Average
        yeo7_counts[yeo7_counts == 0] = 1
        yeo7_matrix /= yeo7_counts
        
        return yeo7_matrix, yeo_labels
    else:
        # Data-driven: use spectral clustering on the connectivity matrix
        from scipy.cluster.hierarchy import fcluster, linkage
        
        # Use connectivity profiles as features
        # Each component's row in the matrix = its connectivity profile
        profiles = netmat_100x100.copy()
        np.fill_diagonal(profiles, 0)
        
        # Cluster into 7 groups
        Z = linkage(profiles, method='ward')
        clusters = fcluster(Z, t=7, criterion='maxclust')
        
        # Aggregate to cluster-level matrix
        n_clusters = 7
        cluster_matrix = np.zeros((n_clusters, n_clusters))
        cluster_counts = np.zeros((n_clusters, n_clusters))
        
        for i in range(100):
            for j in range(100):
                if i == j:
                    continue
                ci = clusters[i] - 1
                cj = clusters[j] - 1
                cluster_matrix[ci, cj] += netmat_100x100[i, j]
                cluster_counts[ci, cj] += 1
        
        cluster_counts[cluster_counts == 0] = 1
        cluster_matrix /= cluster_counts
        
        labels = [f"Cluster_{i+1}" for i in range(7)]
        return cluster_matrix, labels


# =========================================================================
# Φ_PD COMPUTATION
# =========================================================================

def compute_phi_pd_from_matrix(conn_matrix, labels, use_abs=True):
    """
    Compute Φ_PD from a pre-computed connectivity matrix.
    
    conn_matrix: NxN connectivity matrix (correlation or partial correlation)
    labels: network/cluster labels
    use_abs: if True, use |conn| as integration proxy
    """
    N = conn_matrix.shape[0]
    
    # Clean the matrix
    conn_matrix = np.nan_to_num(conn_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Integration: absolute connectivity
    if use_abs:
        I_matrix = np.abs(conn_matrix)
    else:
        I_matrix = conn_matrix.copy()
        I_matrix[I_matrix < 0] = 0  # Only positive connections as integration
    
    np.fill_diagonal(I_matrix, 0)
    
    # Conflict: higher for anticorrelated, lower for positively correlated
    C_matrix = np.zeros_like(conn_matrix)
    for i in range(N):
        for j in range(i+1, N):
            c = conn_matrix[i, j]
            if c < 0:
                C_matrix[i, j] = 1.0  # Maximum conflict
            else:
                C_matrix[i, j] = 1.0 - c
            C_matrix[j, i] = C_matrix[i, j]
    
    # Compute Φ_PD
    phi_pd = 0.0
    n_pairs = 0
    pair_contributions = {}
    
    for i in range(N):
        for j in range(i+1, N):
            contrib = C_matrix[i, j] * I_matrix[i, j]
            phi_pd += contrib
            n_pairs += 1
            if labels:
                pair_contributions[(labels[i], labels[j])] = contrib
    
    phi_pd_norm = (2.0 / (N * (N - 1))) * phi_pd if n_pairs > 0 else 0
    
    # Eigenanalysis
    J_matrix = np.nan_to_num(I_matrix * C_matrix, nan=0.0)
    try:
        eigenvalues = eigvalsh(J_matrix)
        lambda_max = np.max(eigenvalues)
    except:
        lambda_max = 0.0
    
    # Summary stats
    mean_conn = np.mean(np.abs(conn_matrix[np.triu_indices(N, k=1)]))
    anticorr_vals = conn_matrix[np.triu_indices(N, k=1)]
    n_anticorr = np.sum(anticorr_vals < 0)
    
    return {
        'phi_pd': phi_pd,
        'phi_pd_norm': phi_pd_norm,
        'lambda_max': lambda_max,
        'mean_connectivity': mean_conn,
        'n_anticorrelated_pairs': int(n_anticorr),
        'n_pairs': n_pairs,
        'pair_contributions': pair_contributions,
    }


# =========================================================================
# MODE A: PTN NETMATS ANALYSIS
# =========================================================================

def load_ptn_data(ptn_dir):
    """Load PTN netmats files."""
    
    # Find the netmats files
    netmats1_path = None
    netmats2_path = None
    subjects_path = None
    
    # Search common locations
    for root, dirs, files in os.walk(ptn_dir):
        for f in files:
            fpath = os.path.join(root, f)
            if f == 'netmats1.txt' or f == 'netmats1_correlationZ.txt':
                netmats1_path = fpath
            elif f == 'netmats2.txt' or f == 'netmats2_partial-correlation.txt':
                netmats2_path = fpath
            elif f == 'subjectIDs.txt':
                subjects_path = fpath
    
    if netmats1_path is None:
        # Try to find any netmats file
        for root, dirs, files in os.walk(ptn_dir):
            for f in files:
                if 'netmats' in f.lower() and f.endswith('.txt'):
                    print(f"  Found: {os.path.join(root, f)}")
    
    data = {}
    
    if netmats1_path:
        print(f"  Loading netmats1 (correlations): {netmats1_path}")
        data['netmats1'] = np.loadtxt(netmats1_path)
        print(f"    Shape: {data['netmats1'].shape}")
    
    if netmats2_path:
        print(f"  Loading netmats2 (partial correlations): {netmats2_path}")
        data['netmats2'] = np.loadtxt(netmats2_path)
        print(f"    Shape: {data['netmats2'].shape}")
    
    if subjects_path:
        print(f"  Loading subject IDs: {subjects_path}")
        with open(subjects_path) as f:
            data['subjects'] = [line.strip() for line in f if line.strip()]
        print(f"    {len(data['subjects'])} subjects")
    
    return data


def load_behavioral_data(behavioral_path):
    """Load HCP behavioral data CSV."""
    import pandas as pd
    
    if not os.path.exists(behavioral_path):
        print(f"  Behavioral data not found: {behavioral_path}")
        return None
    
    df = pd.read_csv(behavioral_path)
    print(f"  Loaded behavioral data: {df.shape[0]} subjects, {df.shape[1]} variables")
    
    # Key cognitive variables that map to multi-domain integration
    key_vars = [
        'CogTotalComp_AgeAdj',     # Total cognition composite
        'CogFluidComp_AgeAdj',     # Fluid cognition composite
        'CogCrystalComp_AgeAdj',   # Crystallized cognition composite
        'WM_Task_Acc',             # Working memory accuracy
        'Relational_Task_Acc',     # Relational reasoning accuracy
        'Social_Task_Perc_TOM',    # Social cognition (theory of mind)
        'Language_Task_Acc',       # Language task accuracy
        'ListSort_AgeAdj',        # List sorting (working memory)
        'CardSort_AgeAdj',        # Card sorting (executive function)
        'Flanker_AgeAdj',         # Flanker (inhibitory control)
        'ProcSpeed_AgeAdj',       # Processing speed
        'PMAT24_A_CR',            # Penn progressive matrices (fluid reasoning)
        'ReadEng_AgeAdj',         # Reading
        'PicVocab_AgeAdj',        # Picture vocabulary
    ]
    
    available = [v for v in key_vars if v in df.columns]
    print(f"  Key cognitive variables available: {len(available)}/{len(key_vars)}")
    for v in available:
        print(f"    {v}: {df[v].notna().sum()} subjects with data")
    
    return df


def run_ptn_analysis(ptn_dir, behavioral_path=None, output_dir='./pmd_results'):
    """Full PTN analysis pipeline."""
    
    print(f"\n{'='*75}")
    print("MODE A: PTN Netmats Analysis")
    print(f"{'='*75}")
    
    # Load data
    print("\n--- Loading PTN data ---")
    ptn = load_ptn_data(ptn_dir)
    
    if 'netmats1' not in ptn and 'netmats2' not in ptn:
        print("ERROR: No netmats files found")
        return
    
    # Load behavioral data
    behavioral_df = None
    if behavioral_path:
        print("\n--- Loading behavioral data ---")
        behavioral_df = load_behavioral_data(behavioral_path)
    
    # Determine matrix size
    netmats_key = 'netmats2' if 'netmats2' in ptn else 'netmats1'
    netmats = ptn[netmats_key]
    n_subjects = netmats.shape[0]
    n_elements = netmats.shape[1]
    n_nodes = int(np.sqrt(n_elements))
    
    print(f"\n--- Analysis configuration ---")
    print(f"  Subjects: {n_subjects}")
    print(f"  Nodes: {n_nodes}")
    print(f"  Primary netmats: {netmats_key}")
    
    # Try to load ICA-Yeo mapping
    melodic_path = None
    for root, dirs, files in os.walk(ptn_dir):
        for f in files:
            if 'melodic_IC' in f and f.endswith('.nii.gz'):
                melodic_path = os.path.join(root, f)
                break
    
    ica_yeo_mapping = load_ica_to_yeo_mapping(melodic_path, n_nodes)
    if ica_yeo_mapping:
        print(f"  ICA-Yeo mapping: computed from melodic_IC")
    else:
        print(f"  ICA-Yeo mapping: data-driven clustering")
    
    # =====================================================================
    # COMPUTE Φ_PD FOR ALL SUBJECTS
    # =====================================================================
    print(f"\n{'='*75}")
    print(f"Computing Φ_PD for {n_subjects} subjects...")
    print(f"{'='*75}")
    
    results = []
    
    for mat_name in ['netmats1', 'netmats2']:
        if mat_name not in ptn:
            continue
        
        print(f"\n--- {mat_name} ---")
        netmats = ptn[mat_name]
        
        phi_pds = []
        lambda_maxs = []
        mean_conns = []
        
        for i in range(n_subjects):
            # Reshape flattened row to NxN matrix
            conn_matrix = netmats[i].reshape(n_nodes, n_nodes)
            
            # Map to Yeo-7
            yeo7_matrix, yeo7_labels = map_netmat_to_yeo7(
                conn_matrix, ica_yeo_mapping)
            
            # Compute Φ_PD on the 7-network matrix
            result = compute_phi_pd_from_matrix(yeo7_matrix, yeo7_labels)
            
            phi_pds.append(result['phi_pd_norm'])
            lambda_maxs.append(result['lambda_max'])
            mean_conns.append(result['mean_connectivity'])
            
            if i % 100 == 0 and i > 0:
                print(f"    Processed {i}/{n_subjects}...")
        
        phi_pds = np.array(phi_pds)
        lambda_maxs = np.array(lambda_maxs)
        mean_conns = np.array(mean_conns)
        
        print(f"\n  Results ({mat_name}):")
        print(f"    Φ_PD: mean={np.mean(phi_pds):.6f}, std={np.std(phi_pds):.6f}")
        print(f"    λ_max: mean={np.mean(lambda_maxs):.6f}, std={np.std(lambda_maxs):.6f}")
        print(f"    Connectivity: mean={np.mean(mean_conns):.6f}, std={np.std(mean_conns):.6f}")
        print(f"    Φ_PD range: [{np.min(phi_pds):.6f}, {np.max(phi_pds):.6f}]")
        
        results.append({
            'matrix_type': mat_name,
            'phi_pds': phi_pds,
            'lambda_maxs': lambda_maxs,
            'mean_conns': mean_conns,
        })
    
    # =====================================================================
    # PREDICTION 1: Individual differences in Φ_PD exist
    # =====================================================================
    print(f"\n{'='*75}")
    print("PREDICTION 1: Individual differences in Φ_PD")
    print(f"{'='*75}")
    
    for r in results:
        cv = np.std(r['phi_pds']) / np.mean(r['phi_pds']) * 100
        print(f"\n  {r['matrix_type']}:")
        print(f"    Coefficient of variation: {cv:.1f}%")
        print(f"    Range: {np.min(r['phi_pds']):.6f} to {np.max(r['phi_pds']):.6f}")
        print(f"    Ratio max/min: {np.max(r['phi_pds'])/np.min(r['phi_pds']):.2f}x")
        
        if cv > 5:
            print(f"    ★ Meaningful individual differences exist (CV > 5%)")
        else:
            print(f"    Minimal individual differences (CV < 5%)")
    
    # =====================================================================
    # PREDICTION 2: Φ_PD correlates with cognitive performance
    # =====================================================================
    if behavioral_df is not None and 'subjects' in ptn:
        print(f"\n{'='*75}")
        print("PREDICTION 2: Φ_PD predicts cognitive performance")
        print(f"{'='*75}")
        
        subjects = ptn['subjects']
        
        # Key cognitive variables
        cog_vars = [
            ('CogTotalComp_AgeAdj', 'Total Cognition'),
            ('CogFluidComp_AgeAdj', 'Fluid Cognition'),
            ('CogCrystalComp_AgeAdj', 'Crystallized Cognition'),
            ('PMAT24_A_CR', 'Matrix Reasoning'),
            ('ListSort_AgeAdj', 'Working Memory (List Sort)'),
            ('CardSort_AgeAdj', 'Executive Function (Card Sort)'),
            ('Flanker_AgeAdj', 'Inhibitory Control (Flanker)'),
            ('ProcSpeed_AgeAdj', 'Processing Speed'),
            ('ReadEng_AgeAdj', 'Reading'),
            ('PicVocab_AgeAdj', 'Vocabulary'),
        ]
        
        for r in results:
            print(f"\n--- {r['matrix_type']} ---")
            print(f"{'Variable':<40} {'r':>8} {'p':>10} {'n':>6} {'sig':>5}")
            print("-" * 75)
            
            sig_count = 0
            
            for var_name, var_label in cog_vars:
                if var_name not in behavioral_df.columns:
                    continue
                
                # Match subjects
                phi_vals = []
                cog_vals = []
                
                for idx, sub_id in enumerate(subjects):
                    if idx >= len(r['phi_pds']):
                        break
                    
                    # Find this subject in behavioral data
                    sub_match = behavioral_df[behavioral_df['Subject'].astype(str) == str(sub_id)]
                    if len(sub_match) == 0:
                        continue
                    
                    cog_val = sub_match[var_name].values[0]
                    if np.isnan(cog_val):
                        continue
                    
                    phi_vals.append(r['phi_pds'][idx])
                    cog_vals.append(cog_val)
                
                if len(phi_vals) < 10:
                    continue
                
                phi_arr = np.array(phi_vals)
                cog_arr = np.array(cog_vals)
                
                corr_r, corr_p = stats.pearsonr(phi_arr, cog_arr)
                sig = "***" if corr_p < 0.001 else "**" if corr_p < 0.01 else "*" if corr_p < 0.05 else "ns"
                
                if corr_p < 0.05:
                    sig_count += 1
                
                print(f"{var_label:<40} {corr_r:>8.4f} {corr_p:>10.6f} {len(phi_vals):>6} {sig:>5}")
            
            print(f"\n  Significant correlations: {sig_count}")
            if sig_count >= 3:
                print(f"  ★ Φ_PD significantly predicts multiple cognitive measures")
            elif sig_count > 0:
                print(f"  Partial support: some cognitive correlations significant")
            else:
                print(f"  No significant cognitive correlations found")
    
    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    os.makedirs(output_dir, exist_ok=True)
    
    for r in results:
        # Save per-subject Φ_PD values
        out_path = os.path.join(output_dir, f"hcp_{r['matrix_type']}_phi_pd.csv")
        header = "subject_id,phi_pd_norm,lambda_max,mean_connectivity"
        subjects = ptn.get('subjects', [str(i) for i in range(len(r['phi_pds']))])
        
        with open(out_path, 'w') as f:
            f.write(header + '\n')
            for i in range(len(r['phi_pds'])):
                sub = subjects[i] if i < len(subjects) else str(i)
                f.write(f"{sub},{r['phi_pds'][i]:.6f},{r['lambda_maxs'][i]:.6f},{r['mean_conns'][i]:.6f}\n")
        
        print(f"\n  Saved: {out_path}")
    
    print(f"\n{'='*75}")
    print("HCP ANALYSIS COMPLETE")
    print(f"{'='*75}")


# =========================================================================
# MODE B: SINGLE SUBJECT VALIDATION
# =========================================================================

def run_single_subject(subject_dir, output_dir='./pmd_results'):
    """Process a single subject's preprocessed resting-state data."""
    
    print(f"\n{'='*75}")
    print("MODE B: Single Subject Validation")
    print(f"{'='*75}")
    
    # Find resting-state files
    rest_files = []
    for root, dirs, files in os.walk(subject_dir):
        for f in files:
            if 'rfMRI_REST' in f and f.endswith('_Atlas_MSMAll.dtseries.nii'):
                rest_files.append(os.path.join(root, f))
            elif 'rfMRI_REST' in f and f.endswith('_Atlas.dtseries.nii'):
                rest_files.append(os.path.join(root, f))
            elif 'rfMRI_REST' in f and f.endswith('_hp2000_clean.nii.gz'):
                rest_files.append(os.path.join(root, f))
    
    # Also find task files
    task_files = {}
    task_names = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
    for root, dirs, files in os.walk(subject_dir):
        for f in files:
            for task in task_names:
                if f'tfMRI_{task}' in f and (f.endswith('_Atlas_MSMAll.dtseries.nii') or 
                                               f.endswith('_hp200_s2_level2.feat') or
                                               f.endswith('_hp2000_clean.nii.gz')):
                    if task not in task_files:
                        task_files[task] = []
                    task_files[task].append(os.path.join(root, f))
    
    print(f"\n  Resting-state files found: {len(rest_files)}")
    for f in rest_files:
        print(f"    {os.path.basename(f)}")
    
    print(f"\n  Task files found:")
    for task, files in task_files.items():
        print(f"    {task}: {len(files)} files")
    
    if not rest_files:
        # Try finding any NIfTI files
        print("\n  Searching for any NIfTI files...")
        for root, dirs, files in os.walk(subject_dir):
            for f in files:
                if f.endswith('.nii.gz') and 'rfMRI' in f:
                    print(f"    Found: {os.path.join(root, f)}")
                    rest_files.append(os.path.join(root, f))
    
    if not rest_files:
        print("\n  No resting-state files found.")
        print("  Looking for any processable files...")
        nifti_count = 0
        for root, dirs, files in os.walk(subject_dir):
            for f in files:
                if f.endswith('.nii.gz'):
                    size = os.path.getsize(os.path.join(root, f))
                    if size > 10*1024*1024:  # > 10 MB
                        print(f"    {os.path.relpath(os.path.join(root, f), subject_dir)} ({size/1024/1024:.0f} MB)")
                        nifti_count += 1
                        if nifti_count >= 20:
                            print("    ... (truncated)")
                            break
        return
    
    # Process resting-state with Yeo-7
    print(f"\n--- Processing resting-state data ---")
    
    from nilearn import datasets, maskers, image
    
    yeo = datasets.fetch_atlas_yeo_2011(n_networks=7)
    if hasattr(yeo, 'maps'):
        atlas_img = yeo.maps
    else:
        atlas_img = yeo['thick_7']
    
    yeo_labels = ["Visual", "Somatomotor", "DorsalAttention",
                  "VentralAttention", "Limbic", "Frontoparietal", "Default"]
    
    for rest_file in rest_files[:4]:  # Process up to 4 runs
        print(f"\n  Processing: {os.path.basename(rest_file)}")
        
        # Check if it's a CIFTI file (dtseries) or NIfTI
        if rest_file.endswith('.dtseries.nii'):
            print("    CIFTI format — requires Connectome Workbench for extraction")
            print("    Skipping CIFTI for now (NIfTI pipeline only)")
            continue
        
        masker = maskers.NiftiLabelsMasker(
            labels_img=atlas_img,
            standardize=True,
            detrend=True,
            low_pass=0.08,
            high_pass=0.01,
            t_r=0.72,  # HCP TR
            memory='nilearn_cache',
            verbose=0
        )
        
        try:
            timeseries = masker.fit_transform(rest_file)
            print(f"    Extracted: {timeseries.shape}")
            
            # Handle zero-variance
            for col in range(timeseries.shape[1]):
                if np.std(timeseries[:, col]) < 1e-10:
                    timeseries[:, col] = np.random.randn(timeseries.shape[0]) * 1e-6
            
            # Compute correlation matrix
            corr = np.corrcoef(timeseries.T)
            corr = np.nan_to_num(corr, nan=0.0)
            
            # Compute Φ_PD
            result = compute_phi_pd_from_matrix(corr, yeo_labels)
            
            print(f"    Φ_PD_norm: {result['phi_pd_norm']:.6f}")
            print(f"    λ_max:     {result['lambda_max']:.6f}")
            print(f"    Mean conn: {result['mean_connectivity']:.6f}")
            print(f"    Anticorr pairs: {result['n_anticorrelated_pairs']}")
            
            # Print pair contributions
            print(f"\n    Pair contributions:")
            sorted_pairs = sorted(result['pair_contributions'].items(),
                                  key=lambda x: x[1], reverse=True)
            for (a, b), contrib in sorted_pairs:
                print(f"      {a:20s} - {b:20s}: {contrib:.6f}")
            
        except Exception as e:
            print(f"    Error: {e}")
    
    print(f"\n{'='*75}")
    print("SINGLE SUBJECT VALIDATION COMPLETE")
    print(f"{'='*75}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="PMD HCP Analysis Pipeline")
    parser.add_argument('--mode', choices=['ptn', 'single'], required=True,
                       help='Analysis mode: ptn (netmats) or single (one subject)')
    parser.add_argument('--ptn-dir', type=str, help='Directory containing PTN files')
    parser.add_argument('--subject-dir', type=str, help='Single subject data directory')
    parser.add_argument('--behavioral', type=str, help='Behavioral data CSV path')
    parser.add_argument('--output-dir', type=str, default='./pmd_results')
    args = parser.parse_args()
    
    print("=" * 75)
    print("PRESSURE MAKES DIAMONDS")
    print("HCP Analysis Pipeline")
    print("=" * 75)
    
    if args.mode == 'ptn':
        if not args.ptn_dir:
            print("ERROR: --ptn-dir required for PTN mode")
            return
        run_ptn_analysis(args.ptn_dir, args.behavioral, args.output_dir)
    
    elif args.mode == 'single':
        if not args.subject_dir:
            print("ERROR: --subject-dir required for single mode")
            return
        run_single_subject(args.subject_dir, args.output_dir)


if __name__ == "__main__":
    main()
