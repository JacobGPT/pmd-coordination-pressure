#!/usr/bin/env python3
"""
===========================================================================
PMD CONSCIOUSNESS STATE PREDICTION — ds006623

THE DECISIVE EXPERIMENT:
Does Φ*_PD track the conscious-to-unconscious transition?

DATASET: ds006623 (Michigan Human Anesthesia fMRI Dataset-1)
  - 26 healthy volunteers
  - Mental imagery tasks (tennis, navigation, hand squeeze)
  - Motor response task (actual hand squeeze)
  - Graded propofol sedation: Baseline → PreLOR → LOR → Recovery
  - BIDS formatted, fMRIPrep preprocessed

APPROACH:
  Step 1: Download dataset from OpenNeuro
  Step 2: For each subject at each sedation level:
    - Extract multi-voxel patterns during task periods
    - Decode task conditions (imagery vs rest, or motor vs rest)
    - Compute per-trial Φ*_PD using Level 3 pipeline
    - Compute session-level Φ*_PD mean
  Step 3: Compare Φ*_PD across consciousness states:
    - Baseline (awake) vs LOR (unconscious) vs Recovery
    - Test for phase transition signature (sharp drop, not linear decline)

PMD PREDICTIONS:
  1. Φ*_PD(Baseline) > Φ*_PD(LOR) — coordination pressure drops with consciousness
  2. Φ*_PD(Recovery) ≈ Φ*_PD(Baseline) — consciousness restoration restores pressure
  3. The drop is SHARP at LOR, not gradual — phase transition signature
  4. Gated metric works; ungated metric misorders states (as in Level 2A result)
  5. Eigenvector structure should dissolve at LOR (rank-1 dominance drops)

USAGE:
  python pmd_consciousness_states.py --download   # Download dataset first
  python pmd_consciousness_states.py --analyze    # Run analysis
  python pmd_consciousness_states.py --all        # Both
===========================================================================
"""

import os, sys, json, warnings, argparse, subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')

DATASET_ID = "ds006623"
DATA_DIR = Path(f"./{DATASET_ID}")


def download_dataset():
    """Download ds006623 from OpenNeuro using DataLad or direct download."""
    print("="*80)
    print(f"DOWNLOADING {DATASET_ID} FROM OPENNEURO")
    print("="*80)
    
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        print(f"\n  {DATA_DIR} already exists. Skipping download.")
        print(f"  Delete the folder to re-download.")
        return True
    
    # Try openneuro-py first (fastest)
    print("\n  Attempting download with openneuro-py...")
    try:
        result = subprocess.run(
            ["openneuro-py", "download", "--dataset", DATASET_ID, "--target_dir", str(DATA_DIR)],
            capture_output=True, text=True, timeout=3600
        )
        if result.returncode == 0:
            print(f"  ✓ Download complete: {DATA_DIR}")
            return True
        else:
            print(f"  openneuro-py failed: {result.stderr[:200]}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  openneuro-py not found or timed out.")
    
    # Try datalad
    print("\n  Attempting download with datalad...")
    try:
        result = subprocess.run(
            ["datalad", "install", f"https://github.com/OpenNeuroDatasets/{DATASET_ID}.git"],
            capture_output=True, text=True, timeout=3600
        )
        if result.returncode == 0:
            print(f"  ✓ Datalad clone complete. Getting data...")
            subprocess.run(["datalad", "get", "-d", DATASET_ID, "-r", "."], timeout=7200)
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  datalad not found or timed out.")
    
    # Try AWS CLI (OpenNeuro is on S3)
    print("\n  Attempting download via AWS S3...")
    try:
        result = subprocess.run(
            ["aws", "s3", "sync", "--no-sign-request",
             f"s3://openneuro.org/{DATASET_ID}/", str(DATA_DIR) + "/"],
            capture_output=True, text=True, timeout=7200
        )
        if result.returncode == 0:
            print(f"  ✓ S3 download complete: {DATA_DIR}")
            return True
        else:
            print(f"  AWS CLI failed: {result.stderr[:200]}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  AWS CLI not found or timed out.")
    
    # Manual instructions
    print(f"""
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AUTOMATIC DOWNLOAD FAILED
  
  Please download manually:
  
  Option A (recommended):
    pip install openneuro-py
    openneuro-py download --dataset {DATASET_ID} --target_dir {DATA_DIR}
  
  Option B:
    pip install datalad
    datalad install https://github.com/OpenNeuroDatasets/{DATASET_ID}.git
    cd {DATASET_ID}
    datalad get -r .
  
  Option C (web browser):
    Go to https://openneuro.org/datasets/{DATASET_ID}
    Click "Download" 
    Extract to ./{DATASET_ID}/
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
    return False


def explore_dataset():
    """Explore the downloaded dataset structure."""
    print(f"\n{'='*80}")
    print(f"EXPLORING DATASET STRUCTURE")
    print("="*80)
    
    if not DATA_DIR.exists():
        print(f"  Dataset not found at {DATA_DIR}")
        return None
    
    # Find subjects
    subjects = sorted([d.name for d in DATA_DIR.iterdir() 
                       if d.is_dir() and d.name.startswith('sub-')])
    print(f"\n  Found {len(subjects)} subjects: {subjects[:5]}...")
    
    # Check first subject's structure
    if subjects:
        sub = subjects[0]
        sub_dir = DATA_DIR / sub
        print(f"\n  Structure of {sub}:")
        
        # List sessions
        sessions = sorted([d.name for d in sub_dir.iterdir() if d.is_dir()])
        for ses in sessions:
            ses_dir = sub_dir / ses
            print(f"    {ses}/")
            
            # Check func
            func_dir = ses_dir / "func"
            if func_dir.exists():
                files = sorted(func_dir.iterdir())
                for f in files[:10]:
                    size = f.stat().st_size / (1024*1024) if f.is_file() else 0
                    print(f"      {f.name} ({size:.1f} MB)" if size > 0 else f"      {f.name}")
                if len(files) > 10:
                    print(f"      ... and {len(files)-10} more files")
            
            # Check anat
            anat_dir = ses_dir / "anat"
            if anat_dir.exists():
                print(f"      [anat/] {len(list(anat_dir.iterdir()))} files")
        
        # Check for events files
        print(f"\n  Looking for events files...")
        events_files = list(DATA_DIR.rglob("*events.tsv"))
        if events_files:
            print(f"  Found {len(events_files)} events files")
            # Show first one
            ef = events_files[0]
            print(f"  Example: {ef.relative_to(DATA_DIR)}")
            try:
                df = pd.read_csv(ef, sep='\t', nrows=5)
                print(f"  Columns: {df.columns.tolist()}")
                print(f"  First rows:")
                print(df.to_string(index=False))
            except Exception as e:
                print(f"  Error reading: {e}")
        else:
            print("  No events files found — checking for task timing in JSON...")
            json_files = list(DATA_DIR.rglob("*bold.json"))
            if json_files:
                jf = json_files[0]
                print(f"  Example JSON: {jf.relative_to(DATA_DIR)}")
                with open(jf) as f:
                    jdata = json.load(f)
                    for key in ['TaskName', 'RepetitionTime', 'TaskDescription']:
                        if key in jdata:
                            print(f"    {key}: {jdata[key]}")
        
        # Check for derivatives (fMRIPrep)
        deriv_dir = DATA_DIR / "derivatives"
        if deriv_dir.exists():
            print(f"\n  Derivatives found:")
            for d in sorted(deriv_dir.iterdir()):
                print(f"    {d.name}/")
        
        # Check for participant info
        for fname in ['participants.tsv', 'participants.json']:
            pf = DATA_DIR / fname
            if pf.exists():
                print(f"\n  {fname}:")
                if fname.endswith('.tsv'):
                    df = pd.read_csv(pf, sep='\t')
                    print(f"  Columns: {df.columns.tolist()}")
                    print(f"  N = {len(df)}")
                    print(df.head().to_string())
    
    return subjects


def main():
    parser = argparse.ArgumentParser(description='PMD Consciousness States Analysis')
    parser.add_argument('--download', action='store_true', help='Download dataset')
    parser.add_argument('--analyze', action='store_true', help='Run analysis')
    parser.add_argument('--all', action='store_true', help='Download + analyze')
    parser.add_argument('--explore', action='store_true', help='Explore dataset structure')
    args = parser.parse_args()
    
    if not any([args.download, args.analyze, args.all, args.explore]):
        args.explore = True  # Default: just explore
        args.download = True
    
    print("="*80)
    print("PRESSURE MAKES DIAMONDS — CONSCIOUSNESS STATE PREDICTION")
    print("The decisive experiment: does Φ*_PD track consciousness transitions?")
    print("="*80)
    print(f"\n  Dataset: {DATASET_ID} (Michigan Human Anesthesia fMRI)")
    print(f"  26 subjects, mental imagery + motor tasks under propofol sedation")
    print(f"  States: Baseline → Pre-LOR → LOR → Recovery")
    
    if args.download or args.all:
        success = download_dataset()
        if not success and not DATA_DIR.exists():
            print("\n  Download failed. Please download manually and re-run with --analyze")
            return
    
    if args.explore or args.download or args.all:
        subjects = explore_dataset()
    
    if args.analyze or args.all:
        print(f"\n{'='*80}")
        print("ANALYSIS")
        print("="*80)
        
        if not DATA_DIR.exists():
            print(f"  Dataset not found at {DATA_DIR}. Run with --download first.")
            return
        
        # The actual Level 3 analysis pipeline will be built after we see
        # the exact structure of the data (session names, task names, 
        # event file format, etc.)
        print("""
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  NEXT STEP: After downloading and exploring the data structure,
  send me the output of --explore and I will build the exact
  Level 3 pipeline adapted to this dataset's format.
  
  The key information I need:
    1. Session names (ses-baseline, ses-sedation, etc.)
    2. Task names in filenames (task-tennis, task-motor, etc.)
    3. Events file format (columns, trial types)
    4. Whether fMRIPrep derivatives are included
    5. Number of usable subjects per condition
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


if __name__ == "__main__":
    main()
