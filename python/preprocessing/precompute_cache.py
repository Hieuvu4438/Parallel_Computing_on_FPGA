#!/usr/bin/env python3
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import python.training.icbhi_kd_pipeline_multiview_ensemble as base

def precompute_for_config(f_max, name):
    print(f"\n=== Precomputing Spectrogram Cache for Config: {name} (f_max={f_max}) ===")
    
    # Get base args and override
    args = base.parse_args()
    args.f_max = f_max
    args.stage = "evaluate"  # Avoid running training
    args.rebuild_splits = False
    
    # Prepare runs to get splits and stats
    output_dir, splits, stats = base.prepare_run(args)
    
    # Collect all records
    all_records = []
    for split_name, records in splits.items():
        if records:
            all_records.extend(records)
            
    # De-duplicate records
    seen = set()
    unique_records = []
    for r in all_records:
        if r.sample_id not in seen:
            seen.add(r.sample_id)
            unique_records.append(r)
            
    print(f"Total unique cycles to process: {len(unique_records)}")
    ds = base.ICBHIDataset(unique_records, args, None, False)
    
    # Run sequentially (audio files will be cached in ds._cache to speed up consecutive cycles)
    for idx in tqdm(range(len(ds)), desc=f"Config {name}"):
        _ = ds[idx]
        
    print(f"Done caching for {name}. Cache directory: {ds.cache_dir}")

def main():
    # E1/E2 config uses f_max = 4000.0
    precompute_for_config(4000.0, "E1_E2")
    
    # E3 config uses f_max = 2500.0
    precompute_for_config(2500.0, "E3")

if __name__ == "__main__":
    main()
