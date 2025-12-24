#!/usr/bin/env python3
from pathlib import Path
import csv
import os
from tqdm import tqdm

# --- CONFIGURATION ---
# Update these paths if necessary
DATASET_ROOT = Path("/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/multich")
MIXED_DIR = DATASET_ROOT / "mixed"
CLEAN_DIR = DATASET_ROOT / "clean"
META_FILE = DATASET_ROOT / "metadata.csv"

def check_full_integrity():
    print(f"---  Checking Full Dataset Integrity ---\n")

    if not META_FILE.exists():
        print(" CRITICAL: metadata.csv not found!")
        return

    # Store valid and broken entries
    valid_rows = []
    broken_entries = []
    
    # Track missing file counts
    missing_mixed_count = 0
    missing_clean_count = 0
    missing_source_mp4_count = 0

    print("Reading metadata and verifying files...")
    
    with open(META_FILE, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

        for row in tqdm(rows):
            is_healthy = True
            
            # 1. Check Generated Files (Mixed & Clean)
            filename = row['filename'] 
            mixed_path = MIXED_DIR / f"{filename}.wav"
            clean_path = CLEAN_DIR / f"{filename}.wav"

            if not mixed_path.exists():
                missing_mixed_count += 1
                is_healthy = False
            
            if not clean_path.exists():
                missing_clean_count += 1
                is_healthy = False

            # 2. Check Source Video (Absolute path from CSV)
            video_path_str = row['video_path']
            video_path = Path(video_path_str)

            if not video_path.exists():
                missing_source_mp4_count += 1
                is_healthy = False

            # Decision
            if is_healthy:
                valid_rows.append(row)
            else:
                broken_entries.append(filename)

    # --- REPORTING ---
    print("\n" + "="*40)
    print(" INTEGRITY REPORT")
    print("="*40)
    print(f"Total Samples in Metadata: {len(rows)}")
    print(f"Healthy Samples:        {len(valid_rows)}")
    print(f"Broken Samples:         {len(broken_entries)}")
    print("-" * 20)
    print(f"Missing Mixed Files:       {missing_mixed_count}")
    print(f"Missing Clean Files:       {missing_clean_count}")
    print(f"Missing Source MP4s:       {missing_source_mp4_count}")
    print("="*40)

    # --- CLEANUP ACTION ---
    if broken_entries:
        print(f"\nFound {len(broken_entries)} broken samples.")
        print("Recommended Action: Cleanup metadata and delete partial files.")
        choice = input("Apply Fix? (yes/no): ").lower()

        if choice == "yes":
            # 1. Rewrite Metadata
            print("Rewriting metadata.csv...")
            with open(META_FILE, "w", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(valid_rows)
            
            # 2. Delete Orphaned Generated Files
            print("Cleaning up generated orphans...")
            removed_files = 0
            for sample_name in broken_entries:
                m_path = MIXED_DIR / f"{sample_name}.wav"
                c_path = CLEAN_DIR / f"{sample_name}.wav"
                
                if m_path.exists():
                    m_path.unlink()
                    removed_files += 1
                if c_path.exists():
                    c_path.unlink()
                    removed_files += 1
            
            print(f"Done. Removed {removed_files} orphaned files.")
            print("Dataset is now clean.")
    else:
        print("\nDataset is 100% Valid.")

if __name__ == "__main__":
    check_full_integrity()