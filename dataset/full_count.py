#!/usr/bin/env python3
from pathlib import Path
import csv
import os
from tqdm import tqdm

# --- CONFIGURATION ---
# Root folders
DATASET_ROOT = Path("/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/multich")
SOURCE_MP4_ROOT = Path("/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/mp4")
SOURCE_WAV_ROOT = Path("/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/wav")

# Subfolders
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
    missing_source_wav_count = 0

    print("Reading metadata and verifying files...")
    
    with open(META_FILE, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

        for row in tqdm(rows):
            is_healthy = True
            
            # 1. Check Generated Files (Mixed & Clean)
            filename = row['filename'] # e.g., sample_00000
            mixed_path = MIXED_DIR / f"{filename}.wav"
            clean_path = CLEAN_DIR / f"{filename}.wav"

            if not mixed_path.exists():
                missing_mixed_count += 1
                is_healthy = False
            
            if not clean_path.exists():
                missing_clean_count += 1
                is_healthy = False

            # 2. Check Source Files (Video & Audio)
            # The metadata contains the full absolute path to the MP4
            video_path_str = row['video_path']
            video_path = Path(video_path_str)
            
            # Derive the WAV path (swap 'mp4' folder/ext for 'wav')
            # Logic: .../dev/mp4/id0001/vid.mp4 -> .../dev/wav/id0001/vid.wav
            wav_path_str = video_path_str.replace("/mp4/", "/wav/").replace(".mp4", ".wav")
            wav_path = Path(wav_path_str)

            if not video_path.exists():
                missing_source_mp4_count += 1
                is_healthy = False
            
            if not wav_path.exists():
                missing_source_wav_count += 1
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
    print(f" Healthy Samples:        {len(valid_rows)}")
    print(f" Broken Samples:         {len(broken_entries)}")
    print("-" * 20)
    print(f"Missing Mixed Files:       {missing_mixed_count}")
    print(f"Missing Clean Files:       {missing_clean_count}")
    print(f"Missing Source MP4s:       {missing_source_mp4_count}")
    print(f"Missing Source WAVs:       {missing_source_wav_count}")
    print("="*40)

    # --- CLEANUP ACTION ---
    if broken_entries:
        print(f"\nFound {len(broken_entries)} broken samples.")
        print("Recommended Action: Remove these entries from metadata and delete partial generated files.")
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
        print("\nDataset is 100% Valid. You are ready to train.")

if __name__ == "__main__":
    check_full_integrity()