#!/usr/bin/env python3
from pathlib import Path

MP4_DIR = Path("/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/mp4")
WAV_DIR = Path("/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/wav")

print("Finding MP4 files without corresponding WAV files...\n")
missing = []

for mp4_file in MP4_DIR.rglob("*.mp4"):
    wav_file = WAV_DIR / mp4_file.relative_to(MP4_DIR).with_suffix('.wav')
    if not wav_file.exists():
        missing.append(mp4_file)
        print(mp4_file)

print(f"\nTotal missing WAV files: {len(missing)}")

# Save to file
if missing:
    with open("missing_wavs.txt", "w") as f:
        for mp4 in missing:
            f.write(f"{mp4}\n")
    print(f"Saved list to missing_wavs.txt")
    
    # Ask if user wants to delete the MP4s
    response = input(f"\nDelete these {len(missing)} MP4 files that have no WAV? (yes/no): ")
    if response.lower() == 'yes':
        for mp4_file in missing:
            mp4_file.unlink()
            print(f"Deleted: {mp4_file}")
        print(f"\nDeleted {len(missing)} MP4 files")