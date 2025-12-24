import os
import subprocess
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# CONFIGURATION
INPUT_ROOT = Path("/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/mp4")
OUTPUT_ROOT = Path("/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/wav")
TARGET_SR = 16000
# Use almost all cores, leaving 2 free for OS tasks
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 2)

def process_single_file(mp4_path):
    """
    Worker function to process a single file.
    """
    try:
        # Calculate output path
        # dev/mp4/id00017/video/001.mp4 -> dev/wav/id00017/video/001.wav
        relative_path = mp4_path.relative_to(INPUT_ROOT)
        wav_path = OUTPUT_ROOT / relative_path.with_suffix(".wav")
        
        # Check if already done
        if wav_path.exists():
            return 0  # 0 indicates "skipped/already done"

        # Create directory (safe for multiprocessing)
        wav_path.parent.mkdir(parents=True, exist_ok=True)

        # FFMPEG Command
        cmd = [
            "ffmpeg", 
            "-hide_banner", "-loglevel", "error", "-y",  # -y overwrites without asking
            "-i", str(mp4_path),
            "-vn",             # No video
            "-ac", "1",        # Mono
            "-ar", str(TARGET_SR), 
            "-f", "wav",
            str(wav_path)
        ]
        
        subprocess.run(cmd, check=True)
        return 1  # 1 indicates "processed successfully"

    except Exception as e:
        # If a file is corrupt, print it but don't crash the whole script
        print(f"\nError processing {mp4_path}: {e}")
        return 0

def convert_dataset_parallel():
    # 1. Scan files
    print(f"Scanning {INPUT_ROOT}...")
    files = list(INPUT_ROOT.rglob("*.mp4"))
    total_files = len(files)
    print(f"Found {total_files} files. Launching {MAX_WORKERS} workers...")

    # 2. Parallel Execution
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all jobs
        futures = [executor.submit(process_single_file, f) for f in files]
        
        # Track progress
        # as_completed yields futures as they finish, updating the bar smoothly
        for _ in tqdm(as_completed(futures), total=total_files, unit="files"):
            pass

if __name__ == "__main__":
    convert_dataset_parallel()