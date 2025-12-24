import soundfile as sf
import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np

# CONFIGURATION
# Update this to your converted WAV folder
INPUT_DIR = Path("/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/wav") 

def scan_lengths():
    print(f"Scanning {INPUT_DIR}...")
    files = list(INPUT_DIR.rglob("*.wav"))
    
    if not files:
        print("Error: No WAV files found!")
        return

    print(f"Found {len(files)} files. Analyzing durations...")

    durations = []
    min_len = float('inf')
    max_len = 0.0
    min_file = ""
    max_file = ""

    # Loop with progress bar
    for f in tqdm(files):
        try:
            # sf.info is fast (reads header only)
            info = sf.info(str(f))
            d = info.duration
            
            durations.append(d)

            # Update Max
            if d > max_len:
                max_len = d
                max_file = f

            # Update Min
            if d < min_len:
                min_len = d
                min_file = f
                
        except Exception as e:
            print(f"Error reading {f.name}: {e}")

    # --- REPORT ---
    if durations:
        avg_len = np.mean(durations)
        print("\n" + "="*40)
        print(f" DATASET STATISTICS")
        print("="*40)
        print(f"Total Files:   {len(durations)}")
        print(f"Min Length:    {min_len:.2f} seconds")
        print(f"   -> File:    {min_file}")
        print(f"Max Length:    {max_len:.2f} seconds")
        print(f"   -> File:    {max_file}")
        print(f"Average Len:   {avg_len:.2f} seconds")
        print("="*40)
        
        # Recommendation
        print("\n--- SIMULATION SETTINGS ---")
        if min_len < 3.0:
            print(f"WARNING: Your shortest file ({min_len:.2f}s) is shorter than 3.0s.")
            print("Action: The simulation script has logic to SKIP these files.")
            print("This is fine, as long as most files are > 3.0s.")
        else:
            print("Great! All files are longer than 3.0s.")

if __name__ == "__main__":
    scan_lengths()