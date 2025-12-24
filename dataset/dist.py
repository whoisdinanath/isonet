import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# CONFIGURATION
INPUT_DIR = Path("/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/wav") 

def analyze_distribution():
    print(f"Scanning {INPUT_DIR}...")
    files = list(INPUT_DIR.rglob("*.wav"))
    
    if not files:
        print("Error: No WAV files found.")
        return

    durations = []
    print(f"Analyzing {len(files)} files...")
    
    # Collect durations
    for f in tqdm(files):
        try:
            info = sf.info(str(f))
            durations.append(info.duration)
        except:
            continue

    durations = np.array(durations)
    
    # --- 1. PRINT TEXT REPORT ---
    print("\n" + "="*40)
    print(" DURATION DISTRIBUTION (Up to 7s)")
    print("="*40)
    
    # Bins: 0-1, 1-2, ... 6-7, >7
    bins = range(0, 9) # 0 to 8
    counts, _ = np.histogram(durations, bins=bins)
    
    total_files = len(durations)
    
    for i in range(7):
        count = counts[i]
        percentage = (count / total_files) * 100
        bar = "#" * int(percentage / 2) # Simple text bar
        print(f"{i}-{i+1}s : {count:5d} files ({percentage:4.1f}%) | {bar}")
        
    # Count remaining
    over_7 = np.sum(durations >= 7)
    over_7_pct = (over_7 / total_files) * 100
    print(f"> 7s : {over_7:5d} files ({over_7_pct:4.1f}%) | {'#' * int(over_7_pct/2)}")
    print("="*40)

    # --- 2. GENERATE GRAPHIC ---
    print("\nGenerating plot 'audio_dist.png'...")
    
    plt.figure(figsize=(10, 6))
    
    # Main Histogram (All data)
    plt.hist(durations, bins=100, range=(0, 20), color='skyblue', edgecolor='black', alpha=0.7)
    
    # Add vertical lines for possible clip lengths
    plt.axvline(x=3.0, color='red', linestyle='--', linewidth=2, label='3.0s (Recommended)')
    plt.axvline(x=6.0, color='green', linestyle='--', linewidth=2, label='6.0s')
    
    plt.title(f"Audio Duration Distribution (N={total_files})")
    plt.xlabel("Duration (Seconds)")
    plt.ylabel("Number of Files")
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    # Zoom in inset (0 to 7s)
    plt.savefig("audio_dist.png")
    print("Done! Check 'audio_dist.png' and the table above.")

if __name__ == "__main__":
    analyze_distribution()