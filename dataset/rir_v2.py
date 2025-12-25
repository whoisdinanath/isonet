import pyroomacoustics as pra
import numpy as np
import soundfile as sf
import os
import glob
import random
import shutil
from pathlib import Path
from tqdm import tqdm
from scipy.signal import convolve
from multiprocessing import Pool, cpu_count, Lock
import functools
import argparse

# --- CONFIGURATION ---
SOURCE_DIR = Path("/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/wav")
VIDEO_SOURCE_DIR = Path("/mnt/DATA/Bibek/Speech/isolate-speech/data/mp4")

# UPDATED: Output to 'multich' folder as requested
OUTPUT_DIR = Path("/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/multich")

NUM_WORKERS = cpu_count() - 1 or 1  # Leave one core free
CLIP_LENGTH = 4.0
FS = 16000
MIN_SNR_DB = 0
MAX_SNR_DB = 20

# --- 7cm SQUARE ARRAY GEOMETRY ---
half = 0.07 / 2
mic_offsets = np.array([
    [ half,  half, 0], 
    [-half,  half, 0], 
    [-half, -half, 0], 
    [ half, -half, 0] 
]).T 

def get_random_wav_chunk(file_list):
    """Get a random chunk from a random WAV file"""
    for _ in range(50):
        try:
            wav_path = random.choice(file_list)
            info = sf.info(wav_path)
            if info.duration < CLIP_LENGTH: continue
            
            start_sec = random.uniform(0, info.duration - CLIP_LENGTH)
            start_sample = int(start_sec * FS)
            
            audio, _ = sf.read(wav_path, start=start_sample, frames=int(CLIP_LENGTH*FS), always_2d=True)
            audio = audio[:, 0]
            
            # Normalize
            peak = np.max(np.abs(audio))
            if peak > 0: audio = audio / peak
            
            # Return audio, full path, and start time
            return audio, wav_path, start_sec
        except: continue
    raise RuntimeError("No valid audio found after 50 attempts")

def process_single_file(args):
    """Process a single target file with RIR simulation"""
    target_file, all_wavs, output_idx = args
    
    # Set random seed based on output_idx to ensure different randomness per process
    np.random.seed(output_idx + int.from_bytes(os.urandom(4), 'little'))
    random.seed(output_idx + int.from_bytes(os.urandom(4), 'little'))
    
    try:
        # Load target audio
        info = sf.info(target_file)
        if info.duration < CLIP_LENGTH:
            return None
            
        start_sec = random.uniform(0, info.duration - CLIP_LENGTH)
        start_sample = int(start_sec * FS)
        
        target_dry, _ = sf.read(target_file, start=start_sample, frames=int(CLIP_LENGTH*FS), always_2d=True)
        target_dry = target_dry[:, 0]
        
        # Normalize target
        peak = np.max(np.abs(target_dry))
        if peak > 0:
            target_dry = target_dry / peak
        
        # Get random noise file (retry logic)
        noise_dry = None
        noise_wav_path = None
        noise_start_sec = 0
        for attempt in range(10):
            try:
                noise_dry, noise_wav_path, noise_start_sec = get_random_wav_chunk(all_wavs)
                break
            except:
                continue
        
        if noise_dry is None:
            return None
        
        # --- Room & Geometry --- (retry until valid positions found)
        max_retries = 20
        for retry in range(max_retries):
            room_dim = np.array([random.uniform(3,8), random.uniform(3,6), random.uniform(2.5,3.5)])
            e_abs, _ = pra.inverse_sabine(random.uniform(0.2, 0.6), room_dim)
            room = pra.ShoeBox(room_dim, fs=FS, materials=pra.Material(e_abs), max_order=10)
            
            center = room_dim / 2
            mic_pos = center + np.array([random.uniform(-0.5,0.5), random.uniform(-0.5,0.5), 0])
            mic_pos[2] = 1.0
            room.add_microphone_array(mic_pos[:, None] + mic_offsets)

            # --- Sources (3D Spherical Coordinates) ---
            # Target source
            t_azimuth = random.uniform(0, 2*np.pi)  # Horizontal angle (radians)
            t_elevation = random.uniform(-np.pi/6, np.pi/6)  # Vertical angle (±30°)
            t_distance = random.uniform(0.8, 1.5)  # Distance from mic center (meters)
            
            # Convert spherical to Cartesian
            t_pos = mic_pos + np.array([
                t_distance * np.cos(t_elevation) * np.cos(t_azimuth),
                t_distance * np.cos(t_elevation) * np.sin(t_azimuth),
                t_distance * np.sin(t_elevation)
            ])
            
            # Noise source
            n_azimuth = random.uniform(0, 2*np.pi)
            n_elevation = random.uniform(-np.pi/6, np.pi/6)
            n_distance = random.uniform(0.8, 1.5)
            
            n_pos = mic_pos + np.array([
                n_distance * np.cos(n_elevation) * np.cos(n_azimuth),
                n_distance * np.cos(n_elevation) * np.sin(n_azimuth),
                n_distance * np.sin(n_elevation)
            ])
            
            if room.is_inside(t_pos) and room.is_inside(n_pos):
                break
        else:
            # Failed to find valid positions after max_retries
            return None

        # --- RIR & Convolve ---
        room.add_source(t_pos)
        room.add_source(n_pos)
        room.compute_rir()

        clean_ref = convolve(target_dry, room.rir[0][0])[:int(CLIP_LENGTH*FS)]
        
        # First pass: convolve all channels and determine minimum length
        sig_targets = []
        sig_noises = []
        min_overall_len = float('inf')
        
        for m in range(4):
            sig_t = convolve(target_dry, room.rir[m][0])
            sig_n = convolve(noise_dry, room.rir[m][1])
            sig_targets.append(sig_t)
            sig_noises.append(sig_n)
            min_overall_len = min(min_overall_len, len(sig_t), len(sig_n))
        
        # Ensure we have at least CLIP_LENGTH samples
        target_samples = int(CLIP_LENGTH * FS)
        if min_overall_len < target_samples:
            return None
        
        min_overall_len = target_samples
        
        # Second pass: truncate and mix
        mics_mix = []
        final_snr = 0
        
        for m in range(4):
            sig_t = sig_targets[m][:min_overall_len]
            sig_n = sig_noises[m][:min_overall_len]

            if m == 0:
                t_rms = np.sqrt(np.mean(sig_t**2))
                n_rms = np.sqrt(np.mean(sig_n**2))
                final_snr = random.uniform(MIN_SNR_DB, MAX_SNR_DB)
                scale = t_rms / (n_rms * (10**(final_snr/20)) + 1e-8)
            
            mics_mix.append(sig_t + (sig_n * scale))
        
        # Stack arrays - now all have same length
        final_mix = np.stack(mics_mix, axis=-1)  # Shape: (samples, 4)

        # --- Normalize ---
        peak = np.max(np.abs(final_mix))
        if peak > 0:
            final_mix = final_mix / peak * 0.9
            clean_ref = clean_ref / peak * 0.9

        filename = f"sample_{output_idx:05d}"
        
        # Save files
        mixed_path = OUTPUT_DIR / "mixed" / f"{filename}.wav"
        clean_path = OUTPUT_DIR / "clean" / f"{filename}.wav"
        
        sf.write(mixed_path, final_mix, FS)
        sf.write(clean_path, clean_ref, FS)

        # Prepare comprehensive metadata
        # Extract relative paths for better portability
        source_wav_rel = str(target_file).replace(str(SOURCE_DIR), "").lstrip("/")
        
        # Convert WAV path to video path
        # WAV structure: .../idXXXXX/video_id/00095.wav
        # Video structure: .../idXXXXX/video_id/00095.mp4
        # Simply replace SOURCE_DIR with VIDEO_SOURCE_DIR and .wav with .mp4
        source_vid_path = Path(str(target_file).replace(str(SOURCE_DIR), str(VIDEO_SOURCE_DIR)).replace(".wav", ".mp4"))
        source_vid_rel = str(source_vid_path).replace(str(VIDEO_SOURCE_DIR), "").lstrip("/")
        
        # Copy video to output directory with sample filename
        video_output_path = OUTPUT_DIR / "video" / f"{filename}.mp4"
        try:
            if source_vid_path.exists():
                shutil.copy2(str(source_vid_path), str(video_output_path))
                video_rel = f"video/{filename}.mp4"
            else:
                print(f"Warning: Video not found: {source_vid_path}")
                video_rel = ""  # Mark as missing if video doesn't exist
        except Exception as e:
            print(f"Warning: Could not copy video for {filename}: {e}")
            video_rel = ""
        
        # Audio output paths (relative to OUTPUT_DIR)
        mixed_rel = f"mixed/{filename}.wav"
        clean_rel = f"clean/{filename}.wav"
        
        # Calculate RT60 from absorption coefficient
        rt60 = pra.inverse_sabine(e_abs, room_dim)[1]
        
        metadata_line = (
            f"{filename},"
            f"{source_wav_rel},{source_vid_rel},{start_sec:.3f},"
            f"{mixed_rel},{clean_rel},{video_rel},"
            f"{t_azimuth:.4f},{t_elevation:.4f},{t_distance:.4f},"
            f"{room_dim[0]:.2f},{room_dim[1]:.2f},{room_dim[2]:.2f},{rt60:.3f},{final_snr:.1f}\n"
        )
        
        return metadata_line

    except Exception as e:
        # Log error for debugging
        print(f"Error processing file {output_idx}: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_simulation(num_files=None):
    # 1. Setup Folders & Verify Write Permissions
    print(f"Target Output Directory: {OUTPUT_DIR}")
    
    try:
        (OUTPUT_DIR / "mixed").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "clean").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "video").mkdir(parents=True, exist_ok=True)
        print("Directories created successfully.")
    except Exception as e:
        print(f"ERROR Creating Directory: {e}")
        return

    # 2. Index Files
    print("Indexing WAV files...")
    all_wavs = list(glob.glob(str(SOURCE_DIR / "**/*.wav"), recursive=True))
    if not all_wavs:
        print("ERROR: No .wav files found in SOURCE_DIR!")
        return
    print(f"Found {len(all_wavs)} source files.")
    
    # Filter files by duration
    print("Filtering files by duration...")
    valid_files = []
    for wav_file in tqdm(all_wavs, desc="Checking durations"):
        try:
            info = sf.info(wav_file)
            if info.duration >= CLIP_LENGTH:
                valid_files.append(wav_file)
        except:
            continue
    
    print(f"Found {len(valid_files)} valid files (>= {CLIP_LENGTH}s duration)")
    
    if not valid_files:
        print("ERROR: No valid files found!")
        return
    
    # Limit number of files if specified
    if num_files is not None and num_files < len(valid_files):
        valid_files = valid_files[:num_files]
        print(f"Processing {num_files} files as requested.")

    # 3. Prepare metadata file
    meta_path = OUTPUT_DIR / "metadata.csv"
    with open(meta_path, "w") as f:
        f.write("filename,"
                "source_wav,source_video,start_time,"
                "mixed_audio,clean_audio,video_file,"
                "target_azimuth,target_elevation,target_distance,"
                "room_x,room_y,room_z,rt60,snr_db\n")

    # 4. Prepare arguments for multiprocessing
    print(f"\nStarting multiprocessing with {NUM_WORKERS} workers...")
    tasks = [(valid_files[i], all_wavs, i) for i in range(len(valid_files))]
    
    # 5. Process files in parallel
    successful_count = 0
    with Pool(processes=NUM_WORKERS) as pool:
        with tqdm(total=len(valid_files), desc="Processing files") as pbar:
            for result in pool.imap_unordered(process_single_file, tasks):
                if result is not None:
                    # Write metadata
                    with open(meta_path, "a") as f:
                        f.write(result)
                    successful_count += 1
                pbar.update(1)
    
    print(f"\nSuccess! Processed {successful_count}/{len(valid_files)} files.")
    print(f"Output saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RIR simulations for speaker isolation")
    parser.add_argument("-n", "--num-files", type=int, default=None, 
                        help="Number of files to process (default: process all files)")
    parser.add_argument("--all", action="store_true",
                        help="Process all files in the directory")
    
    args = parser.parse_args()
    
    # If --all is specified, num_files remains None (process all)
    # Otherwise, use the -n value
    num_files = None if args.all else args.num_files
    
    generate_simulation(num_files)