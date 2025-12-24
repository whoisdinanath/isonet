import pyroomacoustics as pra
import numpy as np
import soundfile as sf
import os
import glob
import random
from pathlib import Path
from tqdm import tqdm
from scipy.signal import convolve
from multiprocessing import Pool, cpu_count
import traceback

# --- CONFIGURATION ---
SOURCE_DIR = Path("/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/wav")
OUTPUT_DIR = Path("/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/multich")

# Leave 2 cores free
NUM_WORKERS = max(1, cpu_count() - 2)
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

# Global variable for workers
worker_all_wavs = []

def init_worker(shared_list):
    """Initialize worker with the read-only file list"""
    global worker_all_wavs
    worker_all_wavs = shared_list

def get_random_noise_chunk():
    """Get a random noise chunk from the global list"""
    global worker_all_wavs
    # Use standard python random here, generally safe across processes if seeded differently
    # But just in case, we rely on the process seeding done in process_single_file
    for _ in range(20):
        try:
            wav_path = random.choice(worker_all_wavs)
            info = sf.info(wav_path)
            if info.duration < CLIP_LENGTH: continue
            
            start_sec = random.uniform(0, info.duration - CLIP_LENGTH)
            start_sample = int(start_sec * FS)
            
            audio, _ = sf.read(wav_path, start=start_sample, frames=int(CLIP_LENGTH*FS), always_2d=True)
            audio = audio[:, 0]
            
            # Normalize
            peak = np.max(np.abs(audio))
            if peak > 0: audio = audio / peak
            
            return audio
        except: continue
    return np.zeros(int(CLIP_LENGTH*FS))

def process_single_file(args):
    """Process a single target file."""
    target_file, output_idx = args
    
    # --- RESUME LOGIC ---
    # Check if this file already exists. If so, SKIP IT.
    # Note: Using 05d to match your previous filenames. 
    # Python handles overflow gracefully (100000 fits in 05d as '100000')
    filename = f"sample_{output_idx:05d}"
    mixed_path = OUTPUT_DIR / "mixed" / f"{filename}.wav"
    
    if mixed_path.exists():
        # Return None to signal "Skipped"
        return None

    # --- SEED FIX ---
    # We combine the process ID, the index, and random bytes, 
    # then mask it with 0xFFFFFFFF to ensure it fits in 32 bits.
    seed_val = (output_idx + os.getpid() + int.from_bytes(os.urandom(4), 'little')) & 0xFFFFFFFF
    np.random.seed(seed_val)
    random.seed(seed_val)
    
    try:
        # 1. Load Target Audio
        info = sf.info(target_file)
        if info.duration < CLIP_LENGTH:
            return None 
            
        start_sec = random.uniform(0, info.duration - CLIP_LENGTH)
        start_sample = int(start_sec * FS)
        
        target_dry, _ = sf.read(target_file, start=start_sample, frames=int(CLIP_LENGTH*FS), always_2d=True)
        target_dry = target_dry[:, 0]
        
        peak = np.max(np.abs(target_dry))
        if peak > 0: target_dry = target_dry / peak
        
        # 2. Load Noise
        noise_dry = get_random_noise_chunk()
        
        # 3. Simulation Loop 
        for attempt in range(10):
            try:
                room_dim = np.array([random.uniform(3,8), random.uniform(3,6), random.uniform(2.5,3.5)])
                e_abs, _ = pra.inverse_sabine(random.uniform(0.2, 0.6), room_dim)
                room = pra.ShoeBox(room_dim, fs=FS, materials=pra.Material(e_abs), max_order=10)
                
                center = room_dim / 2
                mic_pos = center + np.array([random.uniform(-1,1), random.uniform(-1,1), 0])
                mic_pos[2] = 1.0
                
                if not room.is_inside(mic_pos): continue
                room.add_microphone_array(mic_pos[:, None] + mic_offsets)

                t_angle = random.uniform(0, 2*np.pi)
                t_pos = mic_pos + np.array([1.0*np.cos(t_angle), 1.0*np.sin(t_angle), 0.5])
                
                n_angle = random.uniform(0, 2*np.pi)
                n_pos = mic_pos + np.array([1.0*np.cos(n_angle), 1.0*np.sin(n_angle), 0.5])

                if not room.is_inside(t_pos) or not room.is_inside(n_pos):
                    continue 

                room.add_source(t_pos)
                room.add_source(n_pos)
                room.compute_rir()

                clean_ref = convolve(target_dry, room.rir[0][0])[:int(CLIP_LENGTH*FS)]
                
                mics_mix = []
                for m in range(4):
                    sig_t = convolve(target_dry, room.rir[m][0])
                    sig_n = convolve(noise_dry, room.rir[m][1])
                    
                    min_len = min(len(sig_t), len(sig_n))
                    sig_t, sig_n = sig_t[:min_len], sig_n[:min_len]

                    scale = 1.0
                    if m == 0:
                        t_rms = np.sqrt(np.mean(sig_t**2))
                        n_rms = np.sqrt(np.mean(sig_n**2))
                        final_snr = random.uniform(MIN_SNR_DB, MAX_SNR_DB)
                        scale = t_rms / (n_rms * (10**(final_snr/20)) + 1e-8)
                    
                    mics_mix.append(sig_t + (sig_n * scale))
                
                final_mix = np.array(mics_mix).T[:int(CLIP_LENGTH*FS), :]

                peak = np.max(np.abs(final_mix))
                if peak > 0:
                    final_mix = final_mix / peak * 0.9
                    clean_ref = clean_ref / peak * 0.9

                sf.write(OUTPUT_DIR / "mixed" / f"{filename}.wav", final_mix, FS)
                sf.write(OUTPUT_DIR / "clean" / f"{filename}.wav", clean_ref, FS)

                vid_path = str(target_file).replace(".wav", ".mp4").replace("/wav/", "/mp4/")
                return f"{filename},{vid_path},{start_sec:.3f},{t_angle:.4f},{final_snr:.1f}\n"

            except Exception:
                continue
        
        return None

    except Exception as e:
        print(f"Error processing {target_file}: {e}")
        return None

def generate_simulation():
    print(f"--- Simulating for {OUTPUT_DIR} ---")
    (OUTPUT_DIR / "mixed").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "clean").mkdir(parents=True, exist_ok=True)

    print("Indexing WAV files...")
    all_wavs = list(glob.glob(str(SOURCE_DIR / "**/*.wav"), recursive=True))
    if not all_wavs:
        print(" ERROR: No .wav files found!")
        return
    
    # Sort them to ensure consistent ordering for "Resume" logic
    # This ensures index 0 is always the same file
    all_wavs.sort()
    
    # Filter valid files
    valid_files = []
    print("Filtering valid files...")
    for f in tqdm(all_wavs):
        try:
            if sf.info(f).duration >= CLIP_LENGTH:
                valid_files.append(f)
        except: continue
        
    print(f"Found {len(valid_files)} valid files.")

    meta_path = OUTPUT_DIR / "metadata.csv"
    if not meta_path.exists():
        with open(meta_path, "w") as f:
            f.write("filename,video_path,start_time,azimuth,snr_db\n")

    tasks = [(f, i) for i, f in enumerate(valid_files)]
    
    print(f"Starting pool with {NUM_WORKERS} workers...")
    
    successful_new = 0
    skipped = 0
    
    with Pool(processes=NUM_WORKERS, initializer=init_worker, initargs=(all_wavs,)) as pool:
        with open(meta_path, "a") as f_meta:
            # Using imap_unordered for speed
            for result in tqdm(pool.imap_unordered(process_single_file, tasks), total=len(tasks), desc="Processing"):
                if result is not None:
                    # If result is not None, it means we actually generated a NEW file
                    f_meta.write(result)
                    successful_new += 1
                else:
                    # If None, we either skipped it (already exists) or it failed
                    # We can't distinguish easily here without slowing down, 
                    # but the important part is we continue.
                    skipped += 1
    
    print(f"\n Done!")
    print(f"New samples generated: {successful_new}")
    print(f"Skipped/Failed samples: {skipped}")

if __name__ == "__main__":
    generate_simulation()