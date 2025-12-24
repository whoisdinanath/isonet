import pyroomacoustics as pra
import numpy as np
import soundfile as sf
import os
import glob
import random
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION (TEST MODE) ---
SOURCE_DIR = Path("/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/wav")
OUTPUT_DIR = Path("/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/multich_test")
NUM_SAMPLES = 2
CLIP_LENGTH = 3.0
FS = 16000

# SNR range in dB (higher = cleaner target, lower = noisier)
MIN_SNR_DB = 0   # Very noisy
MAX_SNR_DB = 20  # Clean

# --- 7cm SQUARE ARRAY GEOMETRY ---
half = 0.07 / 2
mic_offsets = np.array([
    [ half,  half, 0], 
    [-half,  half, 0], 
    [-half, -half, 0], 
    [ half, -half, 0] 
]).T 

def get_random_wav(file_list):
    max_attempts = 100
    for _ in range(max_attempts):
        wav_path = random.choice(file_list)
        try:
            info = sf.info(wav_path)
            if info.duration < CLIP_LENGTH:
                continue
            start_sample = int(random.uniform(0, info.duration - CLIP_LENGTH) * FS)
            audio, _ = sf.read(wav_path, start=start_sample, frames=int(CLIP_LENGTH*FS), always_2d=True)
            return audio[:, 0], wav_path
        except Exception as e:
            continue
    raise RuntimeError(f"Could not find valid audio file after {max_attempts} attempts")

def calculate_rms(audio):
    """Calculate RMS (Root Mean Square) energy of audio"""
    return np.sqrt(np.mean(audio**2))

def scale_to_snr(target_audio, noise_audio, snr_db):
    """Scale noise to achieve desired SNR relative to target"""
    target_rms = calculate_rms(target_audio)
    noise_rms = calculate_rms(noise_audio)
    
    # Avoid division by zero
    if noise_rms < 1e-10:
        return noise_audio
    
    # Calculate required scaling factor
    snr_linear = 10 ** (snr_db / 20.0)
    scale_factor = target_rms / (noise_rms * snr_linear)
    
    return noise_audio * scale_factor

def generate_simulation():
    # 1. Setup Folders
    print(f"Creating output folder: {OUTPUT_DIR}")
    (OUTPUT_DIR / "mixed").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "clean").mkdir(parents=True, exist_ok=True)
    
    # 2. Get File List
    print("Indexing WAV files...")
    all_wavs = list(glob.glob(str(SOURCE_DIR / "**/*.wav"), recursive=True))
    if len(all_wavs) == 0:
        print("ERROR: No .wav files found!")
        return
    print(f"Found {len(all_wavs)} clean files. Generating {NUM_SAMPLES} test samples...")

    # 3. Initialize metadata file
    meta_path = OUTPUT_DIR / "metadata.csv"
    with open(meta_path, "w") as f:
        f.write("filename,video_path,azimuth,room_dims,absorption,snr_db\n")

    # 4. Simulation Loop
    successful = 0
    attempts = 0
    max_attempts = NUM_SAMPLES * 10
    
    pbar = tqdm(total=NUM_SAMPLES, desc="Generating samples")
    
    while successful < NUM_SAMPLES and attempts < max_attempts:
        attempts += 1
        
        try:
            # Room & Mic Setup
            room_dim = np.array([
                random.uniform(3, 8), 
                random.uniform(3, 6), 
                random.uniform(2.5, 3.5)
            ])
            absorption = random.uniform(0.2, 0.6)
            e_absorption, max_order = pra.inverse_sabine(absorption, room_dim)
            room = pra.ShoeBox(
                room_dim, 
                fs=FS, 
                materials=pra.Material(e_absorption), 
                max_order=10
            )
            
            room_center = room_dim / 2
            array_pos = room_center + np.array([
                random.uniform(-0.5, 0.5), 
                random.uniform(-0.5, 0.5), 
                0
            ])
            array_pos[2] = 1.0 
            room.add_microphone_array(array_pos[:, None] + mic_offsets)

            # Target Source
            target_audio, target_path = get_random_wav(all_wavs)
            t_angle = random.uniform(0, 2 * np.pi)
            t_dist = random.uniform(0.5, 2.0)
            t_pos = array_pos + np.array([
                t_dist * np.cos(t_angle), 
                t_dist * np.sin(t_angle), 
                random.uniform(0.3, 0.7)
            ])
            
            if not room.is_inside(t_pos): 
                continue

            # Noise Source
            noise_audio, _ = get_random_wav(all_wavs)
            n_angle = random.uniform(0, 2 * np.pi)
            n_dist = random.uniform(0.5, 2.0)
            n_pos = array_pos + np.array([
                n_dist * np.cos(n_angle), 
                n_dist * np.sin(n_angle), 
                random.uniform(0.3, 0.7)
            ])
            
            if not room.is_inside(n_pos): 
                continue
            
            # Scale noise to desired SNR BEFORE adding to room
            target_snr_db = random.uniform(MIN_SNR_DB, MAX_SNR_DB)
            noise_audio_scaled = scale_to_snr(target_audio, noise_audio, target_snr_db)
            
            # Add sources to room
            room.add_source(t_pos, signal=target_audio)
            room.add_source(n_pos, signal=noise_audio_scaled)

            # Simulate
            room.simulate()

            # Save
            mix = room.mic_array.signals.T 
            if np.max(np.abs(mix)) > 0:
                mix = mix / np.max(np.abs(mix)) * 0.9
            else:
                continue
            
            filename = f"sample_{successful:05d}"
            sf.write(OUTPUT_DIR / "mixed" / f"{filename}.wav", mix, FS)
            sf.write(OUTPUT_DIR / "clean" / f"{filename}.wav", target_audio, FS)

            # Metadata
            with open(meta_path, "a") as f:
                target_vid_path = str(target_path).replace(".wav", ".mp4").replace("/wav/", "/mp4/")
                room_dims_str = f"{room_dim[0]:.2f}x{room_dim[1]:.2f}x{room_dim[2]:.2f}"
                f.write(f"{filename},{target_vid_path},{t_angle:.4f},{room_dims_str},{absorption:.3f},{target_snr_db:.1f}\n")
            
            successful += 1
            pbar.update(1)
            
        except Exception as e:
            print(f"\nError on attempt {attempts}: {e}")
            continue
    
    pbar.close()
    
    if successful < NUM_SAMPLES:
        print(f"\nWarning: Only generated {successful}/{NUM_SAMPLES} samples after {attempts} attempts")
    else:
        print(f"\nSuccess! Generated {successful} samples")
    
    print(f"Output folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_simulation()