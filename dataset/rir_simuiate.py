import pyroomacoustics as pra
import numpy as np
import soundfile as sf
import os
import glob
import random
from pathlib import Path
from tqdm import tqdm
from scipy.signal import convolve

# --- CONFIGURATION ---
SOURCE_DIR = Path("/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/wav")
OUTPUT_DIR = Path("/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/multich_test")
NUM_SAMPLES = 10  # Set to 5000 for full run
CLIP_LENGTH = 3.0
FS = 16000
MIN_SNR_DB = 0
MAX_SNR_DB = 20

# --- 7cm SQUARE ARRAY ---
half = 0.07 / 2
mic_offsets = np.array([
    [ half,  half, 0], 
    [-half,  half, 0], 
    [-half, -half, 0], 
    [ half, -half, 0] 
]).T 

def get_random_wav(file_list):
    for _ in range(50):
        try:
            wav_path = random.choice(file_list)
            info = sf.info(wav_path)
            if info.duration < CLIP_LENGTH: continue
            
            start = int(random.uniform(0, info.duration - CLIP_LENGTH) * FS)
            audio, _ = sf.read(wav_path, start=start, frames=int(CLIP_LENGTH*FS), always_2d=True)
            
            # NORMALIZE INPUT immediately (Fixes the "too low volume" input issue)
            audio = audio[:, 0]
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            return audio, wav_path
        except: continue
    raise RuntimeError("No valid audio found")

def generate_simulation():
    (OUTPUT_DIR / "mixed").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "clean").mkdir(parents=True, exist_ok=True)
    
    all_wavs = list(glob.glob(str(SOURCE_DIR / "**/*.wav"), recursive=True))
    meta_path = OUTPUT_DIR / "metadata.csv"
    
    with open(meta_path, "w") as f:
        f.write("filename,video_path,azimuth,snr_db\n")

    for i in tqdm(range(NUM_SAMPLES)):
        try:
            # 1. Setup Room
            room_dim = np.array([random.uniform(3,8), random.uniform(3,6), random.uniform(2.5,3.5)])
            e_abs, _ = pra.inverse_sabine(random.uniform(0.2, 0.6), room_dim)
            room = pra.ShoeBox(room_dim, fs=FS, materials=pra.Material(e_abs), max_order=10)
            
            # 2. Place Array
            center = room_dim / 2
            mic_pos = center + np.array([random.uniform(-1,1), random.uniform(-1,1), 0])
            mic_pos[2] = 1.0
            room.add_microphone_array(mic_pos[:, None] + mic_offsets)

            # 3. Define Positions (Don't add sources yet!)
            # Target Pos
            t_angle = random.uniform(0, 2*np.pi)
            t_dist = random.uniform(0.5, 2.5)
            t_pos = mic_pos + np.array([t_dist*np.cos(t_angle), t_dist*np.sin(t_angle), 0.5])
            if not room.is_inside(t_pos): continue
            
            # Noise Pos
            n_angle = random.uniform(0, 2*np.pi)
            n_dist = random.uniform(0.5, 2.5)
            n_pos = mic_pos + np.array([n_dist*np.cos(n_angle), n_dist*np.sin(n_angle), 0.5])
            if not room.is_inside(n_pos): continue

            # 4. Get Audio Signals
            target_dry, t_path = get_random_wav(all_wavs)
            noise_dry, _ = get_random_wav(all_wavs)

            # 5. Compute RIRs (Room Impulse Responses) manually
            # This allows us to convolve separately!
            room.add_source(t_pos) # Source 0
            room.add_source(n_pos) # Source 1
            room.compute_rir()

            # 6. Convolve to get Component Signals at Mics
            # RIR shape: [num_mics, num_sources]
            
            # --- CREATE TARGET COMPONENT (The "Clean" reference) ---
            # We take the target source and convolve with Mic 0's RIR for the target
            # Note: We usually define the "Clean Reference" as what Mic 0 hears.
            clean_ref_mic0 = convolve(target_dry, room.rir[0][0])[:int(CLIP_LENGTH*FS)]
            
            # --- CREATE MIX ---
            # We need the mix for ALL 4 channels
            mics_mix = []
            for m in range(4):
                # Convolve Target
                sig_t = convolve(target_dry, room.rir[m][0])
                # Convolve Noise
                sig_n = convolve(noise_dry, room.rir[m][1])
                
                # Trim to length
                min_len = min(len(sig_t), len(sig_n))
                sig_t = sig_t[:min_len]
                sig_n = sig_n[:min_len]

                # SNR SCALING (Calculated at Mic 1 for consistency)
                if m == 0:
                    t_rms = np.sqrt(np.mean(sig_t**2))
                    n_rms = np.sqrt(np.mean(sig_n**2))
                    snr_db = random.uniform(MIN_SNR_DB, MAX_SNR_DB)
                    snr_lin = 10**(snr_db/20)
                    scale = t_rms / (n_rms * snr_lin + 1e-8)
                
                # Apply Mix
                channel_mix = sig_t + (sig_n * scale)
                mics_mix.append(channel_mix)
            
            # Stack into (Samples, 4)
            final_mix = np.array(mics_mix).T[:int(CLIP_LENGTH*FS), :]

            # 7. Normalize OUTPUT Volume
            # This fixes your "too low" issue. We maximize the file volume.
            peak = np.max(np.abs(final_mix))
            if peak > 0:
                final_mix = final_mix / peak * 0.9
                # Normalize clean ref by same factor to preserve relative volume
                clean_ref_mic0 = clean_ref_mic0 / peak * 0.9

            # 8. Save
            filename = f"sample_{i:05d}"
            sf.write(OUTPUT_DIR / "mixed" / f"{filename}.wav", final_mix, FS)
            sf.write(OUTPUT_DIR / "clean" / f"{filename}.wav", clean_ref_mic0, FS)

            with open(meta_path, "a") as f:
                vid_path = str(t_path).replace(".wav", ".mp4").replace("/wav/", "/mp4/")
                f.write(f"{filename},{vid_path},{t_angle},{snr_db}\n")

        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    generate_simulation()