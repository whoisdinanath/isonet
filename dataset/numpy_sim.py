import numpy as np
import soundfile as sf
import glob
import random
from pathlib import Path
from tqdm import tqdm
from scipy import signal

# --- CONFIGURATION ---
SOURCE_DIR = Path("/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/wav")
OUTPUT_DIR = Path("/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/multich_test")
NUM_SAMPLES = 2
CLIP_LENGTH = 3.0
FS = 16000

# --- 7cm SQUARE ARRAY GEOMETRY ---
half = 0.07 / 2
SPEED_OF_SOUND = 343.0  # m/s

mic_positions = np.array([
    [ half,  half, 0], 
    [-half,  half, 0], 
    [-half, -half, 0], 
    [ half, -half, 0] 
])

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

def apply_delay_and_attenuation(audio, source_pos, mic_pos, fs=16000):
    """Apply time delay and distance attenuation for a single microphone"""
    distance = np.linalg.norm(source_pos - mic_pos)
    delay_samples = int((distance / SPEED_OF_SOUND) * fs)
    attenuation = 1.0 / max(distance, 0.1)  # Prevent division by zero
    
    # Apply delay
    delayed = np.zeros(len(audio) + delay_samples)
    delayed[delay_samples:delay_samples + len(audio)] = audio * attenuation
    
    return delayed[:len(audio)]

def simple_reverb(audio, rt60=0.3, fs=16000):
    """Add simple reverb using exponentially decaying noise"""
    reverb_length = int(rt60 * fs)
    reverb_ir = np.random.randn(reverb_length) * np.exp(-6 * np.arange(reverb_length) / reverb_length)
    reverb_ir = reverb_ir / np.max(np.abs(reverb_ir)) * 0.1
    return signal.convolve(audio, reverb_ir, mode='same')

def generate_simulation():
    # Setup Folders
    print(f"Creating output folder: {OUTPUT_DIR}")
    (OUTPUT_DIR / "mixed").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "clean").mkdir(parents=True, exist_ok=True)
    
    # Get File List
    print("Indexing WAV files...")
    all_wavs = list(glob.glob(str(SOURCE_DIR / "**/*.wav"), recursive=True))
    if len(all_wavs) == 0:
        print("ERROR: No .wav files found!")
        return
    print(f"Found {len(all_wavs)} files. Generating {NUM_SAMPLES} samples...")

    # Initialize metadata
    meta_path = OUTPUT_DIR / "metadata.csv"
    with open(meta_path, "w") as f:
        f.write("filename,video_path,azimuth,distance,rt60\n")

    successful = 0
    
    for i in tqdm(range(NUM_SAMPLES), desc="Generating samples"):
        try:
            # Get audio sources
            target_audio, target_path = get_random_wav(all_wavs)
            noise_audio, _ = get_random_wav(all_wavs)
            
            # Random source positions (relative to array center)
            t_angle = random.uniform(0, 2 * np.pi)
            t_dist = random.uniform(0.5, 2.0)
            target_pos = np.array([
                t_dist * np.cos(t_angle), 
                t_dist * np.sin(t_angle), 
                random.uniform(0.3, 0.7)
            ])
            
            n_angle = random.uniform(0, 2 * np.pi)
            n_dist = random.uniform(0.5, 2.0)
            noise_pos = np.array([
                n_dist * np.cos(n_angle), 
                n_dist * np.sin(n_angle), 
                random.uniform(0.3, 0.7)
            ])
            
            # Generate multichannel signal
            num_mics = len(mic_positions)
            multichannel = np.zeros((len(target_audio), num_mics))
            
            rt60 = random.uniform(0.2, 0.5)
            
            for mic_idx, mic_pos in enumerate(mic_positions):
                # Process target
                target_ch = apply_delay_and_attenuation(target_audio, target_pos, mic_pos, FS)
                target_ch = simple_reverb(target_ch, rt60, FS)
                
                # Process noise
                noise_ch = apply_delay_and_attenuation(noise_audio, noise_pos, mic_pos, FS)
                noise_ch = simple_reverb(noise_ch, rt60, FS)
                
                # Mix
                multichannel[:, mic_idx] = target_ch + noise_ch * 0.3
            
            # Normalize
            if np.max(np.abs(multichannel)) > 0:
                multichannel = multichannel / np.max(np.abs(multichannel)) * 0.9
            
            # Save
            filename = f"sample_{successful:05d}"
            sf.write(OUTPUT_DIR / "mixed" / f"{filename}.wav", multichannel, FS)
            sf.write(OUTPUT_DIR / "clean" / f"{filename}.wav", target_audio, FS)
            
            # Metadata
            with open(meta_path, "a") as f:
                target_vid_path = str(target_path).replace(".wav", ".mp4").replace("/wav/", "/mp4/")
                f.write(f"{filename},{target_vid_path},{t_angle:.4f},{t_dist:.2f},{rt60:.3f}\n")
            
            successful += 1
            
        except Exception as e:
            print(f"\nError: {e}")
            continue
    
    print(f"\nGenerated {successful} samples in {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_simulation()