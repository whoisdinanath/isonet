import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os
import cv2
import numpy as np
from pathlib import Path

class IsoNetDataset(Dataset):
    def __init__(self, csv_path, clip_length=4.0, fps=25, video_size=(112, 112)):
        """
        Args:
            csv_path (str): Path to train.csv or val.csv
            clip_length (float): Audio duration in seconds (must match simulation)
            fps (int): Target frames per second for video (VoxCeleb is 25)
            video_size (tuple): Target resize dimension (H, W)
        """
        self.meta = pd.read_csv(csv_path)
        
        # Assume csv is inside 'multich/', so parent is the root
        self.root_dir = Path(csv_path).parent
        
        self.clip_length = clip_length
        self.fps = fps
        self.target_frames = int(clip_length * fps)  # 4.0 * 25 = 100 frames
        self.video_size = video_size

    def __len__(self):
        return len(self.meta)

    def load_video_frames(self, video_path, start_time):
        """
        Efficiently seeks to start_time and reads exactly target_frames.
        Returns: Tensor [Channels, Time, H, W]
        """
        cap = cv2.VideoCapture(str(video_path))
        
        # 1. Get Video Properties
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        if vid_fps == 0 or np.isnan(vid_fps): 
            vid_fps = 25.0
            
        # 2. Calculate Start Frame Index
        start_frame_idx = int(start_time * vid_fps)
        
        # 3. Seek to exact frame (The "Sniper" shot)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        
        frames = []
        for _ in range(self.target_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 4. Preprocessing
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize (Standard for LipReading is 112x112)
            frame = cv2.resize(frame, self.video_size)
            frames.append(frame)
            
        cap.release()
        
        # 5. Handle Edge Case: Video ended too early
        # Padding with the last frame if we are short
        if len(frames) < self.target_frames:
            # If video completely failed to load, create black frames
            if len(frames) == 0:
                frames = [np.zeros((self.video_size[0], self.video_size[1], 3), dtype=np.uint8)] * self.target_frames
            else:
                padding = [frames[-1]] * (self.target_frames - len(frames))
                frames.extend(padding)
        
        # 6. Convert to Tensor
        # Shape: [Time, H, W, Channels] -> PyTorch standard [Channels, Time, H, W]
        # Normalize to 0-1 range
        buffer = np.array(frames, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(buffer)
        return tensor.permute(3, 0, 1, 2)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.meta.iloc[idx]
        
        # 1. Get Paths & Info
        filename = row['filename']
        # The video path in CSV might be absolute, ensure it works
        vid_path_str = row['video_path']
        start_time = float(row['start_time'])
        
        mixed_path = self.root_dir / "mixed" / f"{filename}.wav"
        clean_path = self.root_dir / "clean" / f"{filename}.wav"

        # 2. Load Audio
        # Torchaudio loads as [Channels, Time]
        mixed_audio, _ = torchaudio.load(mixed_path)
        clean_audio, _ = torchaudio.load(clean_path)

        # 3. Load Video
        video_tensor = self.load_video_frames(vid_path_str, start_time)

        # 4. Final Verification (Optional, removed for speed but good for debug)
        # Ensure audio length matches exactly (sometimes MP3/WAV conversion adds ms)
        # For 4.0s @ 16kHz, we expect 64000 samples. 
        # Deep learning models need exact shapes.
        target_samples = int(self.clip_length * 16000)
        
        if mixed_audio.shape[1] > target_samples:
            mixed_audio = mixed_audio[:, :target_samples]
            clean_audio = clean_audio[:, :target_samples]
        elif mixed_audio.shape[1] < target_samples:
            # Pad with zeros if short
            pad_size = target_samples - mixed_audio.shape[1]
            mixed_audio = torch.nn.functional.pad(mixed_audio, (0, pad_size))
            clean_audio = torch.nn.functional.pad(clean_audio, (0, pad_size))

        return mixed_audio, clean_audio, video_tensor

# --- SELF-TEST BLOCK ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys

    TEST_CSV = "/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/multich/metadata.csv"
    
    if not os.path.exists(TEST_CSV):
        print(f"Error: Could not find {TEST_CSV}")
        print("Please run the split_dataset script first or point to metadata.csv")
        sys.exit(1)

    print("Testing IsoNetDataset...")
    dataset = IsoNetDataset(TEST_CSV)
    print(f"Dataset Length: {len(dataset)}")

    # 1. Load one sample
    print("Loading Sample #0...")
    mixed, clean, video = dataset[2]

    # 2. Verify Shapes
    print("\n--- Tensor Shapes ---")
    print(f"Mixed Audio: {mixed.shape}  (Expected: [4, 64000])")
    print(f"Clean Audio: {clean.shape}  (Expected: [1, 64000])")
    print(f"Video:       {video.shape}  (Expected: [3, 100, 112, 112])")

    # 3. Save Visual Check
    print("\n--- Saving Debug Image ---")
    # Take the 50th frame (middle of clip)
    # Permute from [C, T, H, W] -> [H, W, C] for saving
    frame_tensor = video[:, 50, :, :].permute(1, 2, 0)
    plt.imsave("debug_face_loader.png", frame_tensor.numpy())
    print("Saved 'debug_face_loader.png'. Please open it and check if it's a face!")