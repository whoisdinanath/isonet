import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os
import cv2
import numpy as np
from pathlib import Path
import sys

# Import spatial augmentation
sys.path.insert(0, str(Path(__file__).parent))
from spatial_visual_augmentation import SpatialVisualAugmenter

class IsoNetDataset(Dataset):
    def __init__(self, csv_path, clip_length=4.0, fps=25, video_size=(112, 112), 
                 apply_spatial_augmentation=True, augmentation_strength=1.0):
        """
        Args:
            csv_path (str): Path to train.csv or val.csv
            clip_length (float): Audio duration in seconds (must match simulation)
            fps (int): Target frames per second for video (VoxCeleb is 25)
            video_size (tuple): Target resize dimension (H, W)
            apply_spatial_augmentation (bool): Apply spatial-visual augmentation
            augmentation_strength (float): Strength of augmentation (0.0 = none, 1.0 = full)
        """
        self.meta = pd.read_csv(csv_path)
        
        # Assume csv is inside 'multich/', so parent is the root
        self.root_dir = Path(csv_path).parent
        
        self.clip_length = clip_length
        self.fps = fps
        self.target_frames = int(clip_length * fps)  # 4.0 * 25 = 100 frames
        self.video_size = video_size
        
        # NEW: Spatial augmentation
        self.apply_spatial_aug = apply_spatial_augmentation
        if apply_spatial_augmentation:
            self.augmenter = SpatialVisualAugmenter(
                camera_fov_horizontal=90.0,
                camera_fov_vertical=60.0,
                augmentation_strength=augmentation_strength
            )

    def __len__(self):
        return len(self.meta)

    def load_face_tracks(self, txt_path):
        """
        Parses the VoxCeleb face track text file.
        Format: FRAME X Y W H (Normalized 0-1)
        Returns: {frame_idx: (x, y, w, h)}
        """
        tracks = {}
        if not os.path.exists(txt_path):
            return None 

        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        # VoxCeleb headers usually end around line 7. We look for the data start.
        # Data lines start with an integer frame number.
        for line in lines:
            parts = line.strip().split()
            # simple check: need 5 parts and first part must be digit
            if len(parts) < 5 or not parts[0].isdigit():
                continue
            
            # Parse: FRAME X Y W H
            frame_idx = int(parts[0])
            x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tracks[frame_idx] = (x, y, w, h)
            
        return tracks

    def load_video_frames(self, video_path, start_time):
        """
        Efficiently seeks to start_time and reads exactly target_frames.
        Returns: Tensor [Channels, Time, H, W]
        """
        cap = cv2.VideoCapture(str(video_path))
        
        # 1. Locate Face Tracks
        # Assumes .txt file is in the same folder as .mp4 with same name
        # e.g., .../id00017/video.mp4 -> .../id00017/video.txt
        txt_path = str(video_path).replace(".mp4", ".txt")
        face_tracks = self.load_face_tracks(txt_path)
        
        # 2. Get Video Properties for Denormalization
        vid_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        vid_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        if vid_fps == 0 or np.isnan(vid_fps): 
            vid_fps = 25.0
            
        # 3. Calculate Start Frame Index
        start_frame_idx = int(start_time * vid_fps)
        
        # 4. Seek to exact frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        
        frames = []
        for i in range(self.target_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            curr_frame_idx = start_frame_idx + i
            
            # --- CROP LOGIC ---
            # Default to full frame if track missing
            final_frame = frame 
            
            if face_tracks and curr_frame_idx in face_tracks:
                # Get normalized coords
                x, y, w, h = face_tracks[curr_frame_idx]
                
                # Convert to pixels
                x_px = int(x * vid_w)
                y_px = int(y * vid_h)
                w_px = int(w * vid_w)
                h_px = int(h * vid_h)
                
                # Safety Clamp (prevent crashing on edge pixels)
                x_px = max(0, min(x_px, int(vid_w)))
                y_px = max(0, min(y_px, int(vid_h)))
                w_px = max(1, min(w_px, int(vid_w) - x_px)) # Ensure width >= 1
                h_px = max(1, min(h_px, int(vid_h) - y_px)) # Ensure height >= 1
                
                # Perform Crop
                crop = frame[y_px:y_px+h_px, x_px:x_px+w_px]
                
                # Check if crop is valid (not empty)
                if crop.size > 0:
                    final_frame = crop
            
            # ------------------
            
            # 5. Preprocessing
            # BGR to RGB
            final_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
            # Resize (Standard for LipReading is 112x112)
            final_frame = cv2.resize(final_frame, self.video_size)
            frames.append(final_frame)
            
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

        # 4. Apply Spatial Augmentation (NEW!)
        if self.apply_spatial_aug:
            # Check if spatial metadata exists in CSV
            if all(col in row for col in ['target_azimuth', 'target_elevation', 'target_distance']):
                video_tensor = self.augmenter.augment_video(
                    video_tensor,
                    azimuth=float(row['target_azimuth']),
                    elevation=float(row['target_elevation']),
                    distance=float(row['target_distance'])
                )
            else:
                print(f"Warning: Spatial metadata missing for sample {idx}, skipping augmentation")

        # 5. Final Verification (Optional, removed for speed but good for debug)
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

        # 6. Prepare Spatial Metadata for Training
        spatial_meta = {}
        if all(col in row for col in ['target_azimuth', 'target_elevation', 'target_distance']):
            spatial_meta = {
                'azimuth': float(row['target_azimuth']),
                'elevation': float(row['target_elevation']),
                'distance': float(row['target_distance']),
                'noise_azimuth': float(row.get('noise_azimuth', 0.0)),
                'noise_elevation': float(row.get('noise_elevation', 0.0)),
                'rt60': float(row.get('rt60', 0.3)),
                'snr_db': float(row.get('snr_db', 0.0))
            }

        return mixed_audio, clean_audio, video_tensor, spatial_meta

# --- SELF-TEST BLOCK ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys

    TEST_CSV = "/run/media/neuronetix/BACKUP/Dataset/VOX/manual/dev/multich/train.csv"
    
    if not os.path.exists(TEST_CSV):
        print(f"Error: Could not find {TEST_CSV}")
        sys.exit(1)

    print("Testing IsoNetDataset with Face Cropping...")
    dataset = IsoNetDataset(TEST_CSV)
    print(f"Dataset Length: {len(dataset)}")

    # 1. Load one sample
    print("Loading Sample #0...")
    mixed, clean, video = dataset[0]

    # 2. Verify Shapes
    print("\n--- Tensor Shapes ---")
    print(f"Mixed Audio: {mixed.shape}  (Expected: [4, 64000])")
    print(f"Clean Audio: {clean.shape}  (Expected: [1, 64000])")
    print(f"Video:       {video.shape}  (Expected: [3, 100, 112, 112])")

    # 3. Save Visual Check with Multiple Frames
    print("\n--- Saving Cropped Face Samples ---")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Face Cropping Test: 10 Sample Frames', fontsize=16)
    
    # Sample 10 frames evenly across the clip
    sample_frames = np.linspace(0, video.shape[1]-1, 10, dtype=int)
    
    for idx, frame_num in enumerate(sample_frames):
        row = idx // 5
        col = idx % 5
        
        # Permute from [C, T, H, W] -> [H, W, C] for display
        frame_tensor = video[:, frame_num, :, :].permute(1, 2, 0)
        axes[row, col].imshow(frame_tensor.numpy())
        axes[row, col].set_title(f'Frame {frame_num}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig("cropped_face_check.png", dpi=100, bbox_inches='tight')
    print("✅ Saved 'cropped_face_check.png'")
    print("   Open it to verify face cropping works correctly!")
    
    # Also save single frame for quick check
    frame_tensor = video[:, 50, :, :].permute(1, 2, 0)
    plt.imsave("debug_single_face.png", frame_tensor.numpy())
    print("✅ Saved 'debug_single_face.png' (frame 50)")