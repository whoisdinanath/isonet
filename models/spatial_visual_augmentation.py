"""
Spatial-Visual Augmentation for Camera-Aware Training

This module creates visual-spatial correspondence by transforming video frames
to match the speaker's spatial position (azimuth, elevation) relative to the
microphone array.

Key Idea:
    - Speaker at azimuth=45° (right) → Face appears RIGHT in camera frame
    - Speaker at elevation=15° (above) → Face appears HIGHER in camera frame
    - This creates natural visual-spatial mapping for real-world deployment
"""

import numpy as np
import torch
import cv2
from typing import Tuple, Optional


class SpatialVisualAugmenter:
    """
    Augment video frames to create visual-spatial correspondence.
    
    Maps speaker's spatial position (azimuth, elevation) to their position
    in the camera frame, creating a natural visual cue for beamforming.
    """
    
    def __init__(
        self,
        camera_fov_horizontal: float = 90.0,  # Camera horizontal FOV in degrees
        camera_fov_vertical: float = 60.0,    # Camera vertical FOV in degrees
        output_size: Tuple[int, int] = (224, 224),
        augmentation_strength: float = 1.0,   # 0=none, 1=full spatial mapping
    ):
        """
        Args:
            camera_fov_horizontal: Horizontal field of view in degrees
            camera_fov_vertical: Vertical field of view in degrees
            output_size: Output frame size (H, W)
            augmentation_strength: How much to apply spatial augmentation (0-1)
        """
        self.fov_h = np.radians(camera_fov_horizontal)
        self.fov_v = np.radians(camera_fov_vertical)
        self.output_size = output_size
        self.aug_strength = augmentation_strength
    
    def spatial_to_frame_position(
        self,
        azimuth: float,
        elevation: float,
        distance: float = 1.0
    ) -> Tuple[float, float]:
        """
        Convert spatial angles to normalized frame position.
        
        Args:
            azimuth: Horizontal angle in radians (0 = front, π/2 = right)
            elevation: Vertical angle in radians (0 = level, π/6 = up)
            distance: Distance from array (optional, for depth scaling)
        
        Returns:
            (x_norm, y_norm): Position in frame, normalized to [-1, 1]
                x_norm: -1 (left) to +1 (right)
                y_norm: -1 (bottom) to +1 (top)
        """
        # Normalize azimuth to [-π, π] with 0 as front-center
        azimuth_centered = azimuth
        if azimuth_centered > np.pi:
            azimuth_centered -= 2 * np.pi
        
        # Map to frame coordinates based on FOV
        # x: azimuth → horizontal position
        x_norm = np.clip(azimuth_centered / (self.fov_h / 2), -1, 1)
        
        # y: elevation → vertical position (invert because image y is top-down)
        y_norm = -np.clip(elevation / (self.fov_v / 2), -1, 1)
        
        return x_norm, y_norm
    
    def augment_frame(
        self,
        frame: np.ndarray,
        azimuth: float,
        elevation: float,
        distance: float = 1.0
    ) -> np.ndarray:
        """
        Augment a single frame by shifting face to match spatial position.
        
        Args:
            frame: Input frame [H, W, C] in range [0, 255] or [0, 1]
            azimuth: Speaker azimuth in radians
            elevation: Speaker elevation in radians
            distance: Speaker distance in meters (for scale)
        
        Returns:
            Augmented frame [H, W, C]
        """
        H, W = frame.shape[:2]
        
        # Get target position in frame
        x_norm, y_norm = self.spatial_to_frame_position(azimuth, elevation, distance)
        
        # Apply augmentation strength
        x_norm *= self.aug_strength
        y_norm *= self.aug_strength
        
        # Convert normalized position to pixel shifts
        # Shift face toward target position (assuming face starts centered)
        shift_x = int(x_norm * W * 0.3)  # Max 30% of width
        shift_y = int(y_norm * H * 0.3)  # Max 30% of height
        
        # Create translation matrix
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        
        # Apply affine transformation
        augmented = cv2.warpAffine(
            frame,
            M,
            (W, H),
            borderMode=cv2.BORDER_REFLECT_101  # Reflect to avoid black borders
        )
        
        return augmented
    
    def augment_video(
        self,
        video: torch.Tensor,
        azimuth: float,
        elevation: float,
        distance: float = 1.0
    ) -> torch.Tensor:
        """
        Augment entire video tensor.
        
        Args:
            video: Video tensor [C, T, H, W] or [T, H, W, C]
            azimuth: Speaker azimuth in radians
            elevation: Speaker elevation in radians
            distance: Speaker distance in meters
        
        Returns:
            Augmented video tensor (same shape as input)
        """
        # Detect input format
        if video.shape[0] == 3:  # [C, T, H, W]
            video_np = video.permute(1, 2, 3, 0).numpy()  # [T, H, W, C]
        else:  # [T, H, W, C]
            video_np = video.numpy()
        
        # Process each frame
        augmented_frames = []
        for frame in video_np:
            # Convert to 0-255 if needed
            if frame.max() <= 1.0:
                frame_uint8 = (frame * 255).astype(np.uint8)
            else:
                frame_uint8 = frame.astype(np.uint8)
            
            # Augment
            aug_frame = self.augment_frame(frame_uint8, azimuth, elevation, distance)
            
            # Convert back to original range
            if video_np.max() <= 1.0:
                aug_frame = aug_frame.astype(np.float32) / 255.0
            
            augmented_frames.append(aug_frame)
        
        # Stack and convert back to tensor
        augmented_np = np.stack(augmented_frames)
        augmented_tensor = torch.from_numpy(augmented_np)
        
        # Restore original format
        if video.shape[0] == 3:  # Convert back to [C, T, H, W]
            augmented_tensor = augmented_tensor.permute(3, 0, 1, 2)
        
        return augmented_tensor


class SpatialScaleAugmenter:
    """
    Augment face scale based on distance (closer = larger face).
    """
    
    def __init__(self, min_distance: float = 0.8, max_distance: float = 1.5):
        self.min_dist = min_distance
        self.max_dist = max_distance
    
    def scale_frame(self, frame: np.ndarray, distance: float) -> np.ndarray:
        """
        Scale frame based on speaker distance.
        
        Closer speakers (small distance) → zoom in (larger face)
        Farther speakers (large distance) → zoom out (smaller face)
        """
        H, W = frame.shape[:2]
        
        # Map distance to scale factor
        # distance: [0.8, 1.5] → scale: [1.3, 0.9]
        scale = 1.0 + (self.max_dist - distance) / (self.max_dist - self.min_dist) * 0.4
        scale = np.clip(scale, 0.8, 1.5)
        
        # Compute new dimensions
        new_H, new_W = int(H * scale), int(W * scale)
        
        # Resize
        resized = cv2.resize(frame, (new_W, new_H))
        
        # Center crop or pad to original size
        if scale > 1.0:  # Zoom in (crop)
            start_y = (new_H - H) // 2
            start_x = (new_W - W) // 2
            cropped = resized[start_y:start_y+H, start_x:start_x+W]
            return cropped
        else:  # Zoom out (pad)
            pad_y = (H - new_H) // 2
            pad_x = (W - new_W) // 2
            padded = cv2.copyMakeBorder(
                resized,
                pad_y, H - new_H - pad_y,
                pad_x, W - new_W - pad_x,
                cv2.BORDER_REFLECT_101
            )
            return padded


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_augmentation():
    """
    Example: How to use spatial augmentation in training.
    """
    # Initialize augmenter
    augmenter = SpatialVisualAugmenter(
        camera_fov_horizontal=90.0,
        camera_fov_vertical=60.0,
        augmentation_strength=1.0  # Full spatial mapping
    )
    
    # Example metadata from dataset
    azimuth = np.radians(45)  # Speaker is 45° to the right
    elevation = np.radians(15)  # Speaker is 15° above level
    distance = 1.2  # 1.2 meters away
    
    # Load video (example shape: [3, 100, 224, 224])
    # In practice, this comes from your dataloader
    video = torch.randn(3, 100, 224, 224)  # Dummy data
    
    # Apply spatial augmentation
    augmented_video = augmenter.augment_video(video, azimuth, elevation, distance)
    
    print(f"Original shape: {video.shape}")
    print(f"Augmented shape: {augmented_video.shape}")
    print(f"Face shifted to match azimuth={np.degrees(azimuth):.0f}°, elevation={np.degrees(elevation):.0f}°")
    
    return augmented_video


# ============================================================================
# VISUAL-SPATIAL CONSISTENCY LOSS
# ============================================================================

def visual_spatial_consistency_loss(
    visual_features: torch.Tensor,
    spatial_features: torch.Tensor,
    target_azimuth: torch.Tensor,
    target_elevation: torch.Tensor
) -> torch.Tensor:
    """
    Auxiliary loss to enforce visual-spatial correspondence.
    
    Encourages the model to learn that:
    - Visual features should encode spatial position
    - Spatial features should align with visual position
    
    Args:
        visual_features: [Batch, Dim] from visual stream
        spatial_features: [Batch, Dim] from spatial stream
        target_azimuth: [Batch] ground truth azimuth
        target_elevation: [Batch] ground truth elevation
    
    Returns:
        Consistency loss (scalar)
    """
    # Predict spatial angles from visual features
    # This requires adding a small prediction head in your model
    # For now, compute cosine similarity between visual and spatial features
    
    # Normalize features
    visual_norm = torch.nn.functional.normalize(visual_features, dim=1)
    spatial_norm = torch.nn.functional.normalize(spatial_features, dim=1)
    
    # Cosine similarity
    similarity = torch.sum(visual_norm * spatial_norm, dim=1)  # [Batch]
    
    # Loss: maximize similarity (minimize negative similarity)
    loss = -torch.mean(similarity)
    
    return loss


if __name__ == "__main__":
    print("Spatial-Visual Augmentation Example")
    print("=" * 60)
    
    # Run example
    augmented = example_augmentation()
    
    print("\nKey Concepts:")
    print("1. Azimuth (horizontal) → Left/Right position in frame")
    print("2. Elevation (vertical) → Up/Down position in frame")
    print("3. Distance → Face scale (zoom)")
    print("\nThis creates visual-spatial correspondence for camera-aware beamforming!")
