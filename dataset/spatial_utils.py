"""
Spatial Utilities for RIR Simulation and Training

Helper functions for working with 3D spatial coordinates, angular calculations,
and spatial metadata analysis.
"""

import numpy as np
import pandas as pd
import torch
from typing import Tuple, Union


def compute_angular_distance(az1: float, el1: float, az2: float, el2: float) -> float:
    """
    Compute angular distance between two points in spherical coordinates.
    
    Uses the haversine formula for great circle distance on a unit sphere.
    
    Args:
        az1: Azimuth of point 1 (radians)
        el1: Elevation of point 1 (radians)
        az2: Azimuth of point 2 (radians)
        el2: Elevation of point 2 (radians)
        
    Returns:
        Angular distance in radians (0 to π)
    """
    # Convert to Cartesian coordinates on unit sphere
    x1 = np.cos(el1) * np.cos(az1)
    y1 = np.cos(el1) * np.sin(az1)
    z1 = np.sin(el1)
    
    x2 = np.cos(el2) * np.cos(az2)
    y2 = np.cos(el2) * np.sin(az2)
    z2 = np.sin(el2)
    
    # Dot product
    dot = x1*x2 + y1*y2 + z1*z2
    dot = np.clip(dot, -1.0, 1.0)  # Numerical stability
    
    # Angular distance
    return np.arccos(dot)


def spherical_to_cartesian(azimuth: float, elevation: float, distance: float) -> np.ndarray:
    """
    Convert spherical coordinates to Cartesian.
    
    Args:
        azimuth: Horizontal angle (radians)
        elevation: Vertical angle (radians)
        distance: Distance from origin (meters)
        
    Returns:
        3D Cartesian coordinates [x, y, z]
    """
    x = distance * np.cos(elevation) * np.cos(azimuth)
    y = distance * np.cos(elevation) * np.sin(azimuth)
    z = distance * np.sin(elevation)
    return np.array([x, y, z])


def cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert Cartesian coordinates to spherical.
    
    Args:
        x, y, z: Cartesian coordinates
        
    Returns:
        (azimuth, elevation, distance) in radians and meters
    """
    distance = np.sqrt(x**2 + y**2 + z**2)
    
    if distance < 1e-8:
        return 0.0, 0.0, 0.0
    
    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(z / distance)
    
    return azimuth, elevation, distance


def add_angular_separation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add angular separation column to metadata DataFrame.
    
    Args:
        df: DataFrame with target_azimuth, target_elevation, noise_azimuth, noise_elevation
        
    Returns:
        DataFrame with added 'angular_separation' column
    """
    df = df.copy()
    
    df['angular_separation'] = df.apply(
        lambda row: compute_angular_distance(
            row['target_azimuth'], row['target_elevation'],
            row['noise_azimuth'], row['noise_elevation']
        ),
        axis=1
    )
    
    return df


def filter_by_angular_separation(
    df: pd.DataFrame, 
    min_sep: float = 0.0, 
    max_sep: float = np.pi
) -> pd.DataFrame:
    """
    Filter dataset by angular separation between target and noise.
    
    Args:
        df: Metadata DataFrame
        min_sep: Minimum angular separation (radians)
        max_sep: Maximum angular separation (radians)
        
    Returns:
        Filtered DataFrame
    """
    if 'angular_separation' not in df.columns:
        df = add_angular_separation(df)
    
    return df[(df['angular_separation'] >= min_sep) & 
              (df['angular_separation'] <= max_sep)]


def filter_by_spatial_region(
    df: pd.DataFrame,
    azimuth_range: Tuple[float, float] = (0, 2*np.pi),
    elevation_range: Tuple[float, float] = (-np.pi/2, np.pi/2),
    distance_range: Tuple[float, float] = (0, np.inf)
) -> pd.DataFrame:
    """
    Filter dataset by spatial region for target speaker.
    
    Args:
        df: Metadata DataFrame
        azimuth_range: (min, max) azimuth in radians
        elevation_range: (min, max) elevation in radians
        distance_range: (min, max) distance in meters
        
    Returns:
        Filtered DataFrame
    """
    mask = (
        (df['target_azimuth'] >= azimuth_range[0]) &
        (df['target_azimuth'] <= azimuth_range[1]) &
        (df['target_elevation'] >= elevation_range[0]) &
        (df['target_elevation'] <= elevation_range[1]) &
        (df['target_distance'] >= distance_range[0]) &
        (df['target_distance'] <= distance_range[1])
    )
    
    return df[mask]


def create_spatial_bins(df: pd.DataFrame, num_azimuth_bins: int = 8, num_elevation_bins: int = 4):
    """
    Create spatial bin labels for stratified sampling.
    
    Args:
        df: Metadata DataFrame
        num_azimuth_bins: Number of azimuth bins
        num_elevation_bins: Number of elevation bins
        
    Returns:
        DataFrame with 'spatial_bin' column
    """
    df = df.copy()
    
    # Bin edges
    az_bins = np.linspace(0, 2*np.pi, num_azimuth_bins + 1)
    el_bins = np.linspace(-np.pi/6, np.pi/6, num_elevation_bins + 1)
    
    # Digitize
    df['az_bin'] = np.digitize(df['target_azimuth'], az_bins) - 1
    df['el_bin'] = np.digitize(df['target_elevation'], el_bins) - 1
    
    # Combined bin label
    df['spatial_bin'] = df['az_bin'] * num_elevation_bins + df['el_bin']
    
    return df


def compute_tdoa(
    azimuth: float, 
    elevation: float, 
    mic_positions: np.ndarray,
    speed_of_sound: float = 343.0
) -> np.ndarray:
    """
    Compute Time Difference of Arrival (TDOA) for given direction.
    
    Args:
        azimuth: Source azimuth (radians)
        elevation: Source elevation (radians)
        mic_positions: Microphone positions [3, M] where M is number of mics
        speed_of_sound: Speed of sound in m/s
        
    Returns:
        TDOA values [M] relative to first microphone
    """
    # Direction vector (unit vector)
    direction = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation)
    ])
    
    # Project mic positions onto direction
    projections = direction @ mic_positions  # [M]
    
    # TDOA = projection difference / speed of sound
    tdoa = (projections - projections[0]) / speed_of_sound
    
    return tdoa


def create_curriculum_schedule(
    total_epochs: int,
    start_min_separation: float = np.pi * 0.75,  # Start with 135° separation
    end_min_separation: float = np.pi * 0.1,     # End with 18° separation
    schedule_type: str = 'linear'
) -> np.ndarray:
    """
    Create curriculum schedule for minimum angular separation.
    
    Args:
        total_epochs: Total number of training epochs
        start_min_separation: Initial minimum separation (radians)
        end_min_separation: Final minimum separation (radians)
        schedule_type: 'linear', 'exponential', or 'cosine'
        
    Returns:
        Array of minimum separations for each epoch
    """
    epochs = np.arange(total_epochs)
    
    if schedule_type == 'linear':
        schedule = np.linspace(start_min_separation, end_min_separation, total_epochs)
    
    elif schedule_type == 'exponential':
        # Exponential decay
        decay_rate = np.log(end_min_separation / start_min_separation) / total_epochs
        schedule = start_min_separation * np.exp(decay_rate * epochs)
    
    elif schedule_type == 'cosine':
        # Cosine annealing
        schedule = end_min_separation + 0.5 * (start_min_separation - end_min_separation) * \
                   (1 + np.cos(np.pi * epochs / total_epochs))
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    return schedule


class SpatialMetadataEncoder(torch.nn.Module):
    """
    Neural network encoder for spatial metadata.
    
    Encodes spatial parameters (azimuth, elevation, distance, angular separation)
    into a dense embedding for model conditioning.
    """
    
    def __init__(self, input_dim: int = 7, hidden_dim: int = 64, output_dim: int = 128):
        """
        Args:
            input_dim: Number of spatial features (default 7: target az/el/dist, 
                       noise az/el/dist, angular_sep)
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
        """
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_features: [Batch, input_dim] tensor of spatial parameters
            
        Returns:
            Spatial embeddings [Batch, output_dim]
        """
        return self.encoder(spatial_features)


def prepare_spatial_features(
    metadata_row: Union[pd.Series, dict],
    normalize: bool = True
) -> torch.Tensor:
    """
    Prepare spatial features for model input.
    
    Args:
        metadata_row: Row from metadata DataFrame or dict with spatial fields
        normalize: Whether to normalize features to [-1, 1] range
        
    Returns:
        Tensor of spatial features [7]
    """
    features = torch.tensor([
        metadata_row['target_azimuth'],
        metadata_row['target_elevation'],
        metadata_row['target_distance'],
        metadata_row['noise_azimuth'],
        metadata_row['noise_elevation'],
        metadata_row['noise_distance'],
        compute_angular_distance(
            metadata_row['target_azimuth'], metadata_row['target_elevation'],
            metadata_row['noise_azimuth'], metadata_row['noise_elevation']
        )
    ], dtype=torch.float32)
    
    if normalize:
        # Normalize to roughly [-1, 1]
        # Azimuths: [0, 2π] -> [-1, 1]
        features[0] = (features[0] / np.pi) - 1.0
        features[3] = (features[3] / np.pi) - 1.0
        
        # Elevations: [-π/6, π/6] -> [-1, 1]
        features[1] = features[1] / (np.pi/6)
        features[4] = features[4] / (np.pi/6)
        
        # Distances: [0.8, 1.5] -> [-1, 1]
        features[2] = (features[2] - 1.15) / 0.35
        features[5] = (features[5] - 1.15) / 0.35
        
        # Angular separation: [0, π] -> [0, 1]
        features[6] = features[6] / np.pi
    
    return features


if __name__ == "__main__":
    # Example usage
    print("Spatial Utils - Example Usage")
    print("=" * 60)
    
    # Test angular distance
    az1, el1 = 0.0, 0.0  # Front
    az2, el2 = np.pi, 0.0  # Back
    
    dist = compute_angular_distance(az1, el1, az2, el2)
    print(f"\nAngular distance between front (0°) and back (180°):")
    print(f"  {dist:.4f} rad = {np.degrees(dist):.1f}°")
    
    # Test coordinate conversion
    az, el, d = np.pi/4, np.pi/12, 1.2
    xyz = spherical_to_cartesian(az, el, d)
    az2, el2, d2 = cartesian_to_spherical(*xyz)
    
    print(f"\nCoordinate conversion test:")
    print(f"  Input:  az={np.degrees(az):.1f}°, el={np.degrees(el):.1f}°, d={d:.2f}m")
    print(f"  XYZ:    [{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}]")
    print(f"  Output: az={np.degrees(az2):.1f}°, el={np.degrees(el2):.1f}°, d={d2:.2f}m")
    
    # Test curriculum schedule
    schedule = create_curriculum_schedule(100, schedule_type='cosine')
    print(f"\nCurriculum schedule (100 epochs, cosine):")
    print(f"  Epoch 1:   min_sep = {schedule[0]:.3f} rad ({np.degrees(schedule[0]):.1f}°)")
    print(f"  Epoch 50:  min_sep = {schedule[49]:.3f} rad ({np.degrees(schedule[49]):.1f}°)")
    print(f"  Epoch 100: min_sep = {schedule[99]:.3f} rad ({np.degrees(schedule[99]):.1f}°)")
