"""
Script to draw bounding boxes on VoxCeleb2 videos using the provided bbox annotations.

Usage:
    python draw_bboxes.py --video_path <path_to_video> --output <output_path>
    python draw_bboxes.py --csv <train.csv> --sample_idx <index> --output <output_path>
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import pandas as pd
from typing import Dict, List, Tuple


def parse_bbox_file(bbox_path: str) -> Dict[int, Tuple[float, float, float, float]]:
    """
    Parse VoxCeleb2 bbox text file.
    
    Args:
        bbox_path: Path to .txt file with bounding box annotations
        
    Returns:
        Dictionary mapping frame_number -> (x, y, w, h) in normalized coordinates [0, 1]
    """
    bboxes = {}
    
    with open(bbox_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines (first 7 lines)
    # Format: FRAME   X       Y       W       H
    for line in lines[7:]:
        parts = line.strip().split()
        if len(parts) == 5:
            frame_num = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            bboxes[frame_num] = (x, y, w, h)
    
    return bboxes


def get_bbox_path_from_video(video_path: str, bbox_root: str) -> str:
    """
    Convert video path to corresponding bbox txt path.
    
    Example:
        video: /data/mp4/id00019/abc123/00001.mp4
        bbox:  /data/vox2_dev_txt/txt/id00019/abc123/00001.txt
    """
    video_path = Path(video_path)
    identity = video_path.parts[-3]  # id00019
    reference = video_path.parts[-2]  # abc123
    basename = video_path.stem       # 00001
    
    bbox_path = Path(bbox_root) / identity / reference / f"{basename}.txt"
    return str(bbox_path)


def draw_bbox_on_frame(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: str = None
) -> np.ndarray:
    """
    Draw bounding box on video frame.
    
    Args:
        frame: Video frame (H, W, 3) BGR
        bbox: (x_center, y_center, w, h) in normalized coordinates [0, 1] (VoxCeleb format)
        color: BGR color tuple
        thickness: Line thickness
        label: Optional text label
        
    Returns:
        Frame with bounding box drawn
    """
    h, w = frame.shape[:2]
    x_center, y_center, w_norm, h_norm = bbox
    
    # VoxCeleb2 format: (x_center, y_center, width, height) in normalized coords
    # Y-axis appears to be inverted (0 at bottom)
    y_center = 1.0 - y_center
    
    # Convert to corner coordinates
    x1 = int((x_center - w_norm / 2) * w)
    y1 = int((y_center - h_norm / 2) * h)
    x2 = int((x_center + w_norm / 2) * w)
    y2 = int((y_center + h_norm / 2) * h)
    
    # Ensure coordinates are within frame bounds
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    
    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label if provided
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)
    
    return frame


def process_video_with_bbox(
    video_path: str,
    bbox_path: str,
    output_path: str = None,
    display: bool = False,
    start_time: float = 0.0,
    duration: float = None
) -> None:
    """
    Process video and draw bounding boxes on each frame.
    
    Args:
        video_path: Path to input video
        bbox_path: Path to bbox annotation file
        output_path: Path to save output video (None = display only)
        display: Whether to display video in real-time
        start_time: Start time in seconds (for clipped videos)
        duration: Duration in seconds (None = full video)
    """
    # Parse bounding boxes
    if not Path(bbox_path).exists():
        print(f"Bbox file not found: {bbox_path}")
        return
    
    bboxes = parse_bbox_file(bbox_path)
    print(f"Loaded {len(bboxes)} bounding boxes from {bbox_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame range
    start_frame = int(start_time * fps)
    end_frame = int((start_time + duration) * fps) if duration else total_frames
    
    print(f"Video: {video_path}")
    print(f"  Resolution: {frame_width}x{frame_height}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Processing Frames: {start_frame} to {end_frame}")
    
    # Setup video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"Saving to: {output_path}")
    
    # Get the actual starting frame number from video metadata
    # VoxCeleb videos may be clips from longer videos
    # We need to find which bbox frames correspond to our clip
    
    # First, let's identify the bbox frame range
    bbox_frames = sorted(bboxes.keys())
    if not bbox_frames:
        print("Warning: No bounding boxes found in file")
        cap.release()
        return
    
    print(f"  BBox Frame Range: {bbox_frames[0]} to {bbox_frames[-1]}")
    
    # For VoxCeleb, the start_time corresponds to which part of the original video
    # We need to calculate the absolute frame number
    # Assuming the video clip starts at a certain absolute frame number
    
    # Read the video to find the correlation
    # We'll process frames and match them to bbox frames based on start_time
    frame_count = 0
    processed = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= end_frame:
            break
        
        # Calculate absolute frame number based on start_time
        # The bbox file contains absolute frame numbers from original video
        # The video we have might be a clip, so we need to map:
        # video_frame = frame_count
        # absolute_frame = bbox_frames[0] + frame_count (assuming clip starts at first bbox frame)
        
        # Try to find matching bbox frame
        absolute_frame = None
        
        # Strategy 1: Assume the clip starts at the first bbox frame
        if frame_count < len(bbox_frames):
            absolute_frame = bbox_frames[frame_count]
        
        # Draw bounding box if available for this frame
        if absolute_frame and absolute_frame in bboxes:
            bbox = bboxes[absolute_frame]
            label = f"Frame {frame_count} (Abs: {absolute_frame})"
            frame = draw_bbox_on_frame(frame, bbox, color=(0, 255, 0), thickness=2, label=label)
        else:
            # Frame has no bbox annotation - draw warning
            cv2.putText(frame, f"Frame {frame_count}: No BBox", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Write frame
        if writer:
            writer.write(frame)
        
        # Display frame
        if display:
            cv2.imshow('Video with BBox', frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        processed += 1
        
        # Progress indicator
        if processed % 25 == 0:
            print(f"Processed {processed} frames...", end='\r')
    
    print(f"\nProcessed {processed} frames total")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()


def process_from_csv(
    csv_path: str,
    sample_idx: int,
    bbox_root: str,
    output_path: str = None,
    display: bool = False,
    clip_length: float = 4.0
) -> None:
    """
    Process a video sample from the training CSV.
    
    Args:
        csv_path: Path to train.csv or val.csv
        sample_idx: Row index in CSV
        bbox_root: Root directory for bbox annotations
        output_path: Output video path
        display: Whether to display video
        clip_length: Duration of clip in seconds
    """
    df = pd.read_csv(csv_path)
    
    if sample_idx >= len(df):
        print(f"Index {sample_idx} out of range. CSV has {len(df)} samples.")
        return
    
    row = df.iloc[sample_idx]
    video_path = row['video_path']
    start_time = row['start_time']
    
    print(f"\nProcessing sample {sample_idx}:")
    print(f"  Video: {video_path}")
    print(f"  Start Time: {start_time:.3f}s")
    print(f"  Clip Length: {clip_length}s")
    
    # Get bbox path
    bbox_path = get_bbox_path_from_video(video_path, bbox_root)
    
    # Generate output path if not provided
    if output_path is None:
        output_dir = Path("output_videos")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"sample_{sample_idx:05d}_with_bbox.mp4"
    
    # Process video
    process_video_with_bbox(
        video_path=video_path,
        bbox_path=bbox_path,
        output_path=str(output_path),
        display=display,
        start_time=start_time,
        duration=clip_length
    )


def main():
    parser = argparse.ArgumentParser(description="Draw bounding boxes on VoxCeleb2 videos")
    
    # Mode 1: Direct video processing
    parser.add_argument('--video_path', type=str, help='Path to input video file')
    parser.add_argument('--bbox_path', type=str, help='Path to bbox annotation file')
    
    # Mode 2: Process from CSV
    parser.add_argument('--csv', type=str, help='Path to train.csv or val.csv')
    parser.add_argument('--sample_idx', type=int, help='Sample index in CSV')
    parser.add_argument('--bbox_root', type=str, 
                       default='/mnt/DATA/Bibek/Speech/isolate-speech/data/vox2_dev_txt/txt',
                       help='Root directory for bbox txt files')
    
    # Common options
    parser.add_argument('--output', type=str, help='Output video path')
    parser.add_argument('--display', action='store_true', help='Display video in real-time')
    parser.add_argument('--start_time', type=float, default=0.0, help='Start time in seconds')
    parser.add_argument('--duration', type=float, default=None, help='Duration in seconds')
    parser.add_argument('--clip_length', type=float, default=4.0, help='Clip length for CSV mode')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.csv and args.sample_idx is not None:
        # CSV mode
        process_from_csv(
            csv_path=args.csv,
            sample_idx=args.sample_idx,
            bbox_root=args.bbox_root,
            output_path=args.output,
            display=args.display,
            clip_length=args.clip_length
        )
    elif args.video_path and args.bbox_path:
        # Direct video mode
        process_video_with_bbox(
            video_path=args.video_path,
            bbox_path=args.bbox_path,
            output_path=args.output,
            display=args.display,
            start_time=args.start_time,
            duration=args.duration
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Process sample from CSV:")
        print("  python draw_bboxes.py --csv data/multich/train.csv --sample_idx 0 --display")
        print("\n  # Process video directly:")
        print("  python draw_bboxes.py --video_path data/mp4/id00019/.../00001.mp4 \\")
        print("                        --bbox_path data/vox2_dev_txt/txt/id00019/.../00001.txt \\")
        print("                        --output output.mp4")


if __name__ == "__main__":
    main()
