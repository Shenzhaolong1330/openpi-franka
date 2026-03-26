#!/usr/bin/env python3
"""
Merge split video files in a LeRobot dataset into single files.
This fixes the issue where LeRobot incorrectly calculates frame indices
when videos are split into multiple files.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def get_video_frames(video_path: Path) -> int:
    """Get the number of frames in a video file."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", 
         "-count_packets", "-show_entries", "stream=nb_read_packets", 
         "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True
    )
    output = result.stdout.strip()
    if not output:
        raise ValueError(f"Could not get frame count for {video_path}: {result.stderr}")
    return int(output)


def merge_videos(video_dir: Path, output_path: Path, codec: str = "libx264", crf: int = 23):
    """Merge all mp4 files in a directory into a single video."""
    # Only process file-000.mp4, file-001.mp4, etc. (skip merged files)
    video_files = sorted([f for f in video_dir.glob("file-*.mp4") if not f.name.startswith("file-merged")])
    
    if len(video_files) == 0:
        print(f"  No video files found in {video_dir}")
        return False
    
    if len(video_files) == 1:
        print(f"  Only one video file, no merge needed")
        return False
    
    print(f"  Found {len(video_files)} video files to merge")
    for vf in video_files:
        frames = get_video_frames(vf)
        print(f"    {vf.name}: {frames} frames")
    
    # Create concat file list
    concat_file = video_dir / "concat_list.txt"
    with open(concat_file, "w") as f:
        for vf in video_files:
            f.write(f"file '{vf.name}'\n")
    
    # Merge videos - use fast encoding
    print(f"  Merging videos...")
    temp_output = video_dir / "file-merged.mp4"
    
    # Use libx264 with fast preset for speed
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c:v", codec, "-crf", str(crf), "-preset", "fast",
        str(temp_output)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error merging videos: {result.stderr}")
        concat_file.unlink()
        return False
    
    # Verify merged video
    merged_frames = get_video_frames(temp_output)
    print(f"  Merged video: {merged_frames} frames")
    
    # Backup original files
    backup_dir = video_dir / "backup"
    backup_dir.mkdir(exist_ok=True)
    for vf in video_files:
        shutil.move(str(vf), str(backup_dir / vf.name))
    
    # Rename merged file to file-000.mp4
    temp_output.rename(video_dir / "file-000.mp4")
    
    # Cleanup
    concat_file.unlink()
    
    print(f"  Done! Original files backed up to {backup_dir}")
    return True


def update_episode_metadata(dataset_path: Path, video_keys: list[str]):
    """Update episode metadata to point to file-000.mp4 for all episodes."""
    meta_path = dataset_path / "meta" / "episodes"
    
    for chunk_dir in sorted(meta_path.iterdir()):
        if not chunk_dir.is_dir():
            continue
        
        for parquet_file in sorted(chunk_dir.glob("*.parquet")):
            import pandas as pd
            
            df = pd.read_parquet(parquet_file)
            modified = False
            
            for video_key in video_keys:
                file_index_col = f"videos/{video_key}/file_index"
                if file_index_col in df.columns:
                    # Set all file_index to 0
                    if (df[file_index_col] != 0).any():
                        df[file_index_col] = 0
                        modified = True
                        print(f"  Updated {file_index_col} in {parquet_file.name}")
            
            if modified:
                df.to_parquet(parquet_file, index=False)
                print(f"  Saved updated {parquet_file}")


def main():
    parser = argparse.ArgumentParser(description="Merge split video files in a LeRobot dataset")
    parser.add_argument("dataset_path", type=Path, help="Path to the LeRobot dataset")
    parser.add_argument("--codec", default="libaom-av1", help="Video codec for merged output")
    parser.add_argument("--crf", type=int, default=30, help="CRF value for video encoding")
    parser.add_argument("--skip-metadata", action="store_true", help="Skip updating episode metadata")
    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    videos_path = dataset_path / "videos"
    
    if not videos_path.exists():
        print(f"Error: Videos directory not found at {videos_path}")
        return
    
    # Find all video keys
    video_keys = [d.name for d in videos_path.iterdir() if d.is_dir()]
    print(f"Found video keys: {video_keys}")
    
    # Merge videos for each video key
    for video_key in video_keys:
        print(f"\nProcessing {video_key}:")
        video_dir = videos_path / video_key / "chunk-000"
        if video_dir.exists():
            merge_videos(video_dir, video_dir / "file-merged.mp4", args.codec, args.crf)
    
    # Update episode metadata
    if not args.skip_metadata:
        print(f"\nUpdating episode metadata...")
        update_episode_metadata(dataset_path, video_keys)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
