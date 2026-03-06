import subprocess

# File paths
VIDEO_PATH = r"C:\Users\elega\Videos\2026-02-24 14-27-44.mkv"
OUTPUT_PATH = r"C:\Users\elega\Videos\2026-02-24 14-27-44_trimmed.mkv"

# Trim parameters
START_TIME = 2 # Start time in seconds
DURATION = 14     # Duration in seconds

# Step 1: Seek to keyframe (fast), decode from there
# Step 2: Re-encode *just the start*, copy the rest

# subprocess.run([
#     'ffmpeg',
#     '-ss', str(START_TIME),
#     '-i', VIDEO_PATH,
#     '-t', str(DURATION),
#     '-c:v', 'ffv1',        # Lossless video codec
#     '-c:a', 'copy',        # Copy audio
#     OUTPUT_PATH
# ])

# print(f"Trimmed video saved to {OUTPUT_PATH}")

# First, let's check the current video encoding
print("Checking current video encoding...")
result = subprocess.run([
    'ffprobe',
    '-v', 'quiet',
    '-print_format', 'json',
    '-show_format',
    '-show_streams',
    VIDEO_PATH
], capture_output=True, text=True)

print("Video info:")
print(result.stdout)

print(f"\nTrimming video from {START_TIME}s for {DURATION}s...")
print(f"Input: {VIDEO_PATH}")
print(f"Output: {OUTPUT_PATH}")

# Try multiple approaches for precise trimming
try:
    # Approach 1: Precise trimming with minimal re-encoding (best quality/size balance)
    print("Attempting precise trimming with H.264 (frame-accurate)...")
    cmd = [
        'ffmpeg',
        '-ss', str(START_TIME),
        '-i', VIDEO_PATH,
        '-t', str(DURATION),
        '-c:v', 'libx264',     # H.264 codec
        '-preset', 'ultrafast', # Fastest encoding
        '-crf', '23',          # Good quality (23 is default, good balance)
        '-an',                 # No audio (since there's no audio in the source)
        '-avoid_negative_ts', 'make_zero',  # Handle timestamp issues
        OUTPUT_PATH
    ]
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("Successfully trimmed with H.264 (frame-accurate)")
    
except subprocess.CalledProcessError:
    print("H.264 failed, trying stream copy (may not be frame-accurate)...")
    try:
        # Approach 2: Stream copy (fastest but may not be frame-accurate)
        cmd = [
            'ffmpeg',
            '-ss', str(START_TIME),
            '-i', VIDEO_PATH,
            '-t', str(DURATION),
            '-c', 'copy',          # Copy both video and audio streams
            OUTPUT_PATH
        ]
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print("Successfully copied streams (may not be frame-accurate)")
        
    except subprocess.CalledProcessError:
        print("Stream copy failed, trying FFV1 with memory optimization...")
        # Approach 3: FFV1 with memory optimization (lossless but large)
        subprocess.run([
            'ffmpeg',
            '-ss', str(START_TIME),
            '-i', VIDEO_PATH,
            '-t', str(DURATION),
            '-c:v', 'ffv1',        # Lossless video codec
            '-level', '3',         # Lower level = less memory usage
            '-c:a', 'copy',        # Copy audio
            OUTPUT_PATH
        ], check=True)
        print("Successfully encoded with FFV1 (lossless but large file)")

print(f"\nTrimmed video saved to {OUTPUT_PATH}")

# Verify the output file duration
print("\nVerifying output file...")
result = subprocess.run([
    'ffprobe',
    '-v', 'quiet',
    '-print_format', 'json',
    '-show_format',
    OUTPUT_PATH
], capture_output=True, text=True)

print("Output video info:")
print(result.stdout)
