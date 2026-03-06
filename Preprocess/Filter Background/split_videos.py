import cv2
import os
import subprocess
import json

# Load configuration from the config file
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..\\config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

GRID_ROWS = config["grid_rows"]
GRID_COLS = config["grid_cols"]
NUM_CAMS = config["num_cams"]

# Define which cameras to save (0 = don't save, 1 = save)
CAMERA_SAVE_FLAGS = [1,0,0,0,0,0,0,0,0]  # For example: Save cameras 1, 3, and 4, but skip camera 2

# Path to the input video
video_path = r"C:\Users\EMC 1\Documents\GitHub\3D-vessel-imaging\OBS Videos\2024-09-10 17-15-32.mkv"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the video frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate the width and height of each sub-video
sub_width = frame_width // GRID_COLS
sub_height = frame_height // GRID_ROWS

# Prepare directories for output frames and video
output_dir = "OBS Videos\\" + video_path.split('\\')[-1] + "_split"
os.makedirs(output_dir, exist_ok=True)

temp_frame_dir = "temp_frames"
os.makedirs(temp_frame_dir, exist_ok=True)

frame_counter = 0

# Process the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    for cam in range(NUM_CAMS):
        # Only process the cameras that have a 1 in the CAMERA_SAVE_FLAGS array
        if CAMERA_SAVE_FLAGS[cam] == 1:
            row = cam // GRID_COLS
            col = cam % GRID_COLS
            # Extract sub-image corresponding to each camera
            sub_frame = frame[row * sub_height:(row + 1) * sub_height, col * sub_width:(col + 1) * sub_width]
            
            # Save individual frames as images (for each camera)
            frame_filename = os.path.join(temp_frame_dir, f"camera_{cam+1}_frame_{frame_counter:05d}.png")
            cv2.imwrite(frame_filename, sub_frame)
    
    frame_counter += 1

# Release the capture object
cap.release()

# Now use FFMPEG to create videos for each camera feed with lossless FFV1 codec
for cam in range(NUM_CAMS):
    # Only process the cameras that have a 1 in the CAMERA_SAVE_FLAGS array
    if CAMERA_SAVE_FLAGS[cam] == 1:
        output_video_filename = os.path.join(output_dir, f"camera_{cam+1}.mkv")
        input_frame_pattern = os.path.join(temp_frame_dir, f"camera_{cam+1}_frame_%05d.png")
        
        ffmpeg_command = [
            'ffmpeg', '-framerate', '30', '-i', input_frame_pattern,
            '-c:v', 'ffv1', '-preset', 'veryslow', '-level', '3', output_video_filename
        ]
        
        subprocess.run(ffmpeg_command)

# Clean up temporary frames
for file in os.listdir(temp_frame_dir):
    os.remove(os.path.join(temp_frame_dir, file))
os.rmdir(temp_frame_dir)

print("Video splitting completed with lossless FFV1 encoding using FFMPEG!")
