import os
import cv2
import re

def modify_video_with_frames(video_path, frames_dir, output_video_path, fps=30):
    # Open the original video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = fps or original_fps  # Use original FPS if not specified

    # Get a sorted list of augmented frames
    augmented_frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # Map frame numbers to augmented frames based on their filenames
    augmented_frame_mapping = {}
    for frame_path in augmented_frames:
        # Extract frame number using regex to handle 'frameX' naming
        match = re.search(r'frame(\d+)', os.path.basename(frame_path))
        if match:
            frame_number = int(match.group(1))
            augmented_frame_mapping[frame_number] = frame_path

    # Create a writer for the output video
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Process each frame
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {frame_idx}. Skipping...")
            continue

        # Check if the current frame has an augmented replacement
        if frame_idx in augmented_frame_mapping:
            print(f"Replacing frame {frame_idx} with augmented frame.")
            augmented_frame = cv2.imread(augmented_frame_mapping[frame_idx])
            # Ensure the augmented frame has the correct dimensions
            augmented_frame = cv2.resize(augmented_frame, (frame_width, frame_height))
            frame = augmented_frame

        # Write the frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Modified video saved to {output_video_path}")


# Example usage
original_video = "NIKE  - JUST DO IT | Spec ad.mp4"
augmented_frames_dir = "runs/detect/nikeAdTest"
output_video = "nike_done.mp4"

#modify_video_with_frames(original_video, augmented_frames_dir, output_video)