import cv2
import numpy as np
from pytubefix import YouTube
import io
import imageio.v3 as iio
import os
import SIFT
import RANSAC
from modified_detect import detect
from update_video import modify_video_with_frames

url = "https://www.youtube.com/watch?v=freRkr00otY"

video_name = "nikeAd"
directory = "runs"
full_directory = ""
frame_gap = 30

def create_dir(name, num=0, ):
    current_directory = os.getcwd()
    if num > 0:
        final_directory = os.path.join(current_directory, name + str(num))
    else:
        final_directory = os.path.join(current_directory, name)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
        return name + str(num)
    else:
        return create_dir(name, num+1)

try:
    # Fetch YouTube video
    video = YouTube(url)
    stream = video.streams.filter(file_extension="mp4").first()

    full_directory = f"{directory}/{video_name}"
    full_directory = create_dir(full_directory)
    stream.download(filename=f"{full_directory}/original_video.mp4")


except KeyError:
    print("Unable to fetch video information. Please check the video URL or your network connection.")
    exit(0)

# Convert the stream object into something that cv2 recognizes as a video
buffer = io.BytesIO()
stream.stream_to_buffer(buffer)
buffer.seek(0)

video_frames = []

create_dir(f"{full_directory}/unmodified_frames")
# Use imageio to read frames from the buffer
for frame_index, frame in enumerate(iio.imiter(buffer, plugin="pyav")):
    # Convert frame to OpenCV-compatible format
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Get every (frame_gap)th frame of the video
    if frame_index % frame_gap == 0:
        video_frames.append(frame_bgr)
        cv2.imwrite(f"{full_directory}/unmodified_frames/frame{frame_index}.png", frame_bgr)

detect(f"{full_directory}/unmodified_frames", full_directory, "modified_frames")

modify_video_with_frames(f"{full_directory}/original_video.mp4", f"{full_directory}/modified_frames", f"{full_directory}/nike_done.mp4")