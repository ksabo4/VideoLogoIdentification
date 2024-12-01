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

logos = ["Logos/nike1.png"]

canny_lower = 5
canny_upper = 20
ratio_thresh = 0.9
video_name = "nikeAd"
video_file_name = ""
frame_gap = 4

def create_dir(name):
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, name)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

try:
    # Fetch YouTube video
    video = YouTube(url)
    stream = video.streams.filter(file_extension="mp4").first()

    # Another option is to download the file, but it seems unnecessary here
    stream.download(filename=f"{video.title}.mp4")
    video_file_name = f"{video.title}.mp4"

except KeyError:
    print("Unable to fetch video information. Please check the video URL or your network connection.")
    exit(0)

# Convert the stream object into something that cv2 recognizes as a video
buffer = io.BytesIO()
stream.stream_to_buffer(buffer)
buffer.seek(0)

video_frames = []

create_dir("yolov7_logo_detection/data/" + video_name)
# Use imageio to read frames from the buffer
for frame_index, frame in enumerate(iio.imiter(buffer, plugin="pyav")):
    # Convert frame to OpenCV-compatible format
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Get every (frame_gap)th frame of the video
    if frame_index % frame_gap == 0:
        video_frames.append(frame_bgr)
        cv2.imwrite(video_name + "/frame" + str(frame_index) + ".png", frame_bgr)

detect(video_name, "runs/detect", "nikeAdTest")

modify_video_with_frames(video_file_name, "runs/detect/nikeAdTest", "nike_done.mp4")