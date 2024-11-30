import cv2
import numpy as np
from pytubefix import YouTube
import io
import imageio.v3 as iio
import SIFT
import RANSAC

url = "https://www.youtube.com/watch?v=freRkr00otY"

logos = ["Logos/nike1.png"]

canny_lower = 5
canny_upper = 20
ratio_thresh = 0.9

try:
    # Fetch YouTube video
    video = YouTube(url)
    stream = video.streams.filter(file_extension="mp4").first()

    # Another option is to download the file, but it seems unnecessary here
    #stream.download(filename=f"{video.title}.mp4")

except KeyError:
    print("Unable to fetch video information. Please check the video URL or your network connection.")
    exit(0)

# Convert the stream object into something that cv2 recognizes as a video
buffer = io.BytesIO()
stream.stream_to_buffer(buffer)
buffer.seek(0)

video_frames = []

# Use imageio to read frames from the buffer
for frame_index, frame in enumerate(iio.imiter(buffer, plugin="pyav")):
    # Convert frame to OpenCV-compatible format
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Get every 30th frame of the video
    if frame_index % 30 == 0:
        video_frames.append(frame_bgr)
        #edges = cv2.Canny(frame_bgr, 60, 150)
        #cv2.imshow("cv2_canny.png", edges)

        #cv2.imshow(f"Frame {frame_index}", frame_bgr)

    # Close window on key press
    #if cv2.waitKey(1) & 0xFF == ord("q"):
        #break

#cv2.destroyAllWindows()
canny_logos = []
for logo in logos:
    logo_temp = cv2.imread(logo, 0)
    edges = cv2.Canny(logo_temp, canny_lower, canny_upper)
    canny_logos.append(edges)

for frame in video_frames:
    canny = cv2.Canny(frame, canny_lower, canny_upper)

    for logo in canny_logos:
        SIFT.SIFT(canny, logo, ratio_thresh)
        #RANSAC.RANSAC(logo, canny)