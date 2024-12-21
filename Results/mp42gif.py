from moviepy.editor import VideoFileClip
from moviepy.video.fx import all as vfx

# Load the video file
video = VideoFileClip("output_video.mp4")

# Speed up the video by a factor of 2
video = video.fx(vfx.speedx, 2)

# Select a portion of the video (optional), for example, the first 10 seconds
#video = video.subclip(0, 10)

# Convert the video to a GIF
video.write_gif("output.gif")
