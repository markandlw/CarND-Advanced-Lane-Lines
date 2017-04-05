from moviepy.editor import VideoFileClip
from map_lane import map_lane

video_output = "output_images/project_video_output.mp4"
clip1 = VideoFileClip("project_video.mp4")

clip1_output = clip1.fl_image(map_lane)
clip1_output.write_videofile(video_output, audio=False)