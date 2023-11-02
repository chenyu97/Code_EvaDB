import subprocess
import os

# 输入视频文件所在目录和输出目录
input_dir_prefix = "./data/train/"
input_video_paths = []
for index in range(1,6):
    input_video_paths.append(input_dir_prefix + "S01/c00" + str(index) + "/vdo.avi")
for index in range(10,16):
    input_video_paths.append(input_dir_prefix + "S03/c0" + str(index) + "/vdo.avi")
for index in range(16,41):
    input_video_paths.append(input_dir_prefix + "S04/c0" + str(index) + "/vdo.avi")

output_dir = "./input_videos/"

# 遍历每个AVI文件并进行转换
for avi_file in input_video_paths:
    # 构建输入和输出文件的完整路径
    input_file = avi_file
    output_file = os.path.join(output_dir, avi_file[-12:-8] + ".mp4")

    # 使用FFmpeg进行转换
    cmd = f"ffmpeg -i {avi_file} {output_file}"
    subprocess.run(cmd, shell=True)

    print(f"转换完成: {avi_file} -> {os.path.basename(output_file)}")
