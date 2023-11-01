import cv2
import time


start_time = time.time()
name_list = []
for index in range(1, 6):
    name_list.append("c00" + str(index))
for index in range(10, 41):
    name_list.append("c0" + str(index))

for name in name_list:
    video_path = "./input_videos/" + name + ".mp4"
    cap = cv2.VideoCapture(video_path)
    while True:
        ret_val, frame_image = cap.read()
        if ret_val == False:
            break
    
end_time = time.time()
print("load time: " + str(end_time - start_time))