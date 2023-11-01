import cv2
import json


def video_bbx_visualize(input_video_path, input_video_bbxs_path, output_video_path):
    input_file = input_video_path
    output_file = output_video_path
    input_bbxs_file = input_video_bbxs_path

    # read boxes
    with open(input_bbxs_file, 'r') as file:
        data = json.load(file)

    boxes = {}
    for key, value in data.items():
        for i in range(len(value["frames"])):
            img_id = int(value["frames"][i][-10:-4]) - 1
            if img_id not in boxes:
                boxes[img_id] = list()
            bbx_content = dict()
            bbx_content["x"] = value["boxes"][i][0]
            bbx_content["y"] = value["boxes"][i][1]
            bbx_content["w"] = value["boxes"][i][2]
            bbx_content["h"] = value["boxes"][i][3]
            bbx_content["id"] = key
            boxes[img_id].append(bbx_content)

    # read video
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # get attributions of videos
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break 
        
        print("processing: " + str(frame_id + 1) + "/" + str(frame_n))
        
        if frame_id in boxes:
            for box in boxes[frame_id]:
                x, y, w, h, box_id = box['x'], box['y'], box['w'], box['h'], box['id'][:8]
                # draw rectangles
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # draw ID and its bg
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.9
                font_color = (255, 255, 255) 
                font_thickness = 2
                bg_color = (0, 0, 0) 
                text_size = cv2.getTextSize(str(box_id), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                cv2.rectangle(frame, (x, y - 10 - text_size[1]), (x + text_size[0], y - 10 + font_thickness), bg_color, -1)
                cv2.putText(frame, str(box_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255) 
        font_thickness = 2
        bg_color = (0, 0, 0) 
        text = f"Frame: {frame_id + 1}"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        x, y = width - text_size[0] - 10, 20 + text_size[1]
        cv2.rectangle(frame, (x, y - text_size[1] - 5), (x + text_size[0], y + 5), bg_color, -1)
        cv2.putText(frame, text, (x, y), font, font_scale, font_color, font_thickness)

        out.write(frame)
        frame_id += 1

    # release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    name_list = []
    for index in range(1, 6):
        name_list.append("c00" + str(index))
    for index in range(10, 41):
        name_list.append("c0" + str(index))
    
    for name in name_list:
        print("processing video: " + str(name))
        input_video_path = '../input_videos/' + str(name) + '.mp4'
        output_video_path = './videos_bbx/' + str(name) + '.mp4'
        input_bbxs_path = '../resources/' + str(name) + '.json'

        video_bbx_visualize(input_video_path, input_bbxs_path, output_video_path)

