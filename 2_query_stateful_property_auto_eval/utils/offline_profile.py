import cv2
import json
from pathlib import Path
import time
from aicity_recognize import prepare_recognize
from aicity_recognize import recognize
from collections import Counter
from vqpy.utils.images import crop_image
import numpy as np


if __name__ == "__main__":

    transform_color, model_color = prepare_recognize("color")
    transform_type, model_type = prepare_recognize("type")
    transform_direction, model_direction = prepare_recognize("direction")

    resource_dir = Path(Path(__file__).parent, "resources")
    file_path = (resource_dir / "test-tracks.json").as_posix()

    save_result = dict()
    tracks_list = []
    with open(file_path, 'r') as file:
        data = json.load(file)
        tracks_list = list(data.keys())
        print_count = 0
        for key, value in data.items():
            print_count += 1
            print("detecting track_id: " + str(key) + "(" + str(print_count) + "/" + str(len(tracks_list)) + ")")  
            for i in range(len(value["frames"])):
                bg_img_pth = value["frames"][i]
                bounding_box = value["boxes"][i]
                image_path = "./data/" + bg_img_pth
                image = cv2.imread(image_path)

                if image is None:
                    raise ValueError("Fail to read image!")

                key_track_frame = key[:8] + '_F' + value["frames"][i][-10:-4]
                save_result[key_track_frame] = dict()

                [x, y, w, h] = bounding_box
                tlbr = [x, y, x+w, y+h]
                cropped_image = crop_image(image, np.array(tlbr, dtype=np.float64))

                reference = ["blue", "green", "black", "white", "red", "grey", "silver", "brown"]
                start_time = time.time()
                color = recognize(transform_color, model_color, cropped_image, reference)
                end_time = time.time()
                save_result[key_track_frame]['color_result'] = color
                save_result[key_track_frame]['color_cost'] = end_time - start_time

                reference = ["pickup-truck", "sedan", "suv", "van", "bus", "hatchback", "truck"]
                start_time = time.time()
                type = recognize(transform_type, model_type, cropped_image, reference)
                end_time = time.time()
                save_result[key_track_frame]['type_result'] = type
                save_result[key_track_frame]['type_cost'] = end_time - start_time
        
                reference = [ "straight", "right", "left", "stop"]
                start_time = time.time()
                direction = recognize(transform_direction, model_direction, cropped_image, reference)
                end_time = time.time()
                save_result[key_track_frame]['direction_result'] = direction
                save_result[key_track_frame]['direction_cost'] = end_time - start_time


    json.dump(save_result, open('./resources/offline_profile_3.json', 'w'), indent=4)