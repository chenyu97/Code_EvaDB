import numpy as np
from Codes.vqpy.examples.aicity_query.aicity_recognize import prepare_recognize, recognize
from vqpy.operator.detector.base import DetectorBase
from vqpy.class_names.coco import COCO_CLASSES
from typing import Dict, List
import pickle
from vqpy.operator.detector import register
from pathlib import Path
import json
import cv2
import os

from vqpy.utils.images import crop_image


class FakeDetectorAicity(DetectorBase):
    cls_names = COCO_CLASSES
    output_fields = ["tlbr", "score", "class_id"]
    # Since this is a FAKE detector with which the score is naturally 1,
    # we use the field of score to store the fake track_id to transit to the later FAKE tracker.

    def __init__(self, model_path, **detector_kwargs):
        # load file
        with open(model_path, 'r') as file:
            self.data = json.load(file)
        
        self.frame_id_max = 0
        for key, value in self.data.items():
            last_frame_id = int(value["frames"][-1][-10:-4])
            if last_frame_id > self.frame_id_max:
                self.frame_id_max = last_frame_id
        
        print(self.frame_id_max)
        
        track_count = 0
        self.track_info_list = []
        for key, value in self.data.items():
            track_count += 1
            
            track_info = dict()
            track_info["track_id_original"] = key
            track_info["track_id"] = track_count
            track_info["bbx"] = [None] * self.frame_id_max
            for i in range(len(value["frames"])):
                frame_id = int(value["frames"][i][-10:-4])
                track_info["bbx"][frame_id - 1] = value["boxes"][i]
                #self.save_cropped_image(value["frames"][i], value["boxes"][i], track_count)
            
            self.track_info_list.append(track_info)

        self.frame_id_cur = 0

        self.transform_color, self.model_color = prepare_recognize("color")
        self.transform_type, self.model_type = prepare_recognize("type")
        self.transform_direction, self.model_direction = prepare_recognize("direction")


    def inference(self, img) -> List[Dict]:
        if self.frame_id_cur >= self.frame_id_max:
            self.frame_id_cur += 1
            return []
        
        with open('./result_model_test/single_video_fake_detect_1_np_array', 'a') as file:

            file.write("frame_ID_CUR: " + str(self.frame_id_cur) + ', ')
            outputs = list()
            for track_info in self.track_info_list:
                if track_info["bbx"][self.frame_id_cur] != None:
                    det = dict()
                    box = track_info["bbx"][self.frame_id_cur]
                    tlbr = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
                    det['tlbr'] = np.array(tlbr, dtype=np.float64)
                    det['score'] = track_info['track_id']  # Represents track_id
                    det['class_id'] = 2  # represent car
                    outputs.append(det)

                    cropped_image = crop_image(img, np.array(tlbr, dtype=np.float64))

                    reference = ["blue", "green", "black", "white", "red", "grey", "silver", "brown"]
                    color_result = recognize(self.transform_color, self.model_color, cropped_image, reference)

                    reference = ["pickup-truck", "sedan", "suv", "van", "bus", "hatchback", "truck"]
                    type_result = recognize(self.transform_type, self.model_type, cropped_image, reference)
                    
                    reference = ["straight", "right", "left", "stop"]
                    direction_result = recognize(self.transform_direction, self.model_direction, cropped_image, reference)

                    file.write("track_id: " + str(track_info['track_id']) + ', ' + \
                        str(color_result) + ' ' + str(type_result) + ' ' + str(direction_result) + ', ')
                    
                    file.write('img: ' + str(np.sum(img)%10000) + ', ')
                    
                    #for row in img:
                    #    file.write(' '.join(map(str, row)) + ', ')
                    
                    file.write('tlbr: ' + str(np.sum(np.array(tlbr, dtype=np.float64))%10000) + ', ')
                    #for row in np.array(tlbr, dtype=np.float64):
                    #    file.write(' '.join(map(str, row)) + ', ')              
                    # 
                    #file.write('cropped_img: ' + str(np.prod(cropped_image)%10000) + ', ')  
                    if self.frame_id_cur == 107:
                        with open('array_vqpy.json', 'w') as f:
                            json.dump(cropped_image.tolist(), f)
            
            file.write('\n')

        self.frame_id_cur += 1
        return outputs
    
    
    def save_cropped_image(self, bg_img_pth, bounding_box, track_id):
        image_path = "./data/" + bg_img_pth
        output_dir = './output_cropped_img/'
        os.makedirs(output_dir, exist_ok=True)

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("Fail to read image!")

        [x, y, w, h] = bounding_box
        cropped_image = image[y:y+h, x:x+w]
        output_path = os.path.join(output_dir, str(track_id) + "_" + bg_img_pth[-10:])
        print("Writing " + str(track_id) + "_" + bg_img_pth[-10:])
        cv2.imwrite(output_path, cropped_image)
    

if __name__ == "__main__":
    resource_dir = Path(Path(__file__).parent, "resources/")
    precomputed_path = (resource_dir / f"aicity_detect_result.json").as_posix()
    print(resource_dir)
    print(precomputed_path)
    register("fake_detector_aicity", FakeDetectorAicity, precomputed_path, None)