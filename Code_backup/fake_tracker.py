# coding=utf-8
# Copyright 2018-2023 EvaDB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

import numpy as np
import pandas as pd

from evadb.functions.abstract.abstract_function import AbstractClassifierFunction
from evadb.functions.gpu_compatible import GPUCompatible
from evadb.utils.generic_utils import (
    try_to_import_facenet_pytorch,
    try_to_import_torch,
    try_to_import_torchvision,
)
from evadb.utils.logging_manager import logger


class FakeTracker(AbstractClassifierFunction, GPUCompatible):
    """
    Arguments:
        threshold (float): Threshold for classifier confidence score
    """
    
    def setup(self, threshold=0.85):
        self.threshold = threshold
        #try_to_import_torch()
        #try_to_import_torchvision()
        #try_to_import_facenet_pytorch()
        
        import json
        resource_path = './resources/c001.json'
        self.data = None
        with open(resource_path, 'r') as file:
            self.data = json.load(file)
        
        self.frame_id_max = 0
        for key, value in self.data.items():
            last_frame_id = int(value["frames"][-1][-10:-4])
            if last_frame_id > self.frame_id_max:
                self.frame_id_max = last_frame_id
        
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
            
            self.track_info_list.append(track_info)

        #self.frame_id_cur = 0

        #self.model = MTCNN()

    @property
    def name(self) -> str:
        return "FakeTracker"

    def to_device(self, device: str):
        # pass
        return self

    @property
    def labels(self) -> List[str]:
        return []

    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        """
        Performs predictions on input frames
        Arguments:
            frames (np.ndarray): Frames on which predictions need
            to be performed
        Returns:
            face boxes (List[List[BoundingBox]])
        """   
        outcome = []
        frames_list = frames.transpose().values.tolist()[0]
        frame_id_cur = 0
        for frame in frames_list:
            pred_boxes = []
            pred_tracks = []
            if frame_id_cur >= self.frame_id_max:
                frame_id_cur += 1
            else:
                frame_boxes = []
                frame_tracks = []
                for track_info in self.track_info_list:
                    if track_info["bbx"][frame_id_cur] != None:
                        box = track_info["bbx"][frame_id_cur]
                        tlbr = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
                        frame_boxes.append(np.array(tlbr, dtype=np.float64))    
                        frame_tracks.append(track_info['track_id'])        
                pred_boxes = np.asarray(frame_boxes, dtype="int")
                pred_tracks = frame_tracks
                
                frame_id_cur += 1
            
            outcome.append(
                {"bboxes": pred_boxes, "tracks": pred_tracks},
            )

        return pd.DataFrame(outcome, columns=["bboxes", "tracks"])