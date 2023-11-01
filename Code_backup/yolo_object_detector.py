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
import numpy as np
import pandas as pd

from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe, PyTorchTensor
from evadb.functions.gpu_compatible import GPUCompatible
from evadb.utils.generic_utils import try_to_import_ultralytics
import torch
from torch import Tensor
from ultralytics import YOLO
import json


class Yolo(AbstractFunction, GPUCompatible):
    """
    Arguments:
        threshold (float): Threshold for classifier confidence score
    """

    @property
    def name(self) -> str:
        return "yolo"

    '''
    @setup(cacheable=True, function_type="object_detection", batchable=True)
    def setup(self, model: str, threshold=0.3):
        try_to_import_ultralytics()
        from ultralytics import YOLO

        self.threshold = threshold
        self.model = YOLO(model)
        self.device = "cpu"
    '''

    @setup(cacheable=False, function_type="object_detection", batchable=False)
    def setup(self, threshold=0.85):
        self.times = 0
        self.batchsize_observed = 5
        #custom setup function that is specific for the function
        #self.threshold = threshold
        #self.model = YOLO("yolov8m")
        #self.model = torch.hub.load("ultralytics/yolov5", "yolov5s", verbose=False)
        
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
        
    '''
    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["data"],
                column_types=[NdArrayType.FLOAT32],
                column_shapes=[(None, None, 3)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["labels", "bboxes", "scores"],
                column_types=[
                    NdArrayType.STR,
                    NdArrayType.FLOAT32,
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[(None,), (None,), (None,)],
            )
        ],
    )
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        """
        Performs predictions on input frames
        Arguments:
            frames (np.ndarray): Frames on which predictions need
            to be performed
        Returns:
            tuple containing predicted_classes (List[List[str]]),
            predicted_boxes (List[List[BoundingBox]]),
            predicted_scores (List[List[float]])
        """
        outcome = []
        # Fix me: this should be taken care by decorators
        frames = np.ravel(frames.to_numpy())
        list_of_numpy_images = [its for its in frames]
        predictions = self.model.predict(
            list_of_numpy_images, device=self.device, conf=self.threshold, verbose=False
        )
        for pred in predictions:
            single_result = pred.boxes
            pred_class = [self.model.names[i] for i in single_result.cls.tolist()]
            pred_score = single_result.conf.tolist()
            pred_score = [round(conf, 2) for conf in single_result.conf.tolist()]
            pred_boxes = single_result.xyxy.tolist()
            sorted_list = list(map(lambda i: i < self.threshold, pred_score))
            t = sorted_list.index(True) if True in sorted_list else len(sorted_list)
            outcome.append(
                {
                    "labels": pred_class[:t],
                    "bboxes": pred_boxes[:t],
                    "scores": pred_score[:t],
                },
            )
        return pd.DataFrame(
            outcome,
            columns=[
                "labels",
                "bboxes",
                "scores",
            ],
        )
    '''
    @forward(
        input_signatures=[
            PyTorchTensor(
                name="input_col",
                is_nullable=False,
                type=NdArrayType.FLOAT32,
                dimensions=(1, 3, 540, 960)
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["labels", "bboxes", "scores"],
                column_types=[
                    NdArrayType.STR,
                    NdArrayType.FLOAT32,
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[(None,), (None,), (None,)],
            )
        ],
    )   
    def forward(self, frames: Tensor) -> pd.DataFrame:
        #print(self.times)
        #print("frame numbers: " + str(len(frames)))
        #the custom logic for the function
        outcome = []
        for index in range(self.batchsize_observed):
            frame_id_cur = self.times * self.batchsize_observed + index
            pred_classes = []
            pred_scores = []
            pred_boxes = []
            if frame_id_cur >= self.frame_id_max:
                frame_boxes = []
                frame_tracks = []
                frame_classes = []
                for track_info in self.track_info_list:
                    frame_boxes.append([-1, -1, -1, -1])
                    frame_tracks.append(-1)
                    frame_classes.append(-1)
                pred_boxes = np.array(frame_boxes)
                pred_scores = np.array(frame_tracks)
                pred_classes = np.array(frame_classes)
            else:
                frame_boxes = []
                frame_tracks = []
                frame_classes = []
                for track_info in self.track_info_list:
                    if track_info["bbx"][frame_id_cur] != None:
                        box = track_info["bbx"][frame_id_cur]
                        tlbr = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
                        frame_boxes.append(tlbr)
                        frame_tracks.append(track_info['track_id'])    
                        frame_classes.append(2) # means car 
                    else:
                        frame_boxes.append([-1, -1, -1, -1])
                        frame_tracks.append(-1)
                        frame_classes.append(-1)
                pred_boxes = np.array(frame_boxes)
                pred_scores = np.array(frame_tracks)
                pred_classes = np.array(frame_classes)
                
                frame_id_cur += 1
            outcome.append(
                {"labels": pred_classes, "bboxes": pred_boxes, "scores": pred_scores}
            )            
        self.times += 1
        '''
        frames = torch.permute(frames, (0, 2, 3, 1))
        predictions = self.model([its.cpu().detach().numpy() * 255 for its in frames])

        for i in range(frames.shape[0]):
            single_result = predictions.pandas().xyxy[i]
            pred_class = single_result["name"].tolist()
            pred_score = single_result["confidence"].tolist()
            pred_boxes = single_result[["xmin", "ymin", "xmax", "ymax"]].apply(
                lambda x: list(x), axis=1
            )

            outcome.append(
                {"labels": pred_class, "bboxes": pred_boxes, "scores": pred_score}
            )
        '''
        
        return pd.DataFrame(outcome, columns=["labels", "bboxes", "scores"])

    def to_device(self, device: str):
        self.device = device
        return self
