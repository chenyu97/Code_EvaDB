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
import math
import numpy as np
import evadb
import warnings
import pandas as pd


#from evadb.functions.abstract.tracker_abstract_function import (
    #EvaDBTrackerAbstractFunction,
#)
from property_calculate_abstract import PropertyCalculateAbstract
from evadb.utils.generic_utils import try_to_import_norfair
from evadb.utils.math_utils import get_centroid

DISTANCE_THRESHOLD_CENTROID: int = 30


class PropertyCalculate(PropertyCalculateAbstract):
    @property
    def name(self) -> str:
        return "PropertyCalculate"

    def setup(self, distance_threshold=DISTANCE_THRESHOLD_CENTROID) -> None:
        # https://github.com/tryolabs/norfair/blob/74b11edde83941dd6e32bcccd5fa849e16bf8564/norfair/tracker.py#L18
        try_to_import_norfair()
        from norfair import Tracker

        self.tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=distance_threshold,
        )
        self.prev_frame_id = None
        self.stored_dataframe = pd.DataFrame(columns=['frame_id', 'track_id', 'bbox'])
        self.stored_dataframe_size = 0

    def forward(self, frame_id, track_ids, labels, bboxes, scores):
        #print("Hello: " + str(frame_id))
        cal_output_validation = []
        cal_output_property = []
        # process multiple tracks on one frame
        for (track_id, bbox, score, label) in zip(track_ids, bboxes, scores, labels):
            if score == -1:
                continue
            # save stored_dateframe
            self.stored_dataframe.loc[self.stored_dataframe_size] = [frame_id, track_id, bbox]
            self.stored_dataframe_size += 1
            # calculate velocity
            cur_tlbr = self.bbx2tlbr(bbox)
            last_tlbr = None
            last_bbx = self.get_last_bbx(frame_id, track_id)
            if len(last_bbx) != 0:
                last_tlbr = self.bbx2tlbr(last_bbx)
            velocity = self.velocity(cur_tlbr, last_tlbr)
            cal_output_property.append(velocity)
            # judge speeding
            if velocity > 1.0:
                cal_output_validation.append(1)
            else:
                cal_output_validation.append(0)
            
        return np.array(cal_output_validation), np.array(cal_output_property)

    def get_last_bbx(self, cur_frame_id, cur_track_id):
        # read stored_dataframe
        last_frame_id = cur_frame_id - 1
        for index in range(self.stored_dataframe_size):
            if (self.stored_dataframe.loc[index, 'frame_id'] == last_frame_id) and \
                (self.stored_dataframe.loc[index, 'track_id'] == cur_track_id):
                return self.stored_dataframe.loc[index, 'bbox']
        return []

    def bbx2tlbr(self, bounding_box):
        [x, y, w, h] = bounding_box
        tlbr = [x, y, x+w, y+h]
        return np.array(tlbr)

    def velocity(self, tlbr, last_tlbr):
        fps = 10
        if last_tlbr is None or tlbr is None:
            return 0
        last_center = (last_tlbr[:2] + last_tlbr[2:]) / 2
        cur_center = (tlbr[:2] + tlbr[2:]) / 2
        tlbr_avg = (tlbr + last_tlbr) / 2
        scale = (tlbr_avg[3] - tlbr_avg[1]) / 1.5
        dcenter = (cur_center - last_center) / scale * fps
        return math.sqrt(sum(dcenter * dcenter))