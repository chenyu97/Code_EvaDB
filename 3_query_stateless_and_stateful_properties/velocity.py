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
import pandas as pd

from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from evadb.functions.gpu_compatible import GPUCompatible
from evadb.utils.generic_utils import try_to_import_ultralytics

import time


class Velocity(AbstractFunction):

    @property
    def name(self) -> str:
        return "velocity"

    @setup(cacheable=False, function_type="velocity", batchable=False)
    def setup(self):
        pass

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["bbox", "last_bbox"],
                column_types=[
                    NdArrayType.FLOAT32,
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[(1, 4), (1, 4)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["velocity"],
                column_types=[
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[(1,)],
            )
        ],
    )
    def forward(self, two_bboxes: pd.DataFrame) -> pd.DataFrame:
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
        for index, row in two_bboxes.iterrows():
            outcome.append(
                {
                    "velocity": self.velocity(row['bbox'], row['last_bbox']),
                },
            )
        
        return pd.DataFrame(
            outcome,
            columns=[
                "velocity",
            ],
        )

    def velocity(self, tlbr, last_tlbr):
        fps = 30
        if last_tlbr is None or tlbr is None:
            return 0
        last_center = (last_tlbr[:2] + last_tlbr[2:]) / 2
        cur_center = (tlbr[:2] + tlbr[2:]) / 2
        tlbr_avg = (tlbr + last_tlbr) / 2
        scale = (tlbr_avg[3] - tlbr_avg[1]) / 1.5
        dcenter = (cur_center - last_center) / scale * fps
        return math.sqrt(sum(dcenter * dcenter))