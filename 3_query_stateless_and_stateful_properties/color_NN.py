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
from aicity_recognize import prepare_recognize
from aicity_recognize import recognize

import cv2
import time


class RecognizeColorNN(AbstractFunction):

    @property
    def name(self) -> str:
        return "recognize_color_NN"

    @setup(cacheable=False, function_type="recognize_color_NN", batchable=False)
    def setup(self):
        self.transform_color, self.model_color = prepare_recognize("color")
        self.count = 0
        pass

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["cropped_frame_array"],
                column_types=[
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[(None, None, 3)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["color"],
                column_types=[
                    NdArrayType.STR,
                ],
                column_shapes=[(1,)],
            )
        ],
    )
    def forward(self, input: pd.DataFrame) -> pd.DataFrame:
        self.count += 1
        # print("color_model: " + str(self.count))
        outcome = []
        for index, row in input.iterrows():
            outcome.append(
                {
                    "color": self.getColor(row['cropped_frame_array']),
                },
            )
        return pd.DataFrame(
            outcome,
            columns=[
                "color",
            ],
        )
    
    def getColor(self, image):
        reference = ["blue", "green", "black", "white", "red", "grey", "silver", "brown"]
        color = recognize(self.transform_color, self.model_color, image, reference)
        #print(color)
        return color
    