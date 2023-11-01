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

import cv2
import time


class RecognizeColor(AbstractFunction):

    @property
    def name(self) -> str:
        return "recognize_color"

    @setup(cacheable=False, function_type="recognize_color", batchable=False)
    def setup(self):
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
        import webcolors
        from colordetect import ColorDetect

        class BindedColorDetect(ColorDetect):
            def _find_unique_colors(self, cluster, centroids) -> dict:
                labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
                (hist, _) = np.histogram(cluster.labels_, bins=labels)
                hist = hist.astype("float")
                hist /= hist.sum()

                # iterate through each cluster's color and percentage
                colors = sorted(
                    [(percent * 100, color)
                    for (percent, color) in zip(hist, centroids)],
                    key=lambda x: (x[0], x[1].tolist())
                )

                for (percent, color) in colors:
                    color.astype("uint8").tolist()
                return dict(colors)

        if image is None:
            return None

        maximum_width = 24
        ratio = max(maximum_width / image.shape[0], maximum_width / image.shape[1])
        if ratio < 1:
            size = (int(image.shape[0] * ratio + 0.5),
                    int(image.shape[1] * ratio + 0.5))
            image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

        detector = BindedColorDetect(image)
        result = detector.get_color_count()
        bestrgb, bestp = None, 0
        for color, percent in result.items():
            rgb = webcolors.name_to_rgb(color)
            if max(*rgb) - min(*rgb) <= 15:
                percent /= 5
            if percent > bestp:
                bestrgb, bestp = rgb, percent
        color_get = webcolors.rgb_to_name(bestrgb)
        # add some common color options:
        return_color = 'other_colors'
        position_max = 0
        color_options = ["blue", "green", "black", "white", "red", "grey", "silver", "brown", "gray"]
        for color_option in color_options:
            if color_option in color_get:
                # the color showing at the last of the color_get is the return_color
                position = color_get.find(color_option) + len(color_option)
                if position > position_max:
                    position_max = position
                    return_color = color_option
        return return_color
    