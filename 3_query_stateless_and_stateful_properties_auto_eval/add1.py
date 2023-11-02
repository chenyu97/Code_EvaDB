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
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from evadb.functions.gpu_compatible import GPUCompatible
from evadb.utils.generic_utils import try_to_import_ultralytics

import time


class ADD1(AbstractFunction):

    @property
    def name(self) -> str:
        return "add1"

    @setup(cacheable=False, function_type="add1", batchable=False)
    def setup(self):
        pass

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["id", "iid", "bbox"],
                column_types=[
                    NdArrayType.INT32,
                    NdArrayType.INT32,
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[(1,), (None,), (None, 4)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["added_id", "cur_iid", "last_bbox"],
                column_types=[
                    NdArrayType.INT32,
                    NdArrayType.INT32,
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[(1,), (None,), (None, 4)],
            )
        ],
    )
    def forward(self, inputs: pd.DataFrame) -> pd.DataFrame:
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
        for index, row in inputs.iterrows():
            outcome.append(
                {
                    "added_id": row['id'] + 1,
                    "cur_iid": row['iid'],
                    "last_bbox": row['bbox'], 
                },
            )
        
        return pd.DataFrame(
            outcome,
            columns=[
                "added_id",
                "cur_iid", 
                "last_bbox", 
            ],
        )