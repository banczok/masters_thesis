from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import math
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import segmentation_models_pytorch as smp
import cv2
import albumentations as A
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
import gc
import torch.nn.functional as F
import torchseg
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
from albumentations.augmentations.geometric import functional as FA
from albumentations.core.transforms_interface import  DualTransform
from albumentations.core.utils import to_tuple

resume = False

kidney_shapes = {
    "kidney_1_dense": (2279, 1303, 912), 
    "kidney_1_voi": (1397, 1928, 1928), 
    "kidney_2": (2217, 1041, 1511), 
    "kidney_3": (1035, 1706, 1510), 
    "kidney_3_dense": (501, 1706, 1510),
    "kidney_3_sparse": (1035, 1706, 1510),
}

########################## HELPER FUNCTIONS #################################
def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(0, 1))
    return dice


def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(0, 1))
    return iou

# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Lookup tables used by surface distance metrics."""



ENCODE_NEIGHBOURHOOD_3D_KERNEL = np.array([[[128, 64], [32, 16]], [[8, 4],
                                                                   [2, 1]]])

# _NEIGHBOUR_CODE_TO_NORMALS is a lookup table.
# For every binary neighbour code
# (2x2x2 neighbourhood = 8 neighbours = 8 bits = 256 codes)
# it contains the surface normals of the triangles (called "surfel" for
# "surface element" in the following). The length of the normal
# vector encodes the surfel area.
#
# created using the marching_cube algorithm
# see e.g. https://en.wikipedia.org/wiki/Marching_cubes
# pylint: disable=line-too-long
_NEIGHBOUR_CODE_TO_NORMALS = [
    [[0, 0, 0]],
    [[0.125, 0.125, 0.125]],
    [[-0.125, -0.125, 0.125]],
    [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
    [[0.125, -0.125, 0.125]],
    [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25]],
    [[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[-0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25]],
    [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
    [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25], [-0.125, 0.125, -0.125]],
    [[-0.5, 0.0, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
    [[0.125, -0.125, -0.125]],
    [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25]],
    [[-0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, -0.5, 0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, 0.0, -0.5], [0.25, 0.25, 0.25], [-0.125, -0.125, -0.125]],
    [[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], [0.125, -0.125, -0.125]],
    [[0.125, 0.125, 0.125], [0.375, 0.375, 0.375], [0.0, -0.25, 0.25], [-0.25, 0.0, 0.25]],
    [[0.125, -0.125, -0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
    [[0.375, 0.375, 0.375], [0.0, 0.25, -0.25], [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
    [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25], [0.125, 0.125, 0.125]],
    [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25]],
    [[0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
    [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125], [0.25, 0.25, -0.25]],
    [[0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.125, -0.125, 0.125], [-0.25, -0.0, -0.25], [0.25, 0.0, 0.25]],
    [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25], [0.125, -0.125, 0.125]],
    [[-0.375, -0.375, 0.375], [-0.0, 0.25, 0.25], [0.125, 0.125, -0.125], [-0.25, -0.0, -0.25]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.125, 0.125, -0.125], [-0.125, -0.125, 0.125]],
    [[0.125, -0.125, 0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
    [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25], [-0.125, 0.125, -0.125], [0.125, -0.125, 0.125]],
    [[0.0, 0.25, -0.25], [0.375, -0.375, -0.375], [-0.125, 0.125, 0.125], [0.25, 0.25, 0.0]],
    [[-0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]],
    [[0.0, 0.5, 0.0], [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
    [[0.0, 0.5, 0.0], [0.125, -0.125, 0.125], [-0.25, 0.25, -0.25]],
    [[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]],
    [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0], [0.125, -0.125, 0.125]],
    [[-0.375, -0.375, -0.375], [-0.25, 0.0, 0.25], [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
    [[0.125, 0.125, 0.125], [0.0, -0.5, 0.0], [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125]],
    [[0.0, -0.5, 0.0], [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125]],
    [[-0.125, 0.125, 0.125], [0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]],
    [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.375, 0.375, -0.375], [-0.25, -0.25, 0.0], [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
    [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
    [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
    [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], [-0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]],
    [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]],
    [[-0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
    [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25]],
    [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125]],
    [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.375, -0.375, 0.375], [0.0, -0.25, -0.25], [-0.125, 0.125, -0.125], [0.25, 0.25, 0.0]],
    [[-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25]],
    [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125]],
    [[-0.25, 0.25, -0.25], [-0.25, 0.25, -0.25], [-0.125, 0.125, -0.125], [-0.125, 0.125, -0.125]],
    [[-0.25, 0.0, -0.25], [0.375, -0.375, -0.375], [0.0, 0.25, -0.25], [-0.125, 0.125, 0.125]],
    [[0.5, 0.0, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125]],
    [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
    [[-0.0, 0.0, 0.5], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
    [[-0.25, -0.0, -0.25], [-0.375, 0.375, 0.375], [-0.25, -0.25, 0.0], [-0.125, 0.125, 0.125]],
    [[0.0, 0.0, -0.5], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
    [[-0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
    [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
    [[0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
    [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25], [-0.125, 0.125, 0.125]],
    [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
    [[0.125, -0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
    [[0.25, 0.0, 0.25], [-0.375, -0.375, 0.375], [-0.25, 0.25, 0.0], [-0.125, -0.125, 0.125]],
    [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
    [[0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
    [[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
    [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
    [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [0.125, -0.125, 0.125]],
    [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
    [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25], [0.125, -0.125, -0.125]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
    [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.5, 0.0, -0.0], [0.25, -0.25, -0.25], [0.125, -0.125, -0.125]],
    [[-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125], [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
    [[0.375, -0.375, 0.375], [0.0, 0.25, 0.25], [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
    [[0.0, -0.5, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[-0.375, -0.375, 0.375], [0.25, -0.25, 0.0], [0.0, 0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [-0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
    [[0.125, 0.125, 0.125], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
    [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
    [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
    [[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
    [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25], [0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
    [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0], [0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0], [0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
    [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25], [0.125, 0.125, 0.125]],
    [[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
    [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
    [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
    [[0.125, 0.125, 0.125], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
    [[-0.125, 0.125, 0.125], [-0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
    [[-0.375, -0.375, 0.375], [0.25, -0.25, 0.0], [0.0, 0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.0, -0.5, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[0.375, -0.375, 0.375], [0.0, 0.25, 0.25], [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
    [[-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125], [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
    [[0.5, 0.0, -0.0], [0.25, -0.25, -0.25], [0.125, -0.125, -0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25], [0.125, -0.125, -0.125]],
    [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
    [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [0.125, -0.125, 0.125]],
    [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
    [[0.125, 0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
    [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125]],
    [[0.25, 0.0, 0.25], [-0.375, -0.375, 0.375], [-0.25, 0.25, 0.0], [-0.125, -0.125, 0.125]],
    [[0.125, -0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
    [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
    [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25], [-0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
    [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
    [[-0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
    [[0.0, 0.0, -0.5], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
    [[-0.25, -0.0, -0.25], [-0.375, 0.375, 0.375], [-0.25, -0.25, 0.0], [-0.125, 0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
    [[-0.0, 0.0, 0.5], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
    [[0.5, 0.0, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125]],
    [[-0.25, 0.0, -0.25], [0.375, -0.375, -0.375], [0.0, 0.25, -0.25], [-0.125, 0.125, 0.125]],
    [[-0.25, 0.25, -0.25], [-0.25, 0.25, -0.25], [-0.125, 0.125, -0.125], [-0.125, 0.125, -0.125]],
    [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125]],
    [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[0.375, -0.375, 0.375], [0.0, -0.25, -0.25], [-0.125, 0.125, -0.125], [0.25, 0.25, 0.0]],
    [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125]],
    [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25]],
    [[-0.125, -0.125, 0.125], [-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
    [[-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.125, -0.125, 0.125]],
    [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]],
    [[0.125, 0.125, 0.125], [-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]],
    [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], [-0.125, -0.125, 0.125]],
    [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
    [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
    [[-0.375, 0.375, -0.375], [-0.25, -0.25, 0.0], [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
    [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]],
    [[0.0, -0.5, 0.0], [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125]],
    [[0.125, 0.125, 0.125], [0.0, -0.5, 0.0], [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125]],
    [[-0.375, -0.375, -0.375], [-0.25, 0.0, 0.25], [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
    [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0], [0.125, -0.125, 0.125]],
    [[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]],
    [[0.0, 0.5, 0.0], [0.125, -0.125, 0.125], [-0.25, 0.25, -0.25]],
    [[0.0, 0.5, 0.0], [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
    [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]],
    [[-0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.0, 0.25, -0.25], [0.375, -0.375, -0.375], [-0.125, 0.125, 0.125], [0.25, 0.25, 0.0]],
    [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25], [-0.125, 0.125, -0.125], [0.125, -0.125, 0.125]],
    [[0.125, -0.125, 0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.125, 0.125, -0.125], [-0.125, -0.125, 0.125]],
    [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
    [[-0.375, -0.375, 0.375], [-0.0, 0.25, 0.25], [0.125, 0.125, -0.125], [-0.25, -0.0, -0.25]],
    [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25], [0.125, -0.125, 0.125]],
    [[0.125, -0.125, 0.125], [-0.25, -0.0, -0.25], [0.25, 0.0, 0.25]],
    [[0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125], [0.25, 0.25, -0.25]],
    [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.125, -0.125, 0.125]],
    [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25]],
    [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25], [0.125, 0.125, 0.125]],
    [[0.375, 0.375, 0.375], [0.0, 0.25, -0.25], [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
    [[0.125, -0.125, -0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
    [[0.125, 0.125, 0.125], [0.375, 0.375, 0.375], [0.0, -0.25, 0.25], [-0.25, 0.0, 0.25]],
    [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], [0.125, -0.125, -0.125]],
    [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, -0.125]],
    [[-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, 0.0, -0.5], [0.25, 0.25, 0.25], [-0.125, -0.125, -0.125]],
    [[0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, -0.5, 0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25]],
    [[0.125, -0.125, -0.125]],
    [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
    [[-0.5, 0.0, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25], [-0.125, 0.125, -0.125]],
    [[0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
    [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25]],
    [[0.125, 0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[-0.125, 0.125, 0.125]],
    [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25]],
    [[0.125, -0.125, 0.125]],
    [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
    [[-0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125]],
    [[0, 0, 0]]]
# pylint: enable=line-too-long


def create_table_neighbour_code_to_surface_area(spacing_mm):
  """Returns an array mapping neighbourhood code to the surface elements area.

  Note that the normals encode the initial surface area. This function computes
  the area corresponding to the given `spacing_mm`.

  Args:
    spacing_mm: 3-element list-like structure. Voxel spacing in x0, x1 and x2
      direction.
  """
  # compute the area for all 256 possible surface elements
  # (given a 2x2x2 neighbourhood) according to the spacing_mm
  neighbour_code_to_surface_area = np.zeros([256])
  for code in range(256):
    normals = np.array(_NEIGHBOUR_CODE_TO_NORMALS[code])
    sum_area = 0
    for normal_idx in range(normals.shape[0]):
      # normal vector
      n = np.zeros([3])
      n[0] = normals[normal_idx, 0] * spacing_mm[1] * spacing_mm[2]
      n[1] = normals[normal_idx, 1] * spacing_mm[0] * spacing_mm[2]
      n[2] = normals[normal_idx, 2] * spacing_mm[0] * spacing_mm[1]
      area = np.linalg.norm(n)
      sum_area += area
    neighbour_code_to_surface_area[code] = sum_area

  return neighbour_code_to_surface_area


# In the neighbourhood, points are ordered: top left, top right, bottom left,
# bottom right.
ENCODE_NEIGHBOURHOOD_2D_KERNEL = np.array([[8, 4], [2, 1]])


def create_table_neighbour_code_to_contour_length(spacing_mm):
  """Returns an array mapping neighbourhood code to the contour length.

  For the list of possible cases and their figures, see page 38 from:
  https://nccastaff.bournemouth.ac.uk/jmacey/MastersProjects/MSc14/06/thesis.pdf

  In 2D, each point has 4 neighbors. Thus, are 16 configurations. A
  configuration is encoded with '1' meaning "inside the object" and '0' "outside
  the object". The points are ordered: top left, top right, bottom left, bottom
  right.

  The x0 axis is assumed vertical downward, and the x1 axis is horizontal to the
  right:
   (0, 0) --> (0, 1)
     |
   (1, 0)

  Args:
    spacing_mm: 2-element list-like structure. Voxel spacing in x0 and x1
      directions.
  """
  neighbour_code_to_contour_length = np.zeros([16])

  vertical = spacing_mm[0]
  horizontal = spacing_mm[1]
  diag = 0.5 * math.sqrt(spacing_mm[0]**2 + spacing_mm[1]**2)
  # pyformat: disable
  neighbour_code_to_contour_length[int("00"
                                       "01", 2)] = diag

  neighbour_code_to_contour_length[int("00"
                                       "10", 2)] = diag

  neighbour_code_to_contour_length[int("00"
                                       "11", 2)] = horizontal

  neighbour_code_to_contour_length[int("01"
                                       "00", 2)] = diag

  neighbour_code_to_contour_length[int("01"
                                       "01", 2)] = vertical

  neighbour_code_to_contour_length[int("01"
                                       "10", 2)] = 2*diag

  neighbour_code_to_contour_length[int("01"
                                       "11", 2)] = diag

  neighbour_code_to_contour_length[int("10"
                                       "00", 2)] = diag

  neighbour_code_to_contour_length[int("10"
                                       "01", 2)] = 2*diag

  neighbour_code_to_contour_length[int("10"
                                       "10", 2)] = vertical

  neighbour_code_to_contour_length[int("10"
                                       "11", 2)] = diag

  neighbour_code_to_contour_length[int("11"
                                       "00", 2)] = horizontal

  neighbour_code_to_contour_length[int("11"
                                       "01", 2)] = diag

  neighbour_code_to_contour_length[int("11"
                                       "10", 2)] = diag
  # pyformat: enable

  return neighbour_code_to_contour_length

#################################### CLASSES ################################


class VolumeDataset(Dataset):
    def __init__(
        self, 
        data_dir, 
        kidney_orientations,  # Dictionary: { "kidney_1_dense": "normal", "kidney_2": "xz", ... }
        target_size=1024,
        transform=None,  # Albumentations for transformations
        max_workers=16   # Number of parallel workers for image processing
    ):
        self.data_dir = data_dir
        self.kidney_orientations = kidney_orientations
        self.target_size = target_size
        self.transform = transform
        self.all_patches = []  # Store 2.5D stacks of resized images and their masks here

        # Process and load all 2.5D stacks for each kidney in the specified orientations
        self.process_all_kidneys(max_workers)
    
    def __len__(self):
        return len(self.all_patches)  # Each entry is a 2.5D stack with its center slice mask

    def __getitem__(self, idx):
        # Retrieve a preprocessed 2.5D stack and center mask from all_patches
        img_stack, mask = self.all_patches[idx]

        #img_stack = self.histogram_equalization(img_stack.astype('float32') / 65535.)
        img_stack = img_stack.astype(np.float32) / 65535.0
        mask = (mask > 0).astype(np.float32) 

        # Apply transformations if specified
        if self.transform:
            img_stack, mask = self.augment_image(img_stack, mask)
        
        # Convert to torch tensors
        img_stack = torch.tensor(img_stack, dtype=torch.float32).contiguous()
        mask = torch.tensor(mask, dtype=torch.float32).contiguous()
        
        return img_stack, mask

    def process_all_kidneys(self, max_workers=32):
        """Processes all kidneys and their orientations, creates 2.5D stacks, and stores them in all_patches."""
        
        def process_kidney(kidney_name):
            orientation = 'normal'
            if '_xz' in kidney_name:
                orientation = 'xz'
            elif '_yz' in kidney_name:
                orientation = 'yz'
            kidney = kidney_name.replace('_xz', '').replace('_yz', '')
    
            print(f"Processing {kidney} with orientation: {orientation}")
    
            # Load the volume and mask for each kidney and specified orientation
            volume_path = os.path.join(self.data_dir, f"{kidney}.mmap")
            mask_path = os.path.join(self.data_dir, f"{kidney}_mask.mmap")
            volume, mask = self.load_volume_and_mask(volume_path, mask_path, orientation, kidney)
    
            # Create 2.5D stacks for the current kidney
            self.create_stacks(volume, mask)
    
            # Clear memory for the current kidney
            del volume, mask
            torch.cuda.empty_cache()

        # Execute each kidney processing in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_kidney, kidney): kidney for kidney in self.kidney_orientations}
            
            for future in as_completed(futures):
                kidney = futures[future]
                try:
                    future.result()
                    print(f"{kidney} processed successfully.")
                except Exception as exc:
                    print(f"{kidney} generated an exception: {exc}")

    def load_volume_and_mask(self, volume_path, mask_path, orientation, kidney):
        """Loads a kidney's volume and mask and applies the specified orientation."""
        shape = kidney_shapes[kidney] 
        volume = np.memmap(volume_path, dtype=np.uint16, mode="r").reshape(shape)
        mask = np.memmap(mask_path, dtype=np.uint8, mode="r").reshape(shape)
        
        # Apply orientation adjustment
        if orientation == "xz":
            volume = volume.transpose((1, 2, 0))
            mask = mask.transpose((1, 2, 0))
        elif orientation == "yz":
            volume = volume.transpose((2, 0, 1))
            mask = mask.transpose((2, 0, 1))
        
        return volume, mask
    
    def histogram_equalization(self, image, number_bins=1024):
        """Applies histogram equalization to normalize the image."""
        image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
        cdf = image_histogram.cumsum()
        cdf = (number_bins - 1) * cdf / cdf[-1]
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
        return image_equalized.reshape(image.shape)

    def create_stacks(self, volume, mask):
        """Creates 2.5D stacks of consecutive slices and corresponding masks, then appends to all_patches."""
        for slice_idx in range(1, volume.shape[0] - 1):
            # Select three consecutive slices centered at slice_idx
            img_stack = volume[slice_idx - 1:slice_idx + 2]
            mask_stack = mask[slice_idx - 1:slice_idx + 2]  # Corresponding masks for each slice

            # Resize slices and masks
            resized_img_stack = np.stack([cv2.resize(slice_, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
                                        for slice_ in img_stack], axis=0)
            resized_mask_stack = np.stack([cv2.resize(mask_slice, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
                                        for mask_slice in mask_stack], axis=0)

            # Append resized stack and corresponding masks to all_patches
            self.all_patches.append((resized_img_stack, resized_mask_stack))


    def augment_image(self, image, mask):
        """Applies Albumentations transformations."""
        image_np = image.transpose(1, 2, 0) if image.ndim == 3 else image
        mask_np = mask.transpose(1, 2, 0) if mask.ndim == 3 else mask

        augmented = self.transform(image=image_np, mask=mask_np)
        aug_image = augmented["image"].transpose(2, 0, 1)
        aug_mask = augmented["mask"].transpose(2, 0, 1)

        return aug_image, aug_mask



    

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, epochs_no_improve=0, best_loss=None):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = best_loss
        self.epochs_no_improve = epochs_no_improve
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.should_stop = True

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        power = 2**np.arange(0, 8).reshape(1, 1, 2, 2, 2).astype(np.float32)
        area = create_table_neighbour_code_to_surface_area((1, 1, 1)).astype(np.float32)
        self.power = nn.Parameter(torch.from_numpy(power), requires_grad=False)
        self.kernel = nn.Parameter(torch.ones(1, 1, 2, 2, 2), requires_grad=False)
        self.area = nn.Parameter(torch.from_numpy(area), requires_grad=False)
        
    def forward(self, preds, targets, eps=1e-5):
        """
        preds: tensor of shape [bs, 1, d, h, w]
        targets: tensor of shape [bs, 1, d, h, w]
        """
        bsz = preds.shape[0]

        # voxel logits to cube logits
        foreground_probs = F.conv3d(F.logsigmoid(preds), self.kernel).exp().flatten(1)
        background_probs = F.conv3d(F.logsigmoid(-preds), self.kernel).exp().flatten(1)
        surface_probs = 1 - foreground_probs - background_probs

        # ground truth to neighbour code
        with torch.no_grad():
            cubes_byte = F.conv3d(targets, self.power).to(torch.int32)
            gt_area = self.area[cubes_byte.reshape(-1)].reshape(bsz, -1)
            gt_foreground = (cubes_byte == 255).to(torch.float32).reshape(bsz, -1)
            gt_background = (cubes_byte == 0).to(torch.float32).reshape(bsz, -1)
            gt_surface = (gt_area > 0).to(torch.float32).reshape(bsz, -1)
        
        # dice
        foreground_dice = (2*(foreground_probs*gt_foreground).sum(-1)+eps) / (foreground_probs.sum(-1)+gt_foreground.sum(-1)+eps)
        background_dice = (2*(background_probs*gt_background).sum(-1)+eps) / (background_probs.sum(-1)+gt_background.sum(-1)+eps)
        surface_dice = (2*(surface_probs*gt_area).sum(-1)+eps) / (((surface_probs+gt_surface)*gt_area).sum(-1)+eps)
        dice = (foreground_dice + background_dice + surface_dice) / 3
        return 1 - dice.mean()

class BoundaryDoULossBinary(nn.Module):
    def __init__(self):
        super(BoundaryDoULossBinary, self).__init__()
        # Kernel for boundary calculation, set up for 3x3 convolution
        self.kernel = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).view(1, 1, 3, 3)

    def _adaptive_size(self, score, target):
        kernel = self.kernel.to(target.device)
        # Perform conv2d for each channel separately
        boundary_maps = []
        for c in range(score.size(1)):
            Y = nn.functional.conv2d(target[:, c:c+1, :, :], kernel, padding=1)
            Y = Y * target[:, c:c+1, :, :]
            Y[Y == 5] = 0
            
            C = torch.count_nonzero(Y)
            S = torch.count_nonzero(target[:, c:c+1, :, :])
            smooth = 1e-5
            alpha = 1 - (C + smooth) / (S + smooth)
            alpha = torch.clamp(2 * alpha - 1, max=0.8)

            intersect = torch.sum(score[:, c:c+1, :, :] * target[:, c:c+1, :, :])
            y_sum = torch.sum(target[:, c:c+1, :, :] * target[:, c:c+1, :, :])
            z_sum = torch.sum(score[:, c:c+1, :, :] * score[:, c:c+1, :, :])

            loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)
            boundary_maps.append(loss)
        
        return torch.stack(boundary_maps).mean()

    def forward(self, inputs, target):
        # Apply sigmoid for binary probabilities
        inputs = torch.sigmoid(inputs)
        target = target.float()  # Ensure target is float for multiplication
        
        assert inputs.size() == target.size(), f'predict {inputs.size()} & target {target.size()} shape do not match'
        
        return self._adaptive_size(inputs, target)

        
class CombinedLoss(nn.Module):
    def __init__(self, tversky_weight=1.0, focal_weight=0.9, custom_loss_weight=1.0, boundary_dou_weight=0.1):
        super(CombinedLoss, self).__init__()
        
        self.tversky_loss = smp.losses.TverskyLoss(mode="binary", from_logits=True, alpha=0.3, beta=0.7, smooth=1e-6)
        self.focal_loss = smp.losses.FocalLoss(mode="binary")
        self.boundary_dou_loss = BoundaryDoULossBinary()
        self.custom_loss = CustomLoss()
        
        self.custom_loss_weight = custom_loss_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.boundary_dou_weight = boundary_dou_weight

    def forward(self, logits, targets):
        tversky = self.tversky_loss(logits, targets)
        custom = self.custom_loss(logits.unsqueeze(1), targets.unsqueeze(1))
        focal = self.focal_loss(logits, targets)
        boundary_dou = self.boundary_dou_loss(logits, targets)
        
        combined_loss = (
            self.tversky_weight * tversky +
            self.custom_loss_weight * custom +
            self.focal_weight * focal +
            self.boundary_dou_weight * boundary_dou
        )
        
        return combined_loss

class RandomResize(DualTransform):
    """Resize the input to the given height and width.

    Args:
        height (int): Desired height of the output.
        width (int): Desired width of the output.
        scale_limit (float): Maximum scaling factor. Default is 0 (no scaling).
        scale_limit_x (float): Maximum scaling factor in x direction. Overrides scale_limit if provided.
        scale_limit_y (float): Maximum scaling factor in y direction. Overrides scale_limit if provided.
        interpolation (OpenCV flag): Interpolation method for resizing (e.g., cv2.INTER_LINEAR).
        p (float): Probability of applying the transform. Default is 1.
    """

    def __init__(self, height, width, scale_limit=0.0, scale_limit_x=None, scale_limit_y=None, interpolation=cv2.INTER_LINEAR, p=1):
        super(RandomResize, self).__init__(always_apply=False, p=p)  # Corrected this line
        self.height = height
        self.width = width
        self.scale_limit = (1 - scale_limit, 1 + scale_limit) if scale_limit != 0 else (1.0, 1.0)
        self.scale_limit_x = (1 - scale_limit_x, 1 + scale_limit_x) if scale_limit_x is not None else self.scale_limit
        self.scale_limit_y = (1 - scale_limit_y, 1 + scale_limit_y) if scale_limit_y is not None else self.scale_limit
        self.interpolation = interpolation
        
    def get_params(self):
        scale_x = random.uniform(self.scale_limit_x[0], self.scale_limit_x[1])
        scale_y = random.uniform(self.scale_limit_y[0], self.scale_limit_y[1])
        return {"scale_x": scale_x, "scale_y": scale_y}

    def apply(self, img, scale_x=1.0, scale_y=1.0, **params):
        if random.random() > self.p:  # Skip transformation based on probability
            return img
        
        new_height = int(scale_y * self.height)
        new_width = int(scale_x * self.width)
        return cv2.resize(img, (new_width, new_height), interpolation=self.interpolation)

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale-invariant, no changes needed
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        raise NotImplementedError

    def get_transform_init_args_names(self):
        return ("height", "width", "scale_limit", "scale_limit_x", "scale_limit_y", "interpolation")


        
######################################### MAIN PART #########################################
def train(train_dataset, val_dataset, batch_s=56, num_w=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    #create loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_s, shuffle=True, pin_memory=True, num_workers=num_w)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_s, shuffle=False, pin_memory=True, num_workers=num_w)

    #create model
    model = torchseg.Unet(
        "maxvit_small_tf_512",
        in_channels=3,
        classes=3,
        encoder_weights=None,
        encoder_depth=5,
        decoder_channels=(512, 256, 128, 64, 32),
        decoder_attention_type="scse"
    )

    name = model.name
    model = torch.nn.DataParallel(model).to(device)

    #craete loss
    criterion = CombinedLoss().cuda()

    num_epochs = 45
    early_stopping = EarlyStopping(patience=50, min_delta=0)

    # Build optimizer & scheduler
    num_training_steps = len(train_dataloader) * 30
    num_warmup_steps = int(num_training_steps * 0.1)

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=5e-2)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps)

    if resume:
        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scaler = torch.cuda.amp.GradScaler()
    if resume:
        scaler.load_state_dict(checkpoint['scaler'])
        prev_epoch = checkpoint['epoch']+1
        best_iou = checkpoint['val_loss']
    else:
        best_iou = -float('inf')
        prev_epoch = 1
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


    for epoch in range(prev_epoch, num_epochs+1):
        model.train()
        epoch_train_losses = []

        for batch_num, (images, masks) in enumerate(train_dataloader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=True):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            if batch_num % 4 == 0 or batch_num == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()

            train_loss = loss.item()
            epoch_train_losses.append(train_loss)
            

            if batch_num % 20 == 0:
                log_message = f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_num+1}/{len(train_dataloader)}], Batch Loss: {train_loss:.8f}'
                logging.info(log_message)

        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)

        gc.collect()
        torch.cuda.empty_cache()

        # Validation
        model.eval()
        running_val_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        with torch.no_grad():
            for images, masks in val_dataloader:
                images = images.to(device)
                masks = masks.to(device)

                with torch.amp.autocast(device_type="cuda", enabled=True):
                    outputs = model(images)
                    val_loss = criterion(outputs, masks)

                running_val_loss += val_loss.item()

                running_dice += dice_coef(masks, outputs)
                running_iou += iou_coef(masks, outputs)

        avg_val_loss = running_val_loss / len(val_dataloader)
        avg_val_iou = running_iou / len(val_dataloader)
        avg_val_dice = running_dice / len(val_dataloader)

        log_message = f"Epoch {epoch+1} - Training Loss: {avg_train_loss:.8f}, Validation Loss: {avg_val_loss:.8f}, IoU: {avg_val_iou}, Dice: {avg_val_dice}"
        logging.info(log_message)

        gc.collect()
        torch.cuda.empty_cache()

        if avg_val_iou > best_iou:
            logging.info(f"Validation IoU Improved ({best_iou:.6f} --> {avg_val_iou:.6f}), Saving Model")
            best_iou = avg_val_iou
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_iou,
                'scaler': scaler.state_dict()
            }, f"{name}.pth")
            logging.info(f"Model saved as {name}.pth")

        early_stopping(avg_val_iou)
        if early_stopping.should_stop:
            print("Early stopping triggered")
            break

    print('Training complete')


def run():
    
    train_orientations = ["kidney_1_dense"]#, "kidney_1_dense_xz", "kidney_1_dense_yz", "kidney_1_voi", "kidney_2", "kidney_2_xz", "kidney_2_yz", "kidney_3_xz", "kidney_3_yz"]

    val_orientations = ["kidney_3"]

    path = '/mmap/'
    image_size=1024
    
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.Affine(scale={"x":(0.7, 1.3), "y":(0.7, 1.3)}, translate_percent={"x":(0, 0.1), "y":(0, 0.1)}, rotate=(-30, 30), shear=(-20, 20), p=0.5),
        A.OneOf([
            A.Blur(blur_limit=3, p=0.2),
            A.MedianBlur(blur_limit=3, p=0.2),
        ], p=1.0),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, border_mode=1, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.1, border_mode=1, p=0.5)
        ], p=0.4),

        A.Compose([
            RandomResize(height=1024, width=1024, scale_limit=0.2, interpolation=cv2.INTER_LINEAR, p=1.0),
            A.PadIfNeeded(image_size, image_size, position="random", border_mode=cv2.BORDER_REPLICATE, p=1.0),
            A.RandomCrop(image_size, image_size, p=1.0)
        ], p=0.5),

        A.GaussNoise(var_limit=0.05, p=0.2),
    ])

    train_dataset = VolumeDataset(
        data_dir=path,
        kidney_orientations=train_orientations,
        target_size=image_size,
        transform=transforms
    )

    val_dataset = VolumeDataset(
        data_dir=path,
        kidney_orientations=val_orientations,
        transform=A.Compose([A.Resize(image_size, image_size, interpolation=cv2.INTER_NEAREST)])
    )
    
    train(train_dataset, val_dataset, batch_s=24, num_w=32)

if __name__ == "__main__":
    print("Starting training...")
    run()
