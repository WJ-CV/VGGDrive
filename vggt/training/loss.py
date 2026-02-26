# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field
from typing import Callable, Mapping, Any
import torch


@dataclass(eq=False)
class MultitaskLoss:
    """
    """
    def __init__(self, camera, depth, point, track, **kwargs):
        self.camera = camera
        self.depth = depth
        self.point = point
        self.track = track

    def forward(self, predictions) -> torch.Tensor:
        """
        Compute the total loss.
        """
        pass
