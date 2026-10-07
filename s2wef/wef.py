# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2026 - Fujitsu Limited
# All rights reserved.
"""Construction of the Weight Evolving Frequency (WEF) matrix.
"""

from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module


def select_layer(model: Module, layer_index: int) -> Tensor:
    """Return the parameter tensor the WEF-matrix is built from.

    Raises
    ------
    IndexError
        If ``layer_index`` does not address a parameter of ``model``.
    """
    parameters = tuple(model.parameters())
    count = len(parameters)
    if not -count <= layer_index < count:
        raise IndexError(
            f"layer_index={layer_index} is invalid for a model with "
            f"{count} parameter tensors."
        )
    return parameters[layer_index].detach()


def empty_matrix(model: Module, layer_index: int) -> Tensor:
    """Allocate a zero WEF-matrix matching the selected layer.

    The matrix lives on the CPU: it is transmitted to the server as a
    NumPy array and is never involved in gradient computation.
    """
    return torch.zeros(select_layer(model, layer_index).size())


def measure_movement(
    current: Module,
    reference: Module,
    layer_index: int,
) -> Tuple[Tensor, Tensor]:
    """Compare one layer of two models.

    Returns
    -------
    tuple of (Tensor, Tensor)
        The mean absolute change across the layer -- the dynamic threshold
        alpha of equation (1) -- and the element-wise absolute change.

    Raises
    ------
    ValueError
        If the two selected tensors have different shapes.
    """
    current_layer = select_layer(current, layer_index)
    reference_layer = select_layer(reference, layer_index)

    if current_layer.shape != reference_layer.shape:
        raise ValueError(
            "the compared parameter tensors must have the same shape: "
            f"{tuple(current_layer.shape)} != {tuple(reference_layer.shape)}"
        )

    movement = torch.abs(current_layer - reference_layer)
    return movement.mean(), movement


def count_evolved_weights(
    matrix: Tensor,
    current: Module,
    reference: Module,
    layer_index: int,
) -> Tensor:
    """Increment the WEF-matrix for one local iteration.

    Elements that moved further than the mean absolute movement gain one
    count; the rest are left alone. This is the update rule of equation (1).
    """
    threshold, movement = measure_movement(current, reference, layer_index)
    increment = (movement > threshold).to(dtype=matrix.dtype, device=matrix.device)
    return matrix + increment
