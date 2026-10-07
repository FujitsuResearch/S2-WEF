# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2026 - Fujitsu Limited
# All rights reserved.
"""Model architectures used by the participants.

Three architectures cover the four datasets: LeNet-5 for MNIST, a two-layer
perceptron for the Adult table, and a 32x32-oriented ResNet-18 for CIFAR-10
and CIFAR-100. The ResNet definition is third-party code and lives in
:mod:`third_party.pytorch_cifar.resnet`; this module only selects and
instantiates it.

References for the architectures: Y. LeCun et al., "Gradient-based learning
applied to document recognition," Proc. IEEE 1998 (LeNet); I. Tolstikhin et
al., "MLP-Mixer," NeurIPS 2021 (MLP); K. He et al., "Deep Residual Learning
for Image Recognition," CVPR 2016 (ResNet).
"""

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn

from third_party.pytorch_cifar.resnet import BasicBlock, ResNet


#: Residual block counts of ResNet-18, one entry per stage.
RESNET18_BLOCKS = [2, 2, 2, 2]

#: Output width of the CIFAR-10 ResNet. The value exceeds the ten CIFAR-10
#: classes; the surplus units are simply never selected. It is kept because
#: the WEF-matrix is read from the final weight tensor, so its shape -- and
#: therefore every reported score -- depends on this number. Changing it to
#: 10 would alter the published results.
CIFAR10_OUTPUT_WIDTH = 43

#: Output width of the CIFAR-100 ResNet, matching its hundred classes.
CIFAR100_OUTPUT_WIDTH = 100


class LeNet(nn.Module):
    """LeNet-5 for single-channel 28x28 images.

    Two convolutional stages, each followed by a ReLU and a 2x2 max pool,
    feed three fully connected layers. Pooling is folded into the
    convolutional blocks so that the flattened feature map is 16 channels of
    5x5.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 84), nn.ReLU())
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)


class MLP(nn.Module):
    """Two-layer perceptron for the one-hot encoded Adult table.

    Returns log-probabilities rather than logits, which keeps the output on
    the same scale as the other architectures under cross-entropy.
    """

    def __init__(self, input_dim: int = 86, num_classes: int = 2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def _build_cifar10_resnet() -> nn.Module:
    return ResNet(BasicBlock, RESNET18_BLOCKS,
                  num_classes=CIFAR10_OUTPUT_WIDTH)


def _build_cifar100_resnet() -> nn.Module:
    return ResNet(BasicBlock, RESNET18_BLOCKS,
                  num_classes=CIFAR100_OUTPUT_WIDTH)


#: Configuration name to constructor. The names are the values ``model_name``
#: may take in conf.json.
ARCHITECTURES: Dict[str, Any] = {
    "LeNet": LeNet,
    "mlp": MLP,
    "resnet": _build_cifar10_resnet,
    "resnet100": _build_cifar100_resnet,
}


def get_model(name: str) -> nn.Module:
    """Instantiate an architecture by configuration name.

    The model is moved to the GPU when one is available, because every
    participant deep-copies it each round and training it on the CPU would
    dominate the run time.

    Raises
    ------
    ValueError
        If ``name`` is not a known architecture.
    """
    try:
        build = ARCHITECTURES[name]
    except KeyError:
        raise ValueError(
            f"unknown model_name {name!r}; expected one of "
            f"{sorted(ARCHITECTURES)}"
        ) from None

    model = build()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

