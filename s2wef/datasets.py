# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2026 - Fujitsu Limited
# All rights reserved.
"""Dataset loading and per-class index extraction.

Four datasets are supported: MNIST, Adult, CIFAR-10 and CIFAR-100. Each is
returned as three parts:

* a training set, later partitioned among the participants;
* an evaluation set, on which the server measures main-task accuracy;
* a small holdout carved out of the evaluation set, which participants use
  to check the quality of their own local model under an IID split.

The module also exposes the per-class index lists needed to build IID
shards, and the class counts used by the Dirichlet partitioner, so that
dataset-specific branching lives in one place.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchvision.datasets as tv_datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset


#: Number of label values in each dataset.
NUM_CLASSES: Dict[str, int] = {
    "adult": 2,
    "mnist": 10,
    "cifar10": 10,
    "cifar100": 100,
}

#: Channel means and standard deviations used to normalize the CIFAR images.
CIFAR10_STATS = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
CIFAR100_STATS = ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))

#: Fraction of the evaluation set reserved as the participants' holdout.
HOLDOUT_FRACTION = 0.2

#: Seed for the holdout split. Fixed so that the holdout is the same in
#: every run, independently of the experiment seed.
HOLDOUT_SEED = 42

#: Fraction of the Adult training file kept for training; the rest becomes
#: the evaluation set.
ADULT_TRAIN_FRACTION = 0.8


@dataclass
class FederatedDatasets:
    """The three views of a dataset that an experiment needs."""

    train: Any
    evaluation: Any
    holdout: Any


class TabularDataset(Dataset):
    """In-memory dataset for tabular features already encoded as tensors.

    Rows are kept on the requested device so that small tabular batches do
    not pay a host-to-device copy on every step. The ``targets`` attribute
    mirrors the convention of the torchvision datasets, which lets the
    partitioner treat every dataset the same way.
    """

    def __init__(self, features: Tensor, targets: Tensor, device=None) -> None:
        if len(features) != len(targets):
            raise ValueError(
                f"features and targets must have the same length: "
                f"{len(features)} != {len(targets)}"
            )
        self.features = features.to(device)
        self.targets = targets.to(device)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return self.features[index], self.targets[index]


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_datasets(root: str, name: str, device=None) -> FederatedDatasets:
    """Load one of the supported datasets.

    Parameters
    ----------
    root:
        Directory holding the raw data. Image datasets are downloaded here
        on first use.
    name:
        One of the keys of :data:`NUM_CLASSES`.
    device:
        Device the Adult tensors are placed on; ignored otherwise.

    Raises
    ------
    ValueError
        If ``name`` is not a supported dataset.
    """
    loaders = {
        "mnist": _load_mnist,
        "cifar10": _load_cifar10,
        "cifar100": _load_cifar100,
        "adult": _load_adult,
    }
    try:
        loader = loaders[name]
    except KeyError:
        raise ValueError(
            f"unknown dataset {name!r}; expected one of {sorted(loaders)}"
        ) from None

    datasets = loader(root, device)
    print(f"[data] {name}: train={len(datasets.train)} "
          f"eval={len(datasets.evaluation)} holdout={len(datasets.holdout)}")
    return datasets


def _carve_holdout(evaluation):
    """Split off a fixed fraction of the evaluation set for the clients."""
    _, holdout = train_test_split(
        evaluation, test_size=HOLDOUT_FRACTION, random_state=HOLDOUT_SEED
    )
    return holdout


def _load_mnist(root: str, device) -> FederatedDatasets:
    to_tensor = transforms.ToTensor()
    train = tv_datasets.MNIST(root, train=True, download=True, transform=to_tensor)
    evaluation = tv_datasets.MNIST(root, train=False, transform=to_tensor)
    return FederatedDatasets(train, evaluation, _carve_holdout(evaluation))


def _cifar_transforms(stats) -> Tuple[Any, Any]:
    """Augmentation for training and plain normalization for evaluation."""
    mean, std = stats
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transform, eval_transform


def _load_cifar10(root: str, device) -> FederatedDatasets:
    train_transform, eval_transform = _cifar_transforms(CIFAR10_STATS)
    train = tv_datasets.CIFAR10(
        root, train=True, download=True, transform=train_transform
    )
    evaluation = tv_datasets.CIFAR10(root, train=False, transform=eval_transform)
    return FederatedDatasets(train, evaluation, _carve_holdout(evaluation))


def _load_cifar100(root: str, device) -> FederatedDatasets:
    train_transform, eval_transform = _cifar_transforms(CIFAR100_STATS)
    train = tv_datasets.CIFAR100(
        root, train=True, download=True, transform=train_transform
    )
    evaluation = tv_datasets.CIFAR100(root, train=False, transform=eval_transform)
    return FederatedDatasets(train, evaluation, _carve_holdout(evaluation))


def _load_adult(root: str, device) -> FederatedDatasets:
    """Load the Adult census table and split it into the three views.

    The preprocessing -- balancing the two income classes, one-hot encoding
    the categorical columns and standardizing the numeric ones -- lives in
    :mod:`s2wef.load_adult`.
    """
    from s2wef.adult import load_adult as load_adult_frames

    train_features, train_labels, test_features, test_labels = load_adult_frames()

    x_train = torch.tensor(train_features.values, requires_grad=False).float()
    y_train = torch.tensor(train_labels.values, requires_grad=False).long()
    x_test = torch.tensor(test_features.values, requires_grad=False).float()
    y_test = torch.tensor(test_labels.values, requires_grad=False).long()

    _report_class_balance("train", y_train)
    _report_class_balance("test", y_test)

    train_indices, eval_indices = _shuffled_split(
        len(x_train), ADULT_TRAIN_FRACTION
    )

    return FederatedDatasets(
        train=TabularDataset(x_train[train_indices], y_train[train_indices], device),
        evaluation=TabularDataset(x_train[eval_indices], y_train[eval_indices], device),
        holdout=TabularDataset(x_test, y_test, device),
    )


def _report_class_balance(split: str, labels: Tensor) -> None:
    positive = int((labels == 1).sum().item())
    negative = int((labels == 0).sum().item())
    total = positive + negative
    print(f"[data] adult {split}: {total} rows, "
          f"positive={positive} ({positive / total:.2%}), "
          f"negative={negative} ({negative / total:.2%})")


def _shuffled_split(n_samples: int, train_fraction: float):
    """Shuffle ``range(n_samples)`` and cut it in two.

    Note: this draws from the module-level ``random`` generator, which the
    experiment driver also uses. The behaviour is kept as-is so that
    published results stay reproducible; a private generator would change
    the downstream random stream.
    """
    import random

    indices = list(range(n_samples))
    random.shuffle(indices)
    split_point = int(n_samples * train_fraction)
    return indices[:split_point], indices[split_point:]


# ----------------------------------------------------------------------
# Label access
# ----------------------------------------------------------------------
def labels_of(dataset) -> np.ndarray:
    """Return the labels of ``dataset`` as a flat NumPy array.

    Torchvision exposes ``targets`` as a tensor for MNIST and as a Python
    list for the CIFAR datasets; :class:`TabularDataset` keeps a tensor
    that may live on the GPU. This normalizes all three.

    Raises
    ------
    AttributeError
        If the dataset does not expose a ``targets`` attribute.
    """
    try:
        targets = dataset.targets
    except AttributeError:
        raise AttributeError(
            f"{type(dataset).__name__} does not expose a 'targets' attribute"
        ) from None

    if isinstance(targets, torch.Tensor):
        return targets.detach().cpu().numpy()
    return np.asarray(targets)


def class_index_map(dataset, name: str) -> List[List[int]]:
    """Group sample positions by label.

    Returns one list of indices per class, in label order. This is what an
    IID shard is drawn from: each participant samples the same number of
    rows from every class.

    Raises
    ------
    ValueError
        If ``name`` is not a supported dataset.
    """
    try:
        class_count = NUM_CLASSES[name]
    except KeyError:
        raise ValueError(
            f"unknown dataset {name!r}; expected one of {sorted(NUM_CLASSES)}"
        ) from None

    labels = labels_of(dataset)
    return [np.where(labels == label)[0].tolist() for label in range(class_count)]


# ----------------------------------------------------------------------
# Backwards-compatible shim
# ----------------------------------------------------------------------
def get_dataset(root: str, name: str, device=None):
    """Tuple-returning wrapper kept for older call sites."""
    datasets = load_datasets(root, name, device)
    return datasets.train, datasets.evaluation, datasets.holdout