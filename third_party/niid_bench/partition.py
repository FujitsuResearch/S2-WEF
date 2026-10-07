# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Yiqun Diao, Qinbin Li
#
# Derived from the `partition_data` function of NIID-Bench
# (https://github.com/Xtra-Computing/NIID-Bench, utils.py), specifically its
# `noniid-labeldir` partitioning strategy. See the LICENSE file in this
# directory for the full license text.
#
# Reference: Q. Li, Y. Diao, Q. Chen and B. He, "Federated Learning on
# Non-IID Data Silos: An Experimental Study," ICDE 2022.
#
# The original function wraps the draw below in a retry loop and applies a
# dataset-specific label lookup. Those parts are not reproduced here; see
# s2wef/partition.py for the caller that supplies them.
"""Dirichlet label-distribution-skew sampling of client shards."""

from typing import List, Sequence, Tuple

import numpy as np


def draw_label_skewed_shards(
    labels: np.ndarray,
    n_classes: int,
    n_parties: int,
    beta: float,
) -> Tuple[List[List[int]], int]:
    """Perform one draw of the label-distribution-skew partition.

    For each class in turn, the indices of that class are shuffled and split
    among the parties in proportions drawn from a Dirichlet distribution with
    concentration ``beta``. A party that already holds at least ``len(labels)
    / n_parties`` samples receives a proportion of zero for the remaining
    classes, which keeps shard sizes from diverging.

    Parameters
    ----------
    labels:
        Label of every sample, as a flat array.
    n_classes:
        Number of label values.
    n_parties:
        Number of shards to produce.
    beta:
        Concentration parameter of the Dirichlet distribution.

    Returns
    -------
    tuple
        The shards, one index list per party, and the size of the smallest
        shard. The caller decides whether that size is acceptable.
    """
    total = len(labels)
    share = total / n_parties
    shards: List[List[int]] = [[] for _ in range(n_parties)]
    smallest = 0

    for label in range(n_classes):
        class_indices = np.where(labels == label)[0]
        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet(np.repeat(beta, n_parties))
        proportions = np.array([
            weight * (len(shard) < share)
            for weight, shard in zip(proportions, shards)
        ])
        proportions = proportions / proportions.sum()
        cut_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]

        shards = [
            shard + chunk.tolist()
            for shard, chunk in zip(shards, np.split(class_indices, cut_points))
        ]
        smallest = min(len(shard) for shard in shards)

    return shards, smallest


def shuffle_within_shards(shards: Sequence[List[int]]) -> List[List[int]]:
    """Permute the indices inside each shard.

    Without this the indices of a shard stay grouped by class, which would
    make the order in which a client sees its samples depend on the label.
    """
    shuffled = []
    for shard in shards:
        np.random.shuffle(shard)
        shuffled.append(shard)
    return shuffled
