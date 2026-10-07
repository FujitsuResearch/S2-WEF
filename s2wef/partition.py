# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2026 - Fujitsu Limited
# All rights reserved.
"""Non-IID partitioning of a training set across the participants.

The Dirichlet label-distribution-skew draw itself comes from NIID-Bench and
lives in :mod:`third_party.niid_bench.partition`. This module decides how
that draw is used: which labels it sees, how small a shard is allowed to be,
and when to give up retrying.

"""

from typing import Any, Dict, List

from s2wef.datasets import NUM_CLASSES, labels_of
from third_party.niid_bench.partition import (
    draw_label_skewed_shards,
    shuffle_within_shards,
)


#: Smallest shard a party may receive before the draw is repeated. Adult has
#: only two classes, so a skewed draw can leave a party with very little; a
#: higher floor keeps every party trainable.
MIN_SHARD_SIZE: Dict[str, int] = {"adult": 600}
DEFAULT_MIN_SHARD_SIZE = 10

#: The floor is kept only if the training set can supply it with this much
#: headroom, since a skewed draw needs slack to clear the floor at all.
FEASIBLE_HEADROOM = 2

#: Denominator used to pick a replacement floor once the configured one is
#: found to be out of reach.
FEASIBLE_FRACTION = 3

#: Upper bound on retries. The feasibility check above makes the floor
#: reachable, so hitting this bound means the configuration is pathological
#: and should fail loudly rather than hang.
MAX_PARTITION_ATTEMPTS = 1000


def partition_data(
    conf: Dict[str, Any],
    train_dataset,
    n_parties: int,
    beta: float,
) -> List[List[int]]:
    """Split a training set among ``n_parties`` with label distribution skew.

    Parameters
    ----------
    conf:
        Experiment configuration; ``dataset`` selects the label lookup and
        the minimum shard size.
    train_dataset:
        Dataset to split. Must expose ``targets``.
    n_parties:
        Number of participants to split across.
    beta:
        Concentration parameter of the Dirichlet distribution. Smaller
        values concentrate each class in fewer participants.

    Returns
    -------
    list of list of int
        One shuffled index list per participant, in participant order.

    Raises
    ------
    ValueError
        If ``n_parties`` is not positive, or the dataset is unknown.
    RuntimeError
        If no draw clears the minimum shard size within the retry bound.
    """
    if n_parties < 1:
        raise ValueError(f"n_parties must be positive, got {n_parties}")

    name = conf["dataset"]
    try:
        n_classes = NUM_CLASSES[name]
    except KeyError:
        raise ValueError(
            f"unknown dataset {name!r}; expected one of {sorted(NUM_CLASSES)}"
        ) from None

    labels = labels_of(train_dataset)
    n_train = len(labels)
    min_shard_size = _reachable_floor(name, n_parties, n_train)

    print(f"[partition] {name}: {n_train} samples -> {n_parties} parties "
          f"(beta={beta}, min shard={min_shard_size})")

    shards: List[List[int]] = []
    smallest = 0
    attempts = 0

    while smallest < min_shard_size:
        attempts += 1
        if attempts > MAX_PARTITION_ATTEMPTS:
            raise RuntimeError(
                f"could not partition {name} into {n_parties} parties with "
                f"at least {min_shard_size} samples each after "
                f"{MAX_PARTITION_ATTEMPTS} draws "
                f"(best smallest shard: {smallest})"
            )
        shards, smallest = draw_label_skewed_shards(
            labels, n_classes, n_parties, beta
        )

    shards = shuffle_within_shards(shards)

    print(f"[partition] shard sizes: {[len(shard) for shard in shards]} "
          f"(total {sum(len(shard) for shard in shards)}/{n_train})")
    return shards


def _reachable_floor(name: str, n_parties: int, n_train: int) -> int:
    """Return a minimum shard size the retry loop can actually satisfy."""
    requested = MIN_SHARD_SIZE.get(name, DEFAULT_MIN_SHARD_SIZE)

    if requested * n_parties * FEASIBLE_HEADROOM <= n_train:
        return requested

    lowered = max(DEFAULT_MIN_SHARD_SIZE,
                  int(n_train / FEASIBLE_FRACTION / n_parties))
    print(f"[partition] min shard size lowered {requested} -> {lowered} "
          f"(n_parties={n_parties}, n_train={n_train})")
    return lowered
