# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2026 - Fujitsu Limited
# All rights reserved.
"""Vote-based decision that turns scores and clusters into free-rider flags.

Clustering proposes a suspicious group; the per-score threshold tests say
whether that group really looks anomalous. This module combines the two,
which is the step that keeps an unreliable cluster split from producing
free-rider labels on its own (Sec. IV-E of the paper, equations (13) and
(14)).
"""

from typing import Dict, List, Sequence, Tuple, Union


#: Cluster labels that are candidates for being flagged. 0 is the group
#: nearest the origin in the standardized score space and is always
#: treated as benign; 1 is the farthest group and 2, when the three-way
#: variant is used, the second farthest.
SUSPICIOUS_LABELS = (1, 2)


def decide_free_riders_with_gate_c3(
    cluster_far: Sequence[int],
    dev_flags: Sequence[int],
    gamma_flags: Sequence[int],
    majority_thr: float = 0.5,
    strict: bool = False,
    return_details: bool = False,
) -> Union[List[int], Tuple[List[int], Dict]]:
    """Flag the suspicious clusters whose members exceed either threshold.

    Each candidate cluster is examined on its own. Within a cluster, the
    fraction of clients above the Dev threshold and the fraction above the
    gamma threshold are computed; if either reaches ``majority_thr``, every
    client in that cluster is labelled a free-rider. Clusters that pass
    neither test are left benign, and so is cluster 0.

    Works with two-way labels (0/1) as well as three-way labels (0/1/2).
    If every client is in cluster 0 the round is benign and the vote is
    skipped.

    Parameters
    ----------
    cluster_far:
        Cluster label per client: 0 for the benign-side group, 1 and 2 for
        the groups farther from the origin.
    dev_flags:
        1 where the deviation score exceeds its threshold, else 0.
    gamma_flags:
        1 where the similarity score exceeds its threshold, else 0.
    majority_thr:
        Fraction of a cluster that must exceed a threshold.
    strict:
        Compare with ``>`` instead of ``>=``.
    return_details:
        Also return the per-cluster tallies.

    Returns
    -------
    list of int
        1 for each client labelled a free-rider, 0 otherwise. When
        ``return_details`` is set, a dictionary of per-cluster tallies is
        returned alongside.

    Raises
    ------
    ValueError
        If the three input sequences have different lengths.
    """
    if not (len(cluster_far) == len(dev_flags) == len(gamma_flags)):
        raise ValueError(
            "cluster_far, dev_flags and gamma_flags must have the same "
            f"length: {len(cluster_far)}, {len(dev_flags)}, {len(gamma_flags)}"
        )

    n_clients = len(cluster_far)
    flags = [0] * n_clients

    # No cluster was separated out, so there is nothing to vote on.
    if all(label == 0 for label in cluster_far):
        if return_details:
            empty = _empty_tally()
            return flags, {
                "group1": empty,
                "group2": dict(empty),
                "all_clusters_zero": True,
                "note": "every client fell in cluster 0; the round is benign.",
            }
        return flags

    tallies = {}
    for position, label in enumerate(SUSPICIOUS_LABELS, start=1):
        tally = _tally_cluster(
            label, cluster_far, dev_flags, gamma_flags, majority_thr, strict
        )
        tallies[f"group{position}"] = tally
        if tally["gate_passed"]:
            for index in tally["indices"]:
                flags[index] = 1

    if return_details:
        tallies["all_clusters_zero"] = False
        return flags, tallies
    return flags


def _empty_tally() -> Dict:
    return {
        "indices": [],
        "dev_sum": 0,
        "gamma_sum": 0,
        "n": 0,
        "dev_ratio": 0.0,
        "gamma_ratio": 0.0,
        "gate_passed": False,
    }


def _tally_cluster(
    label: int,
    cluster_far: Sequence[int],
    dev_flags: Sequence[int],
    gamma_flags: Sequence[int],
    majority_thr: float,
    strict: bool,
) -> Dict:
    """Count threshold exceedances inside one cluster and apply the vote."""
    indices = [i for i, value in enumerate(cluster_far) if value == label]
    if not indices:
        return _empty_tally()

    dev_sum = sum(dev_flags[i] for i in indices)
    gamma_sum = sum(gamma_flags[i] for i in indices)
    size = len(indices)
    dev_ratio = dev_sum / size
    gamma_ratio = gamma_sum / size

    if strict:
        gate_passed = dev_ratio > majority_thr or gamma_ratio > majority_thr
    else:
        gate_passed = dev_ratio >= majority_thr or gamma_ratio >= majority_thr

    return {
        "indices": indices,
        "dev_sum": dev_sum,
        "gamma_sum": gamma_sum,
        "n": size,
        "dev_ratio": dev_ratio,
        "gamma_ratio": gamma_ratio,
        "gate_passed": gate_passed,
    }
