# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2026 - Fujitsu Limited
# All rights reserved.
"""Client-side local training and update submission.

A participant owns a private shard of the training data, runs a few local
iterations on the model it receives, and returns both the resulting
parameter change and the WEF-matrix recorded while training. A participant
that free-rides in a given round skips training and forges both artefacts
instead; the server cannot tell the two apart from the submission format.
"""

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from s2wef import attacks, wef


#: Samples drawn per class when building an IID shard, chosen so that every
#: dataset yields shards of roughly 5000 samples.
IID_SAMPLES_PER_CLASS = {
    "adult": (2, 2500),
    "mnist": (10, 600),
    "cifar10": (10, 500),
    "cifar100": (100, 50),
}

#: Fraction of a non-IID shard reserved for the participant's own held-out
#: split.
LOCAL_TRAIN_FRACTION = 0.8

#: Seed used for the local train/held-out split. Fixed so that a shard is
#: always divided the same way, independently of the experiment seed.
LOCAL_SPLIT_SEED = 0


@dataclass
class ClientUpdate:
    """What a participant hands back to the server in one round.

    Attributes
    ----------
    delta:
        Parameter change ``w_i - w_g`` for every entry of the state dict.
    wef_matrix:
        Counterfeit or genuine WEF-matrix for this round, as a NumPy array.
    model:
        The local model after the round; kept for optional inspection.
    """

    delta: Dict[str, Tensor]
    wef_matrix: np.ndarray
    model: Module


class Participant:
    """One organization taking part in the federation.

    Parameters
    ----------
    conf:
        Experiment configuration.
    model:
        Reference model, used only to size the local structures.
    train_dataset:
        Full training set; the participant reads its own indices from it.
    shard_indices:
        Under a non-IID split, the per-client index lists produced by the
        Dirichlet partitioner. Under an IID split, the per-class index
        lists to sample from.
    holdout_dataset:
        Shared evaluation set, used for the local accuracy check when the
        split is IID.
    client_id:
        Position of this participant in the federation.
    """

    def __init__(
        self,
        conf: Dict[str, Any],
        model: Module,
        train_dataset,
        shard_indices: Sequence[Sequence[int]],
        holdout_dataset,
        client_id: int,
    ) -> None:
        self.conf = conf
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.holdout_dataset = holdout_dataset

        # Parameters of the model broadcast in the previous round. Every
        # participant records them, because a benign client may switch to
        # free-riding later and the delta-based attacks need this history.
        self._previous_global: List[Tensor] = []

        batch_size = conf["batch_size"]
        non_iid = conf["no_iid"] == "yes"

        if non_iid:
            train_indices, local_test_indices = self._split_shard(
                list(shard_indices[client_id])
            )
            self.train_loader = DataLoader(
                train_dataset, batch_size=batch_size,
                sampler=SubsetRandomSampler(train_indices),
            )
            self.test_loader = DataLoader(
                train_dataset, batch_size=batch_size,
                sampler=SubsetRandomSampler(local_test_indices),
            )
        else:
            train_indices = self._sample_iid_shard(shard_indices)
            self.train_loader = DataLoader(
                train_dataset, batch_size=batch_size,
                sampler=SubsetRandomSampler(train_indices),
            )
            self.test_loader = DataLoader(holdout_dataset, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Shard construction
    # ------------------------------------------------------------------
    def _sample_iid_shard(self, class_indices: Sequence[Sequence[int]]) -> List[int]:
        """Draw a class-balanced shard with replacement.

        Raises
        ------
        KeyError
            If the configured dataset has no IID sampling recipe.
        """
        dataset = self.conf["dataset"]
        try:
            class_count, per_class = IID_SAMPLES_PER_CLASS[dataset]
        except KeyError:
            raise KeyError(
                f"no IID sampling recipe for dataset {dataset!r}; "
                f"known datasets: {sorted(IID_SAMPLES_PER_CLASS)}"
            ) from None

        indices: List[int] = []
        for label in range(class_count):
            indices.extend(random.choices(class_indices[label], k=per_class))
        return indices

    def _split_shard(self, indices: List[int]):
        """Reserve part of a non-IID shard as the participant's own test data.

        Note: this reseeds the module-level ``random`` generator, which the
        experiment driver also draws from. The behaviour is kept as-is so
        that published results stay reproducible; switching to a private
        ``random.Random(LOCAL_SPLIT_SEED)`` would change the attack
        schedule of the round-wise random scenario.
        """
        shuffled = list(indices)
        random.seed(LOCAL_SPLIT_SEED)
        random.shuffle(shuffled)

        split_point = int(len(shuffled) * LOCAL_TRAIN_FRACTION)
        return shuffled[:split_point], shuffled[split_point:]

    # ------------------------------------------------------------------
    # One round
    # ------------------------------------------------------------------
    def run_round(
        self,
        global_model: Module,
        global_round: int,
        free_riding: bool,
    ) -> ClientUpdate:
        """Produce this round's submission.

        Parameters
        ----------
        global_model:
            The model broadcast by the server this round.
        global_round:
            Zero-based round index.
        free_riding:
            Whether this participant forges its update instead of training.
        """
        local_model = copy.deepcopy(global_model)
        layer_index = self.conf["WEF-layer"]

        if free_riding:
            matrix = attacks.forge_update(
                attack=self.conf["free_riders_type"],
                local_model=local_model,
                global_model=global_model,
                previous_global_params=self._previous_global,
                scale=self.conf["sigma"],
                global_round=global_round,
                local_iterations=self.conf["local_epochs"],
                layer_index=layer_index,
            )
        else:
            matrix = self._train_locally(local_model, layer_index)

        delta = self._parameter_delta(local_model, global_model)
        self._previous_global = [
            parameter.detach().clone() for parameter in global_model.parameters()
        ]

        return ClientUpdate(
            delta=delta,
            wef_matrix=matrix.detach().cpu().numpy().copy(),
            model=local_model,
        )

    def _train_locally(self, local_model: Module, layer_index: int) -> Tensor:
        """Run the configured number of local iterations and record the WEF."""
        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=self.conf["lr"],
            momentum=self.conf["momentum"],
        )
        local_model.train()
        matrix = wef.empty_matrix(local_model, layer_index)

        for _ in range(self.conf["local_epochs"]):
            before = copy.deepcopy(local_model)

            for inputs, targets in self.train_loader:
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                optimizer.zero_grad()
                outputs = local_model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, targets.long())
                loss.backward()
                optimizer.step()

            matrix = wef.count_evolved_weights(
                matrix, local_model, before, layer_index
            )

        return matrix

    @staticmethod
    def _parameter_delta(
        local_model: Module,
        global_model: Module,
    ) -> Dict[str, Tensor]:
        """Element-wise difference between the local and the received model."""
        received = global_model.state_dict()
        return {
            name: tensor - received[name]
            for name, tensor in local_model.state_dict().items()
        }

    # ------------------------------------------------------------------
    # Local evaluation
    # ------------------------------------------------------------------
    def local_accuracy(self, model: Module) -> float:
        """Accuracy of ``model`` on this participant's held-out samples."""
        model.eval()
        correct = 0
        seen = 0

        with torch.no_grad():
            for _ in range(self.conf["local_epochs"]):
                for inputs, targets in self.test_loader:
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                    seen += targets.size(0)

                    predictions = model(inputs).argmax(dim=1)
                    correct += predictions.eq(
                        targets.view_as(predictions)
                    ).sum().item()

        if seen == 0:
            return 0.0
        return 100.0 * correct / seen


# ----------------------------------------------------------------------
# Backwards-compatible alias
# ----------------------------------------------------------------------
Client = Participant
