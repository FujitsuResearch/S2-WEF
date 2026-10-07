# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2026 - Fujitsu Limited
# All rights reserved.
"""Server-side model aggregation and evaluation for S2-WEF.

The server keeps the global model, averages the updates submitted by the
clients it considers benign, and measures the accuracy of the resulting
model on a held-out evaluation set.

Aggregation follows FedAvg without sample-count weighting: the server
cannot verify how many samples a client actually used, so every accepted
update contributes equally.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from s2wef import models


StateDict = Mapping[str, torch.Tensor]


class Aggregator:
    """Holds the global model and combines accepted client updates.

    Parameters
    ----------
    conf:
        Experiment configuration. ``model_name`` selects the architecture
        and ``batch_size`` sizes the evaluation loader.
    eval_dataset:
        Dataset used to report main-task accuracy of the global model.
    holdout_dataset:
        Optional secondary dataset kept for convenience; unused by the
        aggregation logic itself.
    """

    def __init__(
        self,
        conf: Dict[str, Any],
        eval_dataset,
        holdout_dataset=None,
    ) -> None:
        self.conf = conf
        self.global_model = models.get_model(conf["model_name"])

        batch_size = conf["batch_size"]
        self.eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
        self.holdout_loader: Optional[DataLoader] = None
        if holdout_dataset is not None:
            self.holdout_loader = DataLoader(
                holdout_dataset, batch_size=batch_size, shuffle=True
            )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def current_state(self) -> Dict[str, torch.Tensor]:
        """Return a detached copy of the global model parameters."""
        return {name: tensor.detach().clone()
                for name, tensor in self.global_model.state_dict().items()}

    def aggregate(self, accepted_states: Sequence[StateDict]) -> int:
        """Replace the global model with the mean of the accepted updates.

        Implements ``w_g^{T+1} = (1 / |B|) * sum_{i in B} w_i``, where ``B``
        is the set of clients the detector did not flag in this round.

        Floating-point tensors are averaged in double precision and cast
        back to their original dtype. Integer buffers -- ``num_batches_tracked``
        of every BatchNorm layer, for instance -- are truncated after
        averaging so that the global model keeps a valid integer count.

        Parameters
        ----------
        accepted_states:
            Full parameter dictionaries of the clients whose updates are
            accepted this round. An empty sequence leaves the global model
            untouched, which happens when every client is flagged.

        Returns
        -------
        int
            Number of updates that were averaged.

        Raises
        ------
        KeyError
            If a submitted update is missing a parameter of the global model.
        """
        if not accepted_states:
            return 0

        reference = self.global_model.state_dict()
        averaged: Dict[str, torch.Tensor] = {}

        for name, reference_tensor in reference.items():
            try:
                contributions = [state[name] for state in accepted_states]
            except KeyError as missing:
                raise KeyError(
                    f"submitted update does not contain parameter {missing}"
                ) from None

            stacked = torch.stack([
                tensor.detach().to(device=reference_tensor.device,
                                   dtype=torch.float64)
                for tensor in contributions
            ])
            mean_tensor = stacked.mean(dim=0)

            if reference_tensor.is_floating_point():
                averaged[name] = mean_tensor.to(reference_tensor.dtype)
            else:
                averaged[name] = mean_tensor.trunc().to(reference_tensor.dtype)

        self.global_model.load_state_dict(averaged)
        return len(accepted_states)

    def aggregate_from_deltas(
        self,
        deltas: Sequence[StateDict],
    ) -> int:
        """Aggregate updates expressed as differences from the global model.

        Convenience wrapper for callers that keep ``w_i - w_g`` rather than
        ``w_i``. Because every client starts the round from the same global
        model, averaging the reconstructed weights is equivalent to adding
        the mean difference to the current global model.
        """
        if not deltas:
            return 0

        base = self.global_model.state_dict()
        reconstructed = [
            {name: base[name] + delta[name] for name in base}
            for delta in deltas
        ]
        return self.aggregate(reconstructed)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, device: str = "cpu") -> Tuple[float, float]:
        """Measure main-task accuracy and mean loss of the global model.

        Returns
        -------
        tuple of (float, float)
            Accuracy in percent and cross-entropy averaged over samples.
        """
        self.global_model.eval()

        total_loss = 0.0
        correct = 0
        seen = 0

        with torch.no_grad():
            for inputs, targets in self.eval_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                seen += targets.size(0)

                outputs = self.global_model(inputs)
                total_loss += torch.nn.functional.cross_entropy(
                    outputs, targets.long(), reduction="sum"
                ).item()
                predictions = outputs.argmax(dim=1)
                correct += predictions.eq(targets.view_as(predictions)).sum().item()

        if seen == 0:
            return 0.0, 0.0
        return 100.0 * correct / seen, total_loss / seen


# ----------------------------------------------------------------------
# Backwards-compatible aliases
# ----------------------------------------------------------------------
Server = Aggregator