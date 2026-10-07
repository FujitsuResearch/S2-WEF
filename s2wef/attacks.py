# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2026 - Fujitsu Limited
# All rights reserved.
"""Free-rider attacks simulated on the client side.

Five attacks are supported. Four come from prior work:

* ``RWA``  -- random weight attack: sample every weight uniformly from
  ``[-scale, scale]``.
* ``SPA``  -- stochastic perturbations attack: add Gaussian noise whose
  amplitude decays with the round index.
* ``DWA``  -- delta weight attack: replay the difference between the two
  most recently broadcast global models.
* ``ADWA`` -- advanced delta weight attack: DWA plus Gaussian noise, so
  that colluding free-riders do not submit identical weights.

The fifth, ``AWCA`` (adaptive WEF-camouflage attack), is introduced in our
paper. It spreads the global-model difference over the local iterations and
rebuilds the WEF-matrix at each step, so the counterfeit matrix takes
intermediate values instead of only 0 and e.

Every function here forges weights *and* the matching counterfeit
WEF-matrix, because a client that is asked to upload a WEF-matrix must
fabricate one consistent with the weights it submits.
"""

import copy
from typing import Sequence

import torch
from scipy.stats import norm
from torch import Tensor
from torch.nn import Module, init

from s2wef import wef


#: Attacks that forge the end-of-round weights in a single step. Their
#: WEF-matrix is derived from one comparison and then scaled by the number
#: of local iterations, per equation (2).
ONE_SHOT_ATTACKS = ("RWA", "SPA", "DWA", "ADWA")

#: Attacks that forge weights iteration by iteration.
ITERATIVE_ATTACKS = ("AWCA",)

SUPPORTED_ATTACKS = ONE_SHOT_ATTACKS + ITERATIVE_ATTACKS

#: Noise scale used by SPA in the first round an attacker participates,
#: before it has observed two global models to estimate from.
SPA_COLD_START_SCALE = 0.01

#: Exponent of the round index in the SPA noise schedule.
SPA_DECAY_EXPONENT = -1


def forge_update(
    attack: str,
    local_model: Module,
    global_model: Module,
    previous_global_params: Sequence[Tensor],
    scale: float,
    global_round: int,
    local_iterations: int,
    layer_index: int,
) -> Tensor:
    """Overwrite ``local_model`` with forged weights and return its WEF-matrix.

    Parameters
    ----------
    attack:
        One of :data:`SUPPORTED_ATTACKS`.
    local_model:
        A private copy of the global model, modified in place.
    global_model:
        The model broadcast this round; read only.
    previous_global_params:
        Detached parameters of the model broadcast in the previous round,
        in ``model.parameters()`` order. Empty when the attacker has not
        yet received two global models.
    scale:
        Bound R for RWA, standard deviation sigma otherwise.
    global_round:
        Zero-based index of the current global communication round; only
        SPA uses it.
    local_iterations:
        Number of local iterations an honest client would perform.
    layer_index:
        Index of the parameter tensor the WEF-matrix is built from.

    Raises
    ------
    ValueError
        If ``attack`` is not supported.
    """
    if attack in ONE_SHOT_ATTACKS:
        return _forge_one_shot(
            attack, local_model, global_model, previous_global_params,
            scale, global_round, local_iterations, layer_index,
        )
    if attack in ITERATIVE_ATTACKS:
        return _forge_awca(
            local_model, global_model, previous_global_params,
            scale, local_iterations, layer_index,
        )
    raise ValueError(
        f"unsupported attack {attack!r}; expected one of {SUPPORTED_ATTACKS}"
    )


# ----------------------------------------------------------------------
# One-shot attacks
# ----------------------------------------------------------------------
def _forge_one_shot(
    attack: str,
    local_model: Module,
    global_model: Module,
    previous_global_params: Sequence[Tensor],
    scale: float,
    global_round: int,
    local_iterations: int,
    layer_index: int,
) -> Tensor:
    matrix = wef.empty_matrix(local_model, layer_index)
    before = copy.deepcopy(local_model)

    if attack == "RWA":
        _apply_random_weights(local_model, scale)
    elif attack == "SPA":
        _apply_stochastic_perturbation(
            local_model, previous_global_params, global_round
        )
    elif attack == "DWA":
        _apply_delta_weights(
            local_model, global_model, previous_global_params, noise_scale=None
        )
    elif attack == "ADWA":
        _apply_delta_weights(
            local_model, global_model, previous_global_params, noise_scale=scale
        )

    matrix = wef.count_evolved_weights(matrix, local_model, before, layer_index)
    # A single comparison can only produce zeros and ones, so the counts are
    # rescaled to the range an honest client would report (equation (2)).
    return matrix * local_iterations


def _apply_random_weights(local_model: Module, bound: float) -> None:
    """Resample every trainable weight from ``U(-bound, bound)``."""
    for parameter in local_model.parameters():
        if parameter.requires_grad:
            init.uniform_(parameter, a=-bound, b=bound)


def _apply_stochastic_perturbation(
    local_model: Module,
    previous_global_params: Sequence[Tensor],
    global_round: int,
) -> None:
    """Add Gaussian noise whose scale is estimated from past global models.

    The attacker fits a normal distribution to the element-wise difference
    between the two most recent global models and uses its standard
    deviation, damped by the round index, as the noise amplitude. No local
    data is required.
    """
    if len(previous_global_params) == 0:
        estimated_scale = SPA_COLD_START_SCALE
    else:
        differences = []
        for parameter, previous in zip(local_model.parameters(),
                                       previous_global_params):
            if parameter.requires_grad:
                delta = (parameter.data - previous).flatten().detach().cpu().numpy()
                differences.extend(delta)
        _, estimated_scale = norm.fit(differences)

    amplitude = estimated_scale * ((global_round + 1) ** SPA_DECAY_EXPONENT)
    for parameter in local_model.parameters():
        if parameter.requires_grad:
            parameter.data.add_(torch.randn_like(parameter) * amplitude)


def _apply_delta_weights(
    local_model: Module,
    global_model: Module,
    previous_global_params: Sequence[Tensor],
    noise_scale,
) -> None:
    """Replay the difference between the last two global models.

    ``noise_scale`` of ``None`` gives plain DWA; a float gives ADWA, which
    perturbs the replayed difference so that colluding attackers do not
    submit byte-identical updates.
    """
    if len(previous_global_params) == 0:
        # Cold start: the attacker has seen only one global model, so there
        # is no difference to replay. With the attack starting in round 3 or
        # later this branch is not reached in the reported experiments.
        for parameter, current in zip(local_model.parameters(),
                                      global_model.parameters()):
            if parameter.requires_grad:
                parameter.data.add_(current)
                if noise_scale is not None:
                    parameter.data.add_(torch.randn_like(parameter) * noise_scale)
        return

    for parameter, current, previous in zip(local_model.parameters(),
                                            global_model.parameters(),
                                            previous_global_params):
        if parameter.requires_grad:
            parameter.data.add_(current - previous)
            if noise_scale is not None:
                parameter.data.add_(torch.randn_like(parameter) * noise_scale)


# ----------------------------------------------------------------------
# Adaptive WEF-camouflage attack
# ----------------------------------------------------------------------
def _forge_awca(
    local_model: Module,
    global_model: Module,
    previous_global_params: Sequence[Tensor],
    noise_scale: float,
    local_iterations: int,
    layer_index: int,
) -> Tensor:
    """Forge weights one local iteration at a time (paper Algorithm 1).

    Each step advances the weights by ``(w_g^T - w_g^(T-1)) / e`` plus
    Gaussian noise and updates the WEF-matrix, so the counterfeit matrix
    accumulates counts the way an honest client's would.
    """
    matrix = wef.empty_matrix(local_model, layer_index)
    step = 1.0 / local_iterations
    cold_start = len(previous_global_params) == 0

    for _ in range(local_iterations):
        before = copy.deepcopy(local_model)

        if cold_start:
            for parameter, current in zip(local_model.parameters(),
                                          global_model.parameters()):
                if parameter.requires_grad:
                    parameter.data.add_(current * step)
                    parameter.data.add_(torch.randn_like(parameter) * noise_scale)
        else:
            for parameter, current, previous in zip(local_model.parameters(),
                                                    global_model.parameters(),
                                                    previous_global_params):
                if parameter.requires_grad:
                    parameter.data.add_((current - previous) * step)
                    parameter.data.add_(torch.randn_like(parameter) * noise_scale)

        matrix = wef.count_evolved_weights(matrix, local_model, before, layer_index)

    return matrix