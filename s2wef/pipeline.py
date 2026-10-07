# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2026 - Fujitsu Limited
# All rights reserved.
"""S2-WEF detection pipeline, executed once per global communication round.

This module contains the server-side detection procedure of our paper:

1. simulate the WEF-matrix a global-model-mimicking attacker would produce
   from the last two broadcast global models;
2. score every submitted WEF-matrix against the simulated one, and against
   the other submissions;
3. cluster the two scores and apply per-score thresholds;
4. combine both by a vote inside the suspicious cluster.

"""

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from s2wef import wef
from s2wef.clustering import HAC_2d_auto, calculate_statistical_z
from s2wef.detector import decide_free_riders_with_gate_c3
from s2wef.visualization import f_savejpg, printlist


@dataclass
class DetectionResult:
    """Everything one round of detection produces.

    Attributes
    ----------
    fr_flags:
        Final per-client flags of the full pipeline: 1 for free-rider.
    anomaly_labels:
        Clustering outcome before the vote.
    Dev_labels, Gamma_labels:
        Per-score threshold decisions on Dev and gamma.
    Dev_all, Gamma_all:
        Raw deviation and similarity scores.
    Dev_all_z, Gamma_all_z:
        The same scores after robust standardization, as clustered.
    F_global:
        The WEF-matrix simulated on the server this round.
    F_delta_all:
        Submitted WEF-matrices after the one-step rescaling; the caller's
        list is rescaled in place as well.
    seconds:
        Wall-clock cost of the stages, keyed ``sim``, ``score``, ``clust``
        and ``total``.
    """

    fr_flags: List[int]
    anomaly_labels: List[int]
    Dev_labels: List[int]
    Gamma_labels: List[int]
    Dev_all: List[float]
    Gamma_all: List[float]
    Dev_all_z: List[float]
    Gamma_all_z: List[float]
    F_global: np.ndarray
    F_delta_all: List[np.ndarray]
    seconds: Dict[str, float] = field(default_factory=dict)


def detect(
    conf: Dict[str, Any],
    F_all: Sequence[np.ndarray],
    F_delta_all: List[np.ndarray],
    global_model,
    pre_global_model,
    random_seed: int,
    round_index: int,
    verbose: bool = True,
) -> DetectionResult:
    """Run one round of S2-WEF detection.

    Parameters
    ----------
    conf:
        Experiment configuration.
    F_all:
        WEF-matrix submitted by each client, in client order.
    F_delta_all:
        The same matrices, used for the similarity score. Entries whose
        maximum is 1 are rescaled by the number of local iterations, in
        place.
    global_model:
        The model broadcast this round.
    pre_global_model:
        The model broadcast in the previous round, or ``[]`` in the first
        round.
    random_seed:
        Passed to the clustering routine.
    round_index:
        Zero-based round index; used only for log and figure names.
    verbose:
        Whether the per-client score lists are printed.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_detect_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Server-side simulation of the attacker's WEF-matrix
    # ------------------------------------------------------------------
    #グローバルモデル側で攻撃時のWEF-matrixを算出し、これに似ているかを確かめる
    WEF_layer = conf["WEF-layer"]
    F = wef.empty_matrix(global_model, WEF_layer)
    global_model_copy = copy.deepcopy(global_model)
    global_model_d = copy.deepcopy(global_model)
    for local_epoch in range(conf["local_epochs"]):
        #ローカルエポックの間繰り返す
        model_d = copy.deepcopy(global_model_d)
        alpha = 1 / conf["local_epochs"]
        #前のグローバルモデルと、今のグローバルモデルの差分を今のグローバルモデルに加算
        if pre_global_model == []:
            for param, param_m in zip(global_model_d.parameters(),
                                      global_model_copy.parameters()):
                if param.requires_grad:
                    param.data.add_(param_m * alpha)
        else:
            for param, param_m, param_p in zip(global_model_d.parameters(),
                                               global_model_copy.parameters(),
                                               pre_global_model.parameters()):
                if param.requires_grad:
                    param.data.add_((param_m - param_p) * alpha)
        #パラメータ更新前後で、変化の平均と変化量の絶対値を取得
        mean_change, elementwise_change = wef.measure_movement(
            global_model_d, model_d, WEF_layer
        )
        F = F + (elementwise_change > mean_change).to(dtype=F.dtype, device=F.device)
    F_global = F.detach().numpy().copy()

    if conf["saveJPG"] == "yes":
        f_savejpg(
            F_global,
            r'./result/WEF_img/global_e{}_globalsimulation.jpg'.format(
                str(round_index + 1)
            ),
        )

    for i in range(conf["k"] + conf["free_riders"]):
        if F_delta_all[i].max() == 1:
            #ローカルエポックの数にスケーリング
            F_delta_all[i] = F_delta_all[i] * conf["local_epochs"]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_sim_end = time.perf_counter()

    # ------------------------------------------------------------------
    # 2a. Similarity to the simulated WEF-matrix
    # ------------------------------------------------------------------
    #中央サーバで作成した攻撃時のWEF-matrixとのコサイン類似度を算出
    CosWithGlobalF_all = []
    L1withGlobalF_all = []
    L2withGlobalF_all = []
    for i in range(conf["k"] + conf["free_riders"]):
        if F_delta_all[i].max() == 0:
            c = 0
        else:
            norm_i = np.linalg.norm(F_delta_all[i])
            norm_j = np.linalg.norm(F_global)
            if norm_i == 0 or norm_j == 0:
                c = 0
                c = float(c)
            else:
                c = np.dot(F_delta_all[i].flatten(),
                           F_global.flatten()) / (norm_i * norm_j)
                c = float(c)
        CosWithGlobalF_all.append(c)
        l1 = np.sum(np.abs(F_delta_all[i] - F_global))
        l2 = (np.linalg.norm(F_delta_all[i] - F_global)) ** 2
        if l1 == 0:#0だと除算時にエラーになる
            l1 = 0.001
        l1 = float(l1)
        if l2 == 0:#0だと除算時にエラーになる
            l2 = 0.001
        l2 = float(l2)
        L1withGlobalF_all.append(l1)
        L2withGlobalF_all.append(l2)
    printlist(CosWithGlobalF_all, "Cos(SimulatedF,F_i)", verbose)
    printlist(L2withGlobalF_all, "L2(SimulatedF,F_i)", verbose)

    # ------------------------------------------------------------------
    # 2b. Mutual comparison among the submitted WEF-matrices
    # ------------------------------------------------------------------
    #WEF-matrix間のユークリッド距離・コサイン類似度、またWEF-matrixの平均値を算出
    Dis_all = []#ユークリッド距離
    Cos_all = []#コサイン類似度
    Avg_all = []#平均値
    for i in range(conf["k"] + conf["free_riders"]):
        d = 0
        c = 0
        m = 0
        for j in range(conf["k"] + conf["free_riders"]):
            if i == j:
                pass
            else:
                #WEF-matrix間のユークリッド距離の合計を算出
                d = d + np.sum((F_all[i] - F_all[j]) ** 2)
                #マンハッタン距離の合計を算出
                m = m + np.sum(np.abs(F_all[i] - F_all[j]))
                #WEF-matrix間のコサイン類似度の合計を算出
                norm_i = np.linalg.norm(F_all[i])
                norm_j = np.linalg.norm(F_all[j])
                if norm_i == 0 or norm_j == 0:
                    pass
                else:
                    c = c + np.dot(F_all[i].flatten(),
                                   F_all[j].flatten()) / (norm_i * norm_j)
        Dis = np.sqrt(float(d))
        Dis_all.append(Dis)
        Cos_all.append(float(c / (conf["k"] + conf["free_riders"] - 1)))
        #WEF-matrixの平均値を算出
        Avg_all.append(float(np.mean(F_all[i])))

    # ------------------------------------------------------------------
    # 2c. Similarity score gamma and its threshold
    # ------------------------------------------------------------------
    if conf["ablation"] == "cos":
        Gamma_all = CosWithGlobalF_all#ablation_study
    elif conf["ablation"] == "cos/l1":
        Gamma_all = [x * 1000 / y for x, y
                          in zip(CosWithGlobalF_all, L1withGlobalF_all)]#ablation_study
    elif conf["ablation"] == "cos/l2":
        Gamma_all = [x * 1000 / y for x, y
                          in zip(CosWithGlobalF_all, L2withGlobalF_all)]
    else:
        Gamma_all = [x * 1000 / y for x, y
                          in zip(CosWithGlobalF_all, L2withGlobalF_all)]
    printlist(Gamma_all, "Gamma_i", verbose)

    gamma_values = np.asarray(Gamma_all, dtype=float)
    gamma_median = float(np.median(gamma_values))
    gamma_threshold_method = conf.get("gamma_threshold_method", "median_x1.5")
    if gamma_threshold_method == "median_plus_mad":
        gamma_threshold_k = float(conf.get("gamma_threshold_k", 1.5))
        gamma_mad = float(np.median(np.abs(gamma_values - gamma_median)))
        gamma_threshold = gamma_median + gamma_threshold_k * gamma_mad
        print("Gamma threshold (median + k*MAD, k={}):{}".format(
            gamma_threshold_k, gamma_threshold))
    elif gamma_threshold_method == "median_x1.5":
        gamma_threshold_multiplier = float(
            conf.get("gamma_threshold_multiplier", 1.5))
        gamma_threshold = gamma_threshold_multiplier * gamma_median
        print("Gamma threshold (multiplier*median, multiplier={}):{}".format(
            gamma_threshold_multiplier, gamma_threshold))
    else:
        raise ValueError("Unsupported gamma_threshold_method: {}".format(
            gamma_threshold_method))

    #γがしきい値より高いかでラベルをつける
    Gamma_labels = []
    for i in range(conf["k"] + conf["free_riders"]):
        if Gamma_all[i] < gamma_threshold:
            Gamma_labels.append(0)
        else:
            Gamma_labels.append(1)
    printlist(Gamma_labels, "Gamma_labels", verbose)
    printlist(Dis_all, "Dis_i", verbose)
    printlist(Cos_all, "Cos_i", verbose)
    printlist(Avg_all, "Avg_i", verbose)

    # ------------------------------------------------------------------
    # 2d. Deviation score Dev and its threshold
    # ------------------------------------------------------------------
    #偏差の絶対値の合計を算出
    sum_dis_d = 0
    sum_cos_d = 0
    sum_avg_d = 0
    for i in range(conf["k"] + conf["free_riders"]):
        sum_dis_d += abs(Dis_all[i] - np.mean(Dis_all))
        sum_cos_d += abs(Cos_all[i] - np.mean(Cos_all))
        sum_avg_d += abs(Avg_all[i] - np.mean(Avg_all))

    Dev_all = []#Dev
    for i in range(conf["k"] + conf["free_riders"]):
        dev_i = abs(Dis_all[i] - np.mean(Dis_all)) / sum_dis_d
        dev_i += abs(Cos_all[i] - np.mean(Cos_all)) / sum_cos_d
        dev_i += abs(Avg_all[i] - np.mean(Avg_all)) / sum_avg_d
        dev_i = float(dev_i)
        Dev_all.append(dev_i)
    printlist(Dev_all, "Dev_i", verbose)

    #Devがしきい値より高いか分類する
    N_clients = conf["k"] + conf["free_riders"]
    th = (conf.get("dev_threshold_eps", 0.05)
          * conf.get("dev_threshold_ref_n", 10) / N_clients)
    Dev_labels = []
    for i in range(conf["k"] + conf["free_riders"]):
        if Dev_all[i] < max(Dev_all) - th:
            Dev_labels.append(0)
        else:
            Dev_labels.append(1)
    printlist(Dev_labels, "Dev_label", verbose)

    Gamma_all_z = calculate_statistical_z(Gamma_all)
    printlist(Gamma_all_z, "Gamma_i_z", verbose)
    Dev_all_z = calculate_statistical_z(Dev_all)
    printlist(Dev_all_z, "Dev_i_z", verbose)

    t_score_end = time.perf_counter()

    # ------------------------------------------------------------------
    # 3-4. Clustering and the vote-based decision
    # ------------------------------------------------------------------
    anomaly_labels = HAC_2d_auto(
        Gamma_all_z, Dev_all_z, random_seed,
        min_cluster_frac=0.0, s_min=0.4, jump_tau=0.9,
    )
    printlist(anomaly_labels, "clustering_result", verbose)

    #クラスタリング結果と，Devとγの判定結果から，多数決でフリーライダーとみなすかを決定
    if conf["majority_vote"] == "yes":
        fr_flags, info = decide_free_riders_with_gate_c3(
            anomaly_labels, Dev_labels, Gamma_labels,
            majority_thr=0.5, strict=False, return_details=True,
        )
    else:
        fr_flags = anomaly_labels

    t_detect_end = time.perf_counter()

    return DetectionResult(
        fr_flags=fr_flags,
        anomaly_labels=anomaly_labels,
        Dev_labels=Dev_labels,
        Gamma_labels=Gamma_labels,
        Dev_all=Dev_all,
        Gamma_all=Gamma_all,
        Dev_all_z=Dev_all_z,
        Gamma_all_z=Gamma_all_z,
        F_global=F_global,
        F_delta_all=F_delta_all,
        seconds={
            "sim": t_sim_end - t_detect_start,
            "score": t_score_end - t_sim_end,
            "clust": t_detect_end - t_score_end,
            "total": t_detect_end - t_detect_start,
        },
    )


def select_decision_flags(conf: Dict[str, Any], result: DetectionResult) -> List[int]:
    """Pick which flags drive the final decision.

    The full pipeline is the method of the paper; the other options isolate
    one component and are used by the ablation and sensitivity studies.

    Raises
    ------
    ValueError
        If ``decision_source`` is not recognized.
    """
    decision_source = conf.get("decision_source", "pipeline")
    sources = {
        "pipeline": result.fr_flags,          # クラスタリング + 多数決（論文の S2-WEF）
        "gamma_only": result.Gamma_labels,    # γ のしきい値判定のみ
        "dev_only": result.Dev_labels,        # Dev のしきい値判定のみ
        "cluster_only": result.anomaly_labels,# クラスタリングのみ（多数決なし）
    }
    try:
        flags = sources[decision_source]
    except KeyError:
        raise ValueError(
            "Unsupported decision_source: {}".format(decision_source)
        ) from None
    print("[decision_source] {}".format(decision_source))
    return flags