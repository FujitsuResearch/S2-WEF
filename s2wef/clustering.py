# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2026 - Fujitsu Limited
# All rights reserved.
import numpy as np

from scipy.cluster.hierarchy import (
    linkage,
    fcluster
)

from sklearn.metrics import (
    silhouette_score
)

from typing import List

def calculate_statistical_z(values, eps: float = 1e-12):
    """
    Robust z-score using median and MAD.
    z = (x - median(x)) / (1.4826 * MAD(x))

    Parameters
    ----------
    values : Sequence[float] | np.ndarray
        1次元の数列。NaN は無視せず、そのままの位置で NaN を返す動作。
    eps : float
        数値安定化用の微小値。MAD==0 の場合でもゼロ除算を防ぐ。

    Returns
    -------
    list[float]
        入力と同じ長さの z 値（リスト）
    """
    x = np.asarray(values, dtype=float)
    med = np.nanmedian(x)  # NaN を含む場合は NaN を無視して中央値を計算
    mad = np.nanmedian(np.abs(x - med))

    scale = 1.4826 * mad  # 正規分布に対する MAD -> 標準偏差の補正係数
    if not np.isfinite(scale) or scale < eps:
        scale = eps  # 極端に分散が小さい/ゼロの場合でも発散させない

    z = (x - med) / scale
    # 元の関数の戻りに合わせて list に変換
    return z.tolist()

def HAC_2d_auto(list1, list2, random_state=None,
                min_cluster_frac=0.1,  # 少数群の最小割合（サイズ制約）
                s_min=0.30,            # 2群時の平均シルエット下限
                jump_tau=0.9):         # 最終マージ跳躍比の下限
    """
    Ward 法で階層を作り、K=1/2 を自動決定:
      - 2群シルエットS < s_min → 切らない(K=1)
      - 最終マージの距離跳躍比 Δ < jump_tau → 切らない(K=1)
      - 2群に切った後、少数群サイズ < m_min → 切らない(K=1)
    K=2 のときは原点から遠い重心側を異常=1として返す。
    """
    data = np.column_stack((list1, list2)).astype(float)
    n = data.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [0]

    # 標準化（任意）：robustにしたい場合は中央値/MADに切替え可
    # mean = data.mean(axis=0, keepdims=True)
    # std  = data.std(axis=0, keepdims=True) + 1e-12
    # Zdata = (data - mean) / std
    Zdata = data

    # Ward 法でリンク
    Z = linkage(Zdata, method='ward', metric='euclidean')

    # ---- K=2 の暫定割当 ----
    labels2 = fcluster(Z, t=2, criterion='maxclust') - 1  # {0,1}
    n0 = np.sum(labels2 == 0)
    n1 = n - n0

    # 2群の平均シルエット（n<3や単一点クラスタだと計算不能→低スコア扱い）
    S = -1.0
    if min(n0, n1) >= 1:
        try:
            S = silhouette_score(Zdata, labels2, metric='euclidean')
        except Exception:
            S = -1.0

    # 最終マージの距離跳躍比 Δ = h_last / h_prev
    heights = Z[:, 2]  # 各マージの距離
    if len(heights) >= 2:
        delta = heights[-1] / max(heights[-2], 1e-12)
    else:
        delta = 0.0

    # 少数群サイズ制約
    m_min = max(1, int(np.ceil(min_cluster_frac * n)))

    # ---- 切る/切らない判定 ----
    cut = True
    if (S < s_min) or (delta < jump_tau) or (min(n0, n1) < m_min):
        cut = False

    if not cut:
        # 切らない: 全員ベニン
        return [0] * n

    # ---- K=2 採用: 原点から遠い重心側を異常に ----
    centers = np.vstack([Zdata[labels2 == k].mean(axis=0) for k in (0, 1)])
    anomaly_cluster = int(np.argmax(np.linalg.norm(centers, axis=1)))
    anomaly_labels = [1 if lbl == anomaly_cluster else 0 for lbl in labels2]
    return anomaly_labels
