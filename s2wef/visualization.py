# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2026 - Fujitsu Limited
# All rights reserved.
import os
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from typing import (
    List,
    Optional
)

#WEF-matrixを保存
def f_savejpg(F, fname):
    directory = os.path.dirname(fname)
    if directory:
        os.makedirs(directory, exist_ok=True)
    plt.figure()
    im = plt.imshow(F)
    plt.colorbar(im)
    plt.savefig(fname)
    plt.close()

#リストを見やすく表示
def printlist(list,name, verbose=True):
    if not verbose:
        return
    list = [round(x,3) for x in list]
    list = [float(x) for x in list]
    print("{0}:{1}".format(name,list))

def plot_dev_gamma_scatter(
    z_dev: List[float],
    z_gamma: List[float],
    cluster_labels: List[int],
    fr_flags: List[int],       # 予測（ゲート後の判定）：FR=1, 良性=0
    true_labels: List[int],    # 真のラベル：FR=1, 良性=0
    epoch: int,
    out_dir: Optional[str] = None,
    dpi: int = 150,
    show_confusion_box: bool = True,
) -> str:
    """
    ロバストz標準化済みの Dev/γ を 2次元散布図で可視化し、JPG 保存する。
    色＝クラスタ、形＝予測（良性=○, FR=✕）、縁色＝真のラベル（良性=緑, FR=赤）。

    Parameters
    ----------
    z_dev : List[float]
        各クライアントの標準化された Dev（ロバスト z）
    z_gamma : List[float]
        各クライアントの標準化された γ（ロバスト z）
    cluster_labels : List[int]
        クラスタリング結果（K=1 のとき全て 0、K=2→{0,1}、K=3→{0,1,2}）。
        0=原点に最も近い, 1=最も遠い, 2=次に遠い（前段の実装方針に準拠）。
    fr_flags : List[int]
        予測ラベル（ゲート判定結果）：1=フリーライダー, 0=良性
    true_labels : List[int]
        真のラベル：1=フリーライダー, 0=良性
    epoch : int
        グローバルエポック（ファイル名に含める）
    out_dir : Optional[str]
        出力ディレクトリ。省略時は "./result/image"
    dpi : int
        保存時の解像度（dpi）
    show_confusion_box : bool
        図内に簡易混同行列（TP/FP/FN/TN）を表示するかどうか

    Returns
    -------
    saved_path : str
        保存した JPG ファイルのフルパス
    """
    # ---- 入力チェック ----
    if not (len(z_dev) == len(z_gamma) == len(cluster_labels) == len(fr_flags) == len(true_labels)):
        raise ValueError("z_dev, z_gamma, cluster_labels, fr_flags, true_labels の長さが一致していません。")

    x = np.asarray(z_dev, dtype=float)       # 横軸：Dev
    y = np.asarray(z_gamma, dtype=float)     # 縦軸：γ
    c = np.asarray(cluster_labels, dtype=int)
    y_pred = np.asarray(fr_flags, dtype=int)     # 予測
    y_true = np.asarray(true_labels, dtype=int)  # 真値

    n = len(x)
    if n == 0:
        raise ValueError("入力が空です。描画できません。")

    # ---- 出力先ディレクトリ ----
    if out_dir is None:
        out_dir = os.path.join(".", "result", "image")
    os.makedirs(out_dir, exist_ok=True)

    # ---- 色マップ（クラスタ毎） ----
    unique_clusters = sorted(np.unique(c))   # 例: [0], [0,1], [0,1,2]
    cmap = plt.get_cmap("tab10")
    color_map = {lab: cmap(i % 10) for i, lab in enumerate(unique_clusters)}

    # ---- 真値に応じた縁色 ----
    edge_color_true = np.where(y_true == 1, "#d62728", "#2ca02c")  # FR=赤, 良性=緑

    # ---- 描画 ----
    fig, ax = plt.subplots(figsize=(7.2, 6))

    # クラスタ別に、さらに「予測=良性/FR」で分けて重ね描き
    # 予測: 良性= 'o'（丸）, FR= 'X'（クロス）
    for lab in unique_clusters:
        in_cluster = (c == lab)

        # 良性予測（○）
        mask_benign = in_cluster & (y_pred == 0)
        if np.any(mask_benign):
            ax.scatter(
                x[mask_benign], y[mask_benign],
                s=60, c=[color_map[lab]], marker='o',
                edgecolors=edge_color_true[mask_benign], linewidths=1.2,
                alpha=0.9, zorder=2
            )

        # FR予測（✕）
        mask_fr = in_cluster & (y_pred == 1)
        if np.any(mask_fr):
            ax.scatter(
                x[mask_fr], y[mask_fr],
                s=90, c=[color_map[lab]], marker='X',
                edgecolors=edge_color_true[mask_fr], linewidths=1.2,
                alpha=0.95, zorder=3
            )

    # ---- 原点補助線・軸ラベル・タイトル ----
    ax.axhline(0.0, color="#888888", lw=1, ls="--", alpha=0.7, zorder=1)
    ax.axvline(0.0, color="#888888", lw=1, ls="--", alpha=0.7, zorder=1)

    ax.set_xlabel("robust z(Dev)", fontsize=12)
    ax.set_ylabel("robust z(gamma)", fontsize=12)
    ax.set_title(f"2D Plot of z(Dev)-z(gamma) E={epoch}", fontsize=13)

    # ---- 枠のゆとり（外れ値があっても窮屈にならないように） ----
    def _pad_lim(arr, pad_ratio=0.08):
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return (-1.0, 1.0)
        if vmin == vmax:
            return (vmin - 1.0, vmax + 1.0)
        span = vmax - vmin
        pad = span * pad_ratio
        return (vmin - pad, vmax + pad)

    ax.set_xlim(*_pad_lim(x))
    ax.set_ylim(*_pad_lim(y))
    ax.grid(True, ls=":", lw=0.8, alpha=0.6)

    # ---- 凡例：色（クラスタ） ----
    cluster_handles = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=color_map[lab], markeredgecolor='#cccccc',
               markeredgewidth=0.8, markersize=7,
               label=f"cluster {lab}")
        for lab in unique_clusters
    ]

    # ---- 凡例：形（予測） ----
    pred_handles = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor="#666666", markeredgecolor='#cccccc',
               markeredgewidth=0.8, markersize=7, label="pred: benign"),
        Line2D([0], [0], marker='X', color='none',
               markerfacecolor="#666666", markeredgecolor='#333333',
               markeredgewidth=0.8, markersize=7, label="pred: FR"),
    ]

    # ---- 凡例：縁色（真値） ----
    truth_handles = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor='white', markeredgecolor='#2ca02c',  # 緑
               markeredgewidth=1.6, markersize=7, label="true: benign"),
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor='white', markeredgecolor='#d62728',  # 赤
               markeredgewidth=1.6, markersize=7, label="true: FR"),
    ]

    # 凡例は 3 段構成（各図の外側に）
    leg1 = ax.legend(handles=cluster_handles, title="cluster(color)",
                     loc="upper left", bbox_to_anchor=(1.02, 1.00))
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=pred_handles, title="pred(shape)",
                     loc="upper left", bbox_to_anchor=(1.02, 0.74))
    ax.add_artist(leg2)
    ax.legend(handles=truth_handles, title="true(green)",
              loc="upper left", bbox_to_anchor=(1.02, 0.48))

    # ---- 簡易混同行列の表示（任意） ----
    if show_confusion_box:
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        txt = f"TP={tp}  FP={fp}\nFN={fn}  TN={tn}"
        ax.text(0.02, 0.02, txt, transform=ax.transAxes,
                fontsize=10, va='bottom', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, ec='#666666'))

    fig.tight_layout()

    # ---- 保存 ----
    filename = f"dev_gamma_scatter_E{epoch}.jpg"
    save_path = os.path.join(out_dir, filename)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return save_path