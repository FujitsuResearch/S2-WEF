# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2026 - Fujitsu Limited
# All rights reserved.
#コマンドラインの引数を扱いやすくするモジュール
import argparse, json
import datetime
import os
import statistics
import math
#ログ出力のためのモジュール
import logging
#PhTorchのインポート
import torch,random
import time
import copy
from s2wef.aggregator import *
from s2wef.datasets import *
from s2wef.visualization import *
from s2wef.detector import *
from s2wef.metrics import *
from s2wef.clustering   import *
from s2wef.participant import Participant
from s2wef.pipeline import detect, select_decision_flags
from s2wef.partition import partition_data
#GPUが使用できるならばCUDA、できないならcpuをdeviceに代入
device = "cuda" if torch.cuda.is_available() else "cpu"
import matplotlib.pyplot as plt


def format_time(seconds):
    hours = int(seconds // 3600)           # 時間
    minutes = int((seconds % 3600) // 60)  # 分
    secs = seconds % 60                     # 秒（小数点以下も含む）
    return f"{hours}:{minutes}:{secs:.2f}"



if __name__ == '__main__':


    

    start = time.perf_counter()
    parser = argparse.ArgumentParser(description='Federated Learning')#parserを作る
    parser.add_argument('-c', '--conf',default="./s2wef/conf.json" ,dest='conf')
    
    args = parser.parse_args()#引数を解析
    #ファイルを読み込みモードで開く
    with open(args.conf, 'r') as f:
        conf = json.load(f)#json文字列をpythonオフジェクトに変換
    
    #ランダムシードの設定
    random_seed = conf.get("random_seed", 1234)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    random_state = random_seed
    torch.backends.cudnn.deterministic = True # PyTorchの畳み込み演算の再現性を確保
    torch.backends.cudnn.benchmark = False #畳み込みの計算のためのアルゴリズムの最適化と高速化をしない（再現性を保つため
    
    #conf["dataset"]で指定されたデータセットを得る
    data = load_datasets("./data/", conf["dataset"], device)
    train_datasets_all = data.train
    server = Aggregator(conf, data.evaluation, data.holdout)
    clients = []

    verbose = conf.get("verbose", "yes") == "yes"

    #フリーライダーを含むクライアント数（この値を分割数とクライアント生成の両方で使う）
    n_clients = conf["k"] + conf["free_riders"]
   
    if conf["no_iid"] == "yes":
        data_indices = partition_data(conf, train_datasets_all, n_clients, conf["beta"])
    else:  # IID
        data_indices = class_index_map(train_datasets_all, conf["dataset"])

    for c in range(n_clients):
        clients.append(Participant(conf, server.global_model, train_datasets_all,
                                   data_indices, data.holdout, c))
    print("\n\n")

    local_acc_final = []
    server_acc_final = []
    server_acc_2_final = []
    global_model_list = []
    candidates = []
    for j in range(conf["k"]+conf["free_riders"]):
        candidates.append(clients[j])

    #フリーライダー検知率を取得する配列
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    #global_F = 0
    pre_global_model = []
    #F_global_pre  = []
    if conf["free_riders_state"] == "dynamic_random":
        stolen_models = np.zeros((conf["free_riders"]+conf["k"],conf["global_epochs"]))
    else:
        stolen_models = np.zeros((conf["free_riders"],conf["global_epochs"]))
    pre_detect_result = []
    #各ラウンドで、どのクライアントがフリーライダーになるかを決める配列
    learning_state = np.zeros((conf["global_epochs"], conf["k"]+conf["free_riders"]))

    if conf["free_riders_state"] == "static":#全ラウンドでフリーライダーは固定
        for i in range(conf["free_riders"]):
            learning_state[:,i + conf["k"]] = 1
    elif conf["free_riders_state"] == "dynamic":#学習途中で良性クライアントがフリーライダーに変化
        freerider_start_round = conf["free_riders_start_round"] - 1
        for i in range(freerider_start_round, conf["global_epochs"]):
            for j in range(conf["free_riders"]):
                learning_state[i, j + conf["k"]] = 1
    elif conf["free_riders_state"] =="dynamic_random":#各ラウンドでランダムにフリーライダーが変化
        attack_start_round = conf.get("attack_start_round",conf["free_riders_start_round"]) - 1
        for i in range(attack_start_round, conf["global_epochs"]):
            fr_positions = random.sample(range(conf["k"]+conf["free_riders"]),conf["free_riders"])
            learning_state[i, fr_positions] = 1
    elif conf["free_riders_state"] =="NoFreeRider":
        pass
    
    print("FreeRidersIndex_in_all_round")
    print(learning_state)

    detect_times = []          # 検知処理時間（全体）
    detect_times_sim = []      # 内訳: サーバ側 WEF シミュレーション
    detect_times_score = []    # 内訳: γ / Dev / z スコア算出
    detect_times_clust = []    # 内訳: クラスタリング + 多数決
    for e in range(conf["global_epochs"]):#グローバルエポックの間繰り返す
        w_locals = []
        accepted_ids = []
        acc_all = []
        weight_all = []
        weight_all1 = []
        F_all = []#WEF-matrix
        F_delta_all = []
        diff_all=[]#モデルパラメータの更新
        local_acc_all = []#各クライアントのデータでの精度
        print("\nglobal_epoch:{0}/{1}".format(e+1,conf["global_epochs"]))
        

        for i,c in enumerate(candidates):
            #各クライアントに訓練させ、パラメータの更新とWEF-matrixを得る
            update = c.run_round(server.global_model, e, bool(learning_state[e, i]))
            F_all.append(update.wef_matrix)
            F_delta_all.append(update.wef_matrix)
            diff_all.append(update.delta)
            local_model = update.model
            #良性クライアントからは、ローカルモデルの検証結果を得る
            if conf["free_riders_state"] =="dynamic_random":
                acc = float(round(c.local_accuracy(update.model), 2))
                local_acc_all.append(acc)
            else:
                if i < conf["k"]:
                    acc = float(round(c.local_accuracy(update.model), 2))
                    local_acc_all.append(acc)

            #必要な場合、WEF-matrixを可視化し保存
            if conf["saveJPG"] == "yes":
                f_savejpg(update.wef_matrix,r'./result/WEF_img/global_e{0}_client_k{1}.jpg'.format(str(e+1),str(i+1)))

        detection = detect(
            conf=conf,
            F_all=F_all,
            F_delta_all=F_delta_all,
            global_model=server.global_model,
            pre_global_model=pre_global_model,
            random_seed=random_seed,
            round_index=e,
            verbose=verbose,
        )

        detect_times.append(detection.seconds["total"])
        detect_times_sim.append(detection.seconds["sim"])
        detect_times_score.append(detection.seconds["score"])
        detect_times_clust.append(detection.seconds["clust"])
        print("[detect-time] round {0}: total={1:.6f}s (sim={2:.6f} score={3:.6f} clust={4:.6f})".format(
            e + 1, detection.seconds["total"], detection.seconds["sim"],
            detection.seconds["score"], detection.seconds["clust"]))

        #必要であれば，z(Dev)とz(gamma)を散布図にプロットして可視化
        if conf["savefig_dev_gamma"] == "yes":
            path = plot_dev_gamma_scatter(
                detection.Dev_all_z, detection.CosWithG_z,
                detection.anomaly_labels, detection.fr_flags,
                learning_state[e], e + 1, show_confusion_box=True)
            print("Saved:", path)

        decision_flags = select_decision_flags(conf, detection)

        #フリーライダーを判定
        Detect_result = []
        FreeRiders = []
        if conf["method"] == "proposed":
            for i in range(conf["k"]+conf["free_riders"]):
                if (decision_flags[i] == 0) and F_delta_all[i].max() != 0:
                    Detect_result.append(0)
                    accepted_ids.append(i)
                    if learning_state[e,i] == 0:
                        tn += 1
                    else:
                        fn += 1
                else:
                    Detect_result.append(1)
                    FreeRiders.append(i + 1)
                    if learning_state[e,i] == 1 or F_delta_all[i].max() == 0:
                        if e < conf["detect_start_round"] - 1:
                            pass
                        else:
                            tp += 1
                    else:
                        fp += 1
        elif conf["method"] == "no_defense":
            for i in range(conf["k"]+conf["free_riders"]):
                Detect_result.append(0)
                accepted_ids.append(i)
        if conf["method"] == "proposed":
            print("[proposed method]FreeRideris:{}".format(FreeRiders))
        
        fr_true = np.where(learning_state[e] == 1)[0]

        print("[true_label]FreeRideris:{}".format(fr_true + 1))

        f_number = Detect_result.count(1)

        pre_global_model = copy.deepcopy(server.global_model)
        pre_detect_result = Detect_result
        n_accepted = server.aggregate_from_deltas([diff_all[i] for i in accepted_ids])
        acc_server, loss = server.evaluate(device)

        if conf["free_riders"] > 0:
            for i in range(conf["free_riders"]):
                if e == 0:
                    stolen_models[i,e] = acc_server
                elif (Detect_result[ conf["k"]+ i ] == 0) and (learning_state[e,conf["k"]+ i] == 1):
                    stolen_models[i,e] = acc_server
                else:
                    pass

        print("local_acc:{}".format(local_acc_all))
        print("global_acc:{}".format(round(acc_server,2)))
    if len(detect_times) > 0:
        N_c = conf["k"] + conf["free_riders"]
        dt = np.array(detect_times, dtype=float)
        dt2 = dt[1:] if len(dt) > 1 else dt   # 1ラウンド目はウォームアップのため除外
        print("\n[detect-time] N={0}, rounds={1}".format(N_c, len(dt)))
        print("[detect-time] mean={0:.6f}s  std={1:.6f}s  median={2:.6f}s".format(
            dt.mean(), dt.std(ddof=1) if len(dt) > 1 else 0.0, np.median(dt)))
        print("[detect-time] mean(excl. 1st round)={0:.6f}s".format(dt2.mean()))
        print("[detect-time] breakdown mean: sim={0:.6f}s score={1:.6f}s clust={2:.6f}s".format(
            np.mean(detect_times_sim), np.mean(detect_times_score),
            np.mean(detect_times_clust)))
        os.makedirs("./result", exist_ok=True)
        run_tag = os.environ.get("FL_RUN_TAG", "")
        suffix = ("_" + run_tag) if run_tag else ""
        #np.savetxt("./result/detect_time_{0}_N{1}_seed{2}{3}.csv".format(
        #    conf["dataset"], N_c, conf["random_seed"], suffix),
        #    np.column_stack([dt, detect_times_sim, detect_times_score, detect_times_clust]),
        #    delimiter=",", header="total,sim,score,clust", comments="")
        
    # if conf["free_riders"] > 0:
    #     stolen_models_acc = np.max(stolen_models, axis=1)
    #     print("stolen_models_acc:{}".format(stolen_models_acc))
    #     stolen_models_acc_mean = np.mean(stolen_models_acc)
    #     print("stolem_models_acc_mean:{}".format(stolen_models_acc_mean))

    #フリーライダー検知精度を表示
    if tp + fp + fn + tn == 0:
        accuracy = 0
    else:
        accuracy = (tp + tn )/ (tp + fp + fn + tn)
    if tp + fp == 0:
        precision = 0
    else:
        precision = (tp)/(tp + fp)
    if tp + fn == 0:
        recall = 0
    else:
        recall = (tp)/(tp + fn)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    if tn + fp == 0:
        fpr = 0
    else:
        fpr = (fp)/ (tn + fp)
    print("\nfreerider_detection_accuracy:{0}/precision:{1}/recall:{2}/f1:{3}/fpr:{4}".format(accuracy,precision,recall,f1,fpr))
    end = time.perf_counter()
    elapsed = end - start
    print("実行時間:", format_time(elapsed))
    #結果をcsvファイルに書き込み
    save_acc(accuracy, precision, recall,f1,acc_server,fpr,conf,format_time(elapsed))

    
