# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2026 - Fujitsu Limited
# All rights reserved.
import csv
import os
import datetime
import fcntl

#フリーライダー検知精度をcsvファイルに書き込む
def save_acc(accuracy, precision, recall, f1, global_acc, fpr, conf, time):
    header = list(conf.keys())
    header.extend(["accuracy", "precision", "recall", "f1", "global_acc", "fpr", "comp_time"])
    values = list(conf.values())
    values.extend([accuracy, precision, recall, f1, global_acc, fpr, time])

    dt_now = datetime.datetime.now().strftime('%Y_%m%d')
    file_path = "result/{}.csv".format(dt_now)
    os.makedirs("result", exist_ok=True)

    with open(file_path, 'a', newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)   # 排他ロック
        f.seek(0, os.SEEK_END)
        write_header = f.tell() == 0             # ロック取得後に判定
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(values)
        f.flush()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)