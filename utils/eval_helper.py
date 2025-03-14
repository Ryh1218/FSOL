import math
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy import spatial as ss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count


class AverageCategoryMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, num_class):
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.cur_val = np.zeros(self.num_class)
        self.sum = np.zeros(self.num_class)

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val


def LMDS_counting(input, w_fname, floc, hthresh):
    input_max = torch.max(input).item()

    """ find local maxima"""
    keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input

    """set the pixel valur of local maxima as 1 for counting"""
    if hthresh:
        input[input < 60.0 / 255.0 * input_max] = 0
    else:
        input[input < 40.0 / 255.0 * input_max] = 0
    input[input > 0] = 1

    if hthresh:
        if input_max < 0.06:
            input = input * 0

    count = int(torch.sum(input).item())

    kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()

    floc.write("{} {} ".format(w_fname, count))
    return kpoint, floc


def generate_point_map(kpoint, floc, rate=1):
    """obtain the location coordinates"""
    pred_coor = np.nonzero(kpoint)

    point_map = (
        np.zeros(
            (int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8"
        )
        + 255
    )
    coord_list = []
    for i in range(0, len(pred_coor[0])):
        h = int(pred_coor[0][i] * rate)
        w = int(pred_coor[1][i] * rate)
        coord_list.append([w, h])
        cv2.circle(point_map, (w, h), 2, (0, 0, 0), -1)

    for data in coord_list:
        floc.write("{} {} ".format(math.floor(data[0]), math.floor(data[1])))
    floc.write("\n")


def Counting(pred, fname, floc, hthresh):
    kpoint, floc = LMDS_counting(pred, fname, floc, hthresh)
    generate_point_map(kpoint, floc, rate=1)
    return kpoint


def mysort(line):
    return line.split()[0]


def file_order(file_path, target_path):
    with open(file_path, "r") as f:
        text = f.readlines()

    f_loc = open(target_path, "w+")
    for line in sorted(text, key=mysort):
        f_loc.write(line)
    f_loc.close()


def hungarian(matrixTF):
    # matrix to adjacent matrix
    edges = np.argwhere(matrixTF)
    lnum, rnum = matrixTF.shape
    graph = [[] for _ in range(lnum)]
    for edge in edges:
        graph[edge[0]].append(edge[1])

    # deep first search
    match = [-1 for _ in range(rnum)]
    vis = [-1 for _ in range(rnum)]

    def dfs(u):
        for v in graph[u]:
            if vis[v]:
                continue
            vis[v] = True
            if match[v] == -1 or dfs(match[v]):
                match[v] = u
                return True
        return False

    ans = 0
    for a in range(lnum):
        for i in range(rnum):
            vis[i] = False
        if dfs(a):
            ans += 1

    assign = np.zeros((lnum, rnum), dtype=bool)
    for i, m in enumerate(match):
        if m >= 0:
            assign[m, i] = True

    return ans, assign


def read_pred_and_gt(pred_file, gt_file):
    pred_data = {}
    with open(pred_file) as f:
        id_read = []
        for line in f.readlines():
            line = line.strip().split(" ")

            if (
                len(line) < 2
                or len(line) % 2 != 0
                or (len(line) - 2) / 2 != int(line[1])
            ):
                sys.exit(1)

            line_data = [int(i) for i in line]
            idx, num = [line_data[0], line_data[1]]
            id_read.append(idx)

            points = []
            if num > 0:
                points = np.array(line_data[2:]).reshape(((len(line) - 2) // 2, 2))
                pred_data[idx] = {"num": num, "points": points}
            else:
                pred_data[idx] = {"num": num, "points": []}

    gt_data = {}
    cell_num = 0
    with open(gt_file) as f:
        for line in f.readlines():
            cell_num += 1
            line = line.strip().split(" ")

            line_data = [int(i) for i in line]
            idx, num = [line_data[0], line_data[1]]
            points_r = []
            if num > 0:
                points_r = np.array(line_data[2:]).reshape(((len(line) - 2) // 5, 5))
                gt_data[idx] = {
                    "num": num,
                    "points": points_r[:, 0:2],
                    "sigma": points_r[:, 2:4],
                    "level": points_r[:, 4],
                }
            else:
                gt_data[idx] = {"num": 0, "points": [], "sigma": [], "level": []}

    return pred_data, gt_data, cell_num


def read_pred_and_gt_carpk(pred_file, gt_file):
    # read pred
    pred_data = {}
    with open(pred_file) as f:
        for line in f.readlines():
            line = line.strip().split(" ")
            # check1

            if (
                len(line) < 2
                or len(line) % 2 != 0
                or (len(line) - 2) / 2 != int(line[1])
            ):
                sys.exit(1)

            line_data = [int(i) for i in line[1:]]
            line_data.insert(0, line[0])
            idx, num = [line_data[0], int(line_data[1])]

            points = []
            if num > 0:
                points = np.array(line_data[2:]).reshape(((len(line) - 2) // 2, 2))
                pred_data[idx] = {"num": num, "points": points}
            else:
                pred_data[idx] = {"num": num, "points": []}

    # read gt
    gt_data = {}
    cell_num = 0
    with open(gt_file) as f:
        for line in f.readlines():
            cell_num += 1
            line = line.strip().split(" ")

            line_data = [int(i) for i in line[1:]]
            line_data.insert(0, line[0])
            idx, num = [line_data[0], int(line_data[1])]

            points_r = []
            if num > 0:
                points_r = np.array(line_data[2:]).reshape(((len(line) - 2) // 5, 5))
                gt_data[idx] = {
                    "num": num,
                    "points": points_r[:, 0:2],
                    "sigma": points_r[:, 2:4],
                    "level": points_r[:, 4],
                }
            else:
                gt_data[idx] = {"num": 0, "points": [], "sigma": [], "level": []}

    return pred_data, gt_data, cell_num


def compute_metrics(
    dist_matrix, match_matrix, pred_num, gt_num, sigma, level, num_classes
):
    for i_pred_p in range(pred_num):
        pred_dist = dist_matrix[i_pred_p, :]
        match_matrix[i_pred_p, :] = pred_dist <= sigma

    tp, assign = hungarian(match_matrix)
    fn_gt_index = np.array(np.where(assign.sum(0) == 0))[0]
    tp_pred_index = np.array(np.where(assign.sum(1) == 1))[0]
    tp_gt_index = np.array(np.where(assign.sum(0) == 1))[0]
    fp_pred_index = np.array(np.where(assign.sum(1) == 0))[0]

    tp = tp_pred_index.shape[0]
    fp = fp_pred_index.shape[0]
    fn = fn_gt_index.shape[0]

    tp_c = np.zeros([num_classes])
    fn_c = np.zeros([num_classes])

    for i_class in range(num_classes):
        tp_c[i_class] = (level[tp_gt_index] == i_class).sum()
        fn_c[i_class] = (level[fn_gt_index] == i_class).sum()

    return tp, fp, fn, tp_c, fn_c


def location_main(pred_file, gt_file, carpk):
    num_classes = 1

    cnt_errors = {
        "mae": AverageMeter(),
        "mse": AverageMeter(),
        "nae": AverageMeter(),
    }
    metrics_s = {
        "tp": AverageMeter(),
        "fp": AverageMeter(),
        "fn": AverageMeter(),
        "tp_c": AverageCategoryMeter(num_classes),
        "fn_c": AverageCategoryMeter(num_classes),
    }
    metrics_l = {
        "tp": AverageMeter(),
        "fp": AverageMeter(),
        "fn": AverageMeter(),
        "tp_c": AverageCategoryMeter(num_classes),
        "fn_c": AverageCategoryMeter(num_classes),
    }

    if carpk:
        pred_data, gt_data, cell_num = read_pred_and_gt_carpk(pred_file, gt_file)
    else:
        pred_data, gt_data, cell_num = read_pred_and_gt(pred_file, gt_file)
    for k, v in pred_data.items():
        # init
        gt_p, pred_p, fn_gt_index, fp_pred_index = [], [], [], []
        tp_s, fp_s, fn_s, tp_l, fp_l, fn_l = [0, 0, 0, 0, 0, 0]
        tp_c_s = np.zeros([num_classes])
        fn_c_s = np.zeros([num_classes])
        tp_c_l = np.zeros([num_classes])
        fn_c_l = np.zeros([num_classes])

        if gt_data[k]["num"] == 0 and pred_data[k]["num"] != 0:
            pred_p = pred_data[k]["points"]
            fp_pred_index = np.array(range(pred_p.shape[0]))
            fp_s = fp_pred_index.shape[0]
            fp_l = fp_pred_index.shape[0]

        if pred_data[k]["num"] == 0 and gt_data[k]["num"] != 0:
            gt_p = gt_data[k]["points"]
            level = gt_data[k]["level"]
            fn_gt_index = np.array(range(gt_p.shape[0]))
            fn_s = fn_gt_index.shape[0]
            fn_l = fn_gt_index.shape[0]
            for i_class in range(num_classes):
                fn_c_s[i_class] = (level[fn_gt_index] == i_class).sum()
                fn_c_l[i_class] = (level[fn_gt_index] == i_class).sum()

        if gt_data[k]["num"] != 0 and pred_data[k]["num"] != 0:
            pred_p = pred_data[k]["points"]
            gt_p = gt_data[k]["points"]
            sigma_s = gt_data[k]["sigma"][:, 0]
            sigma_l = gt_data[k]["sigma"][:, 1]
            level = gt_data[k]["level"]

            # dist
            dist_matrix = ss.distance_matrix(pred_p, gt_p, p=2)
            match_matrix = np.zeros(dist_matrix.shape, dtype=bool)

            # sigma_s and sigma_l
            tp_s, fp_s, fn_s, tp_c_s, fn_c_s = compute_metrics(
                dist_matrix,
                match_matrix,
                pred_p.shape[0],
                gt_p.shape[0],
                sigma_s,
                level,
                num_classes,
            )
            tp_l, fp_l, fn_l, tp_c_l, fn_c_l = compute_metrics(
                dist_matrix,
                match_matrix,
                pred_p.shape[0],
                gt_p.shape[0],
                sigma_l,
                level,
                num_classes,
            )

        metrics_s["tp"].update(tp_s)
        metrics_s["fp"].update(fp_s)
        metrics_s["fn"].update(fn_s)
        metrics_s["tp_c"].update(tp_c_s)
        metrics_s["fn_c"].update(fn_c_s)
        metrics_l["tp"].update(tp_l)
        metrics_l["fp"].update(fp_l)
        metrics_l["fn"].update(fn_l)
        metrics_l["tp_c"].update(tp_c_l)
        metrics_l["fn_c"].update(fn_c_l)

        gt_count, pred_cnt = gt_data[k]["num"], pred_data[k]["num"]
        s_mae = abs(gt_count - pred_cnt)
        s_mse = (gt_count - pred_cnt) * (gt_count - pred_cnt)
        cnt_errors["mae"].update(s_mae)
        cnt_errors["mse"].update(s_mse)

        if gt_count != 0:
            s_nae = abs(gt_count - pred_cnt) / gt_count
            cnt_errors["nae"].update(s_nae)

    ap_s = metrics_s["tp"].sum / (metrics_s["tp"].sum + metrics_s["fp"].sum + 1e-20)
    ar_s = metrics_s["tp"].sum / (metrics_s["tp"].sum + metrics_s["fn"].sum + 1e-20)
    if (ap_s + ar_s) <= 0:
        return 0, 0, 0, 0, 0, 0, 0, 0
    f1m_s = 2 * ap_s * ar_s / (ap_s + ar_s)

    ap_l = metrics_l["tp"].sum / (metrics_l["tp"].sum + metrics_l["fp"].sum + 1e-20)
    ar_l = metrics_l["tp"].sum / (metrics_l["tp"].sum + metrics_l["fn"].sum + 1e-20)
    f1m_l = 2 * ap_l * ar_l / (ap_l + ar_l)

    mae = cnt_errors["mae"].avg
    mse = np.sqrt(cnt_errors["mse"].avg)

    return ap_s, ar_s, f1m_s, ap_l, ar_l, f1m_l, mae, mse


def Localization(floc_path, floc_new, gt_location_file, carpk):
    file_order(floc_path, floc_new)
    ap_s, ar_s, f1m_s, ap_l, ar_l, f1m_l, mae, mse = location_main(
        floc_new, gt_location_file, carpk
    )
    return ap_s, ar_s, f1m_s, ap_l, ar_l, f1m_l, mae, mse
