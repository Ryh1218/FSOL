import json
import math
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def gen_gaussian2d(shape, sigma=1):
    h, w = [_ // 2 for _ in shape]
    y, x = np.ogrid[-h : h + 1, -w : w + 1]
    gaussian = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0
    return gaussian


def draw_gaussian(density, center, radius, k=1, delte=6, overlap="add"):
    diameter = 2 * radius + 1
    gaussian = gen_gaussian2d((diameter, diameter), sigma=diameter / delte)
    gaussian = gaussian / gaussian.sum()
    height, width = density.shape[0:2]
    x, y = center.astype(np.int)
    x = min(x, width - 1)
    y = min(y, height - 1)
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    if overlap == "max":
        masked_density = density[y - top : y + bottom, x - left : x + right]
        masked_gaussian = gaussian[
            radius - top : radius + bottom, radius - left : radius + right
        ]
        np.maximum(masked_density, masked_gaussian * k, out=masked_density)
    elif overlap == "add":
        density[y - top : y + bottom, x - left : x + right] += gaussian[
            radius - top : radius + bottom, radius - left : radius + right
        ]
    else:
        raise NotImplementedError


def _min_dis_global(points):
    """
    points: m x 2, m x [x, y]
    """
    dis_min = float("inf")
    for point in points:
        point = point[None, :]  # 2 -> 1 x 2
        dis = np.sqrt(np.sum((points - point) ** 2, axis=1))  # m x 2 -> m
        dis = sorted(dis)[1]
        if dis_min > dis:
            dis_min = dis
    return dis_min


def _min_dis_local(point, points):
    """
    point: [x, y]
    points: m x 2, m x [x, y]
    """
    point = point[None, :]  # 2 -> 1 x 2
    dis = np.sqrt(np.sum((points - point) ** 2, axis=1))  # m x 2 -> m
    dis = sorted(dis)[1]
    return dis


def points2density(points, max_scale, max_radius):
    """
    points: m x 2, m x [x, y]
    """
    num_points = points.shape[0]
    density = np.zeros(image_size, dtype=np.float32)  # [h, w]
    if num_points == 0:
        return density
    elif num_points == 1:
        radius = max_radius
        for point in points:
            draw_gaussian(density, point, radius, overlap="max")
    else:
        dis_min = _min_dis_global(points)
        for point in points:
            # calculate the minimum distance of a point between its neighbors
            dis = _min_dis_local(point, points)
            dis = min(dis, max_scale * dis_min, max_radius)
            radius = max(3, int(dis))
            draw_gaussian(density, point, radius, overlap="max")
    return density


if __name__ == "__main__":
    current_path = os.path.abspath(__file__)
    root = current_path.split('/gen_gt_density.py')[0]
    
    
    anno_files = [os.path.join(root, 'train.json'), os.path.join(root, 'test.json')]
    root_dirs = [os.path.join(root, 'train_data'), os.path.join(root, 'test_data')]
    for anno_file, root_dir in zip(anno_files, root_dirs):
        img_dir = os.path.join(root_dir, "images")
        gt_dir = os.path.join(root_dir, "gt_density_map")
        os.makedirs(gt_dir, exist_ok=True)

        # read all data
        metas = []
        with open(anno_file, "r+") as fr:
            for line in fr:
                meta = json.loads(line)
                metas.append(meta)

        # create gt density map
        for meta in tqdm(metas):
            filename = meta["filename"]
            filepath = os.path.join(img_dir, filename)
            image = cv2.imread(filepath)
            image_size = image.shape[0:2]  # [h, w]

            points = meta["points"]
            points = np.array(points)
            cnt_gt = points.shape[0]

            density = points2density(points, max_scale=3.0, max_radius=10.0)

            if not cnt_gt == 0:
                cnt_cur = density.sum()
                density = density / cnt_cur * cnt_gt

            filename_, _ = os.path.splitext(filename)
            save_path = os.path.join(gt_dir, filename_ + ".npy")
            np.save(save_path, density)
        # print(f"Success: generate gt density map for {filename}")

    pool = ["test", "train"]

    json_paths = []
    for i in pool:
        json_path = os.path.join(root, "{}.json".format(i))
        json_paths.append(json_path)

    json_paths.sort()

    for json_path in tqdm(json_paths):
        pool_name = json_path.split("/")[-1].split(".")[0]
        f = open(os.path.join(root, "sf_{}_gt.txt".format(pool_name)), "w+")
        json_file = open(json_path, "r", encoding="utf-8")
        for line in json_file.readlines():
            dic = json.loads(line)
            coordinates = np.asarray(dic["points"])

            file_name = dic["filename"].replace(".jpg", "")

            img_path = os.path.join(
                root, "{}_data".format(pool_name), "images", file_name + ".jpg"
            )

            img = Image.open(img_path)
            width = img.size[0]
            height = img.size[1]

            re_coordinates = []
            for cor in coordinates:
                cor[0] = int(cor[0] / width * 512)
                cor[1] = int(cor[1] / height * 512)
                re_coordinates.append([cor[0], cor[1]])
            f.write("{} {} ".format(int(file_name.split("_")[-1]), len(re_coordinates)))

            for data in re_coordinates:
                sigma_s = 4
                sigma_l = 8
                f.write(
                    "{} {} {} {} {} ".format(
                        math.floor(data[0]), math.floor(data[1]), sigma_s, sigma_l, 1
                    )
                )
            f.write("\n")
        f.close()
