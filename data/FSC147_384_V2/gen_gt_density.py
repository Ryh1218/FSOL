import json
import math
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

if __name__ == "__main__":
    current_path = os.path.abspath(__file__)
    
    root = current_path.split('/gen_gt_density.py')[0]
    pool = ["test", "val", "train"]

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

            img_path = os.path.join(root, "images_384_VarV2", file_name + ".jpg")

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
                sigma_s = 5
                sigma_l = 10
                f.write(
                    "{} {} {} {} {} ".format(
                        math.floor(data[0]), math.floor(data[1]), sigma_s, sigma_l, 1
                    )
                )
            f.write("\n")
        f.close()
