import os
from abc import ABC

import cv2
import numpy as np
from scipy import spatial as ss


class Visualizer(ABC):
    def __init__(
        self,
        vis_dir,
        img_dir,
        activation=None,
        normalization=True,
        with_image=True,
        ifvis=False,
    ):
        """
        vis_dir: dir to save the visualization results
        img_dir: dir of img
        normalization: if True, the heatmap 1). rescale to [0,1], 2). * 255, 3). visualize.
                       if False, the heatmap 1). * 255, 2). visualize.
        with_image: if True, the image & heatmap would be combined to visualize.
                    if False, only the heatmap would be visualized.
        """
        self.vis_dir = vis_dir
        self.img_dir = img_dir
        self.activation_fn = (
            self.build_activation_fn(activation) if activation else None
        )
        self.normalization = normalization
        self.with_image = with_image

    def build_activation_fn(self, activation):
        if activation == "sigmoid":

            def _sigmoid(x):
                return 1 / (1 + np.exp(-x))

            return _sigmoid
        else:
            raise NotImplementedError

    def generate_bounding_boxes(self, kpoint, fname, img_path):
        '''change the data path'''
        Img_data = cv2.imread(img_path)
        Img_data = cv2.resize(Img_data, (512, 512), interpolation=cv2.INTER_CUBIC)
        ori_Img_data = Img_data.copy()

        '''generate sigma'''
        pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
        leafsize = 2048
        # build kdtree
        if pts.shape[0] != 0:
            tree = ss.KDTree(pts.copy(), leafsize=leafsize)

            distances, locations = tree.query(pts, k=4)
            for index, pt in enumerate(pts):
                pt2d = np.zeros(kpoint.shape, dtype=np.float32)
                pt2d[pt[1], pt[0]] = 1.
                if np.sum(kpoint) > 1:
                    sigma = (distances[index][1] + distances[index][2] + distances[index][3]) * 0.1
                else:
                    sigma = np.average(np.array(kpoint.shape)) / 2. / 2.  # case: 1 point
                sigma = min(sigma, min(Img_data.shape[0], Img_data.shape[1]) * 0.05)
                if sigma < 6:
                    t = 2
                else:
                    t = 2
                Img_data = cv2.rectangle(Img_data, (int(pt[0] - sigma), int(pt[1] - sigma)),
                                        (int(pt[0] + sigma), int(pt[1] + sigma)), (0, 255, 0), t)
            return ori_Img_data, Img_data
        else:
            return ori_Img_data, ori_Img_data
        

    def show_map(self, input):
        input[input < 0] = 0
        input = input[0][0]
        fidt_map1 = input
        fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
        fidt_map1 = fidt_map1.astype(np.uint8)
        fidt_map1 = cv2.applyColorMap(fidt_map1, 2)
        return fidt_map1
        

    def apply_scoremap(self, image, scoremap, alpha=0.5):
        np_image = np.asarray(image, dtype=float)
        scoremap = (scoremap * 255).astype(np.uint8)
        scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
        scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
        return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

    def vis_result(self, filename, resname, height, width, output):
        """
        filename: str
        image: tensor c x h x w
        """
        output = output.permute(1, 2, 0)  # c x h x w -> h x w x c
        output = output.cpu().detach().numpy()
        output = cv2.resize(output, (int(width), int(height)))
        if self.activation_fn:
            output = self.activation_fn(output)
        if self.normalization:
            output = (output - output.min()) / (output.max() - output.min())
        if self.with_image:
            img_path = os.path.join(self.img_dir, filename)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            output = self.apply_scoremap(image, output)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            output = cv2.resize(output, (512, 512), interpolation=cv2.INTER_CUBIC)
        else:
            output = (output * 255).astype(np.uint8)
            output = cv2.resize(output, (512, 512), interpolation=cv2.INTER_CUBIC)
        return output

    def vis_batch(self, input, kpoint, filename):
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)
        filenames = input["filename"]
        heights, widths = input["height"], input["width"]
        densities = input["density"]
        outputs = input["density_pred"]
        for (filename, height, width, density, output) in zip(
            filenames, heights, widths, densities, outputs):
            filename_, _ = os.path.splitext(filename)
            resname = "{}.png".format(filename_)
            output = self.vis_result(filename, resname, height, width, output)
            filepath = os.path.join(self.vis_dir, resname)
        
        img_path = os.path.join(self.img_dir, filename)
        ori_img, box_img = self.generate_bounding_boxes(kpoint, filename, img_path)
        show_fidt = self.show_map(outputs.data.cpu().numpy())
        gt_show = self.show_map(densities.data.cpu().numpy())
        res = np.hstack((ori_img, gt_show, show_fidt, output, box_img))
        cv2.imwrite(filepath, res)


def build_visualizer(**kwargs):
    return Visualizer(**kwargs)
