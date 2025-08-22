# Few-shot Object Localization
Official PyTorch implementation of Few-shot Object Localization.

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2403.12466-b31b1b.svg)](https://arxiv.org/abs/2403.12466) -->

Existing object localization methods are typically trained under strong supervision to detect specific object classes, relying heavily on large amounts of labeled data. However, obtaining sufficient annotations is often impractical in real-world scenarios, which greatly limits the applicability of these models. To address this challenge, we introduce a novel task, Few-Shot Object Localization (FSOL), which aims to achieve accurate localization with only a few labeled samples. For practical relevance, we further benchmark two FSOL subtasks: class-agnostic FSOL and class-specific FSOL, which localize intra-class objects guided by either image-level prompts or global pre-defined prompts. To tackle these tasks, we propose an efficient and high-performance baseline model that integrates a dual-path feature augmentation module, enhancing shape association and gradient difference recognition between the annotated prompt and the query image, together with a self-query module that further refines the similarity map using the original query features. Extensive experiments demonstrate that our approach significantly outperforms existing methods on FSOL tasks, providing a strong benchmark for future research. The architecture of the model is as follows:

![image](https://github.com/Ryh1218/FSOL/blob/main/assets/FSOL.png)

## Start
### Dependencies
```
conda create -n fsol python=3.8
conda activate fsol
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
pip install easydict==1.9 numpy==1.21.2 opencv_python==4.5.5.64 pillow==9.4.0 pyyaml==6.0 scipy==1.7.2 tqdm==4.64.0
```

### Prepare Datasets
#### FSC-147 
Official website: https://github.com/cvlab-stonybrook/LearningToCountEverything

1. Copy 'images_384_VarV2' and 'gt_density_map_adaptive_384_VarV2' to data/FSC147_384_V2
2. Run gen_gt_density.py
The structure should be as follows:
```
|-- data
    |-- FSC147_384_V2
        |-- images_384_VarV2
        |-- gt_density_map_adaptive_384_VarV2
        |-- train.json
        |-- val.json
        |-- test.json
        |-- gen_gt_density.py
```

#### ShanghaiTech
Official website: https://github.com/desenzhou/ShanghaiTechDataset

For ShanghaiTech partA:
1. Copy 'test_data', 'train_data' to 'data/ShanghaiTech/part_A'
2. Run gen_gt_density.py

For ShanghaiTech partB:
1. Copy 'test_data', 'train_data' to 'data/ShanghaiTech/part_B'
2. Run gen_gt_density.py
The structure should be as follows:
```
|-- data
    |-- ShanghaiTech
        |-- part_A
            |-- train_data
            |-- test_data
            |-- gen_gt_density.py
            |-- train.json
            |-- test.json
            |-- exemplar.json
        |-- part_B
            |-- train_data
            |-- test_data
            |-- gen_gt_density.py
            |-- train.json
            |-- test.json
            |-- exemplar.json
```

#### CARPK
Official website: https://lafi.github.io/LPN/
1. Copy 'CARPK/CARPK_devkit/data/Images' to 'data/CARPK_devkit/'
2. Run gen_gt_density.py
The structure should be as follows:
```
|-- data
    |-- CARPK_devkit
        |-- Images
        |-- gen_gt_density.py
        |-- train.json
        |-- test.json
        |-- exemplar.json
```

#### PUCPR+
Official website: https://lafi.github.io/LPN/
1. Copy 'datasets/PUCPR+_devkit/data/Images' to 'data/PUCPR+_devkit'
2. Run gen_gt_density.py
The structure should be as follows:
```
|-- data
    |-- PUCPR+_devkit
        |-- Images
        |-- gen_gt_density.py
        |-- train.json
        |-- test.json
        |-- exemplar.json
```

#### UCSD
1. Copy 'ucsdpeds/vidf' to 'data/UCSD/'
2. Run gen_gt_density.py
The structure should be as follows:
```
|-- data
    |-- UCSD
        |-- vidf
        |-- gen_gt_density.py
        |-- train.json
        |-- test.json
        |-- exemplar.json
```


## Training
You can train FSOL model on different datasets. Under the root directory, you can first enter the experiment folder by:

**FSC-147:**
`cd experiments/FSC147`

**ShanghaiTech partA:**
`cd experiments/ShanghaiTech/part_A`

**ShanghaiTech partB:**
`cd experiments/ShanghaiTech/part_B`

**CARPK:** 
`cd experiments/CARPK`

**PUCPR+:** 
`cd experiments/PUCPR+`

**UCSD:** 
`cd experiments/UCSD`

Then, you can run `sh train.sh` to train the FSOL model and run `sh eval.sh` or `sh test.sh` to evaluate FSOL model. Inside each .sh file, you can adjust the `CUDA_VISIBLE_DEVICES`.

All the experiments are executed on single RTX 3090 GPU.

<!-- ## Model Weight
You can access the following links to get pretrained weight of one-shot FSOL model. Google Drive: [here](https://drive.google.com/file/d/1oQicG8qlP2oEsOQ5oNUFcIz-RzZKNKpb/view?usp=sharing); Baidu Netdisk: [here](https://pan.baidu.com/s/1tDJSI94L4BnuRfRQm3S4ig?pwd=2uvc). -->

### Load model weight
You can load pre-trained weight of FSOL model through modifying the config file for each experiment. For example, for FSC147 experiment, move the model weight to experiments/FSC147/checkpoints, then access to experiments/FSC147/config.yaml and modify as follows:

```saver:
  ifload: True
  load_weight: FSOL_Final.tar
  save_dir: checkpoints/
  log_dir: log/
```

## Thanks
This code is based on [SAFECount](https://github.com/zhiyuanyou/SAFECount) and [FIDTM](https://github.com/dk-liang/FIDTM). Many thanks for your code implementation.

<!-- ## Reference
```
@article{FSOL,
  title={Few-shot Object Localization},
  author={Yunhan Ren and Bo Li and Chengyang Zhang and Yong Zhang and Baocai Yin},
  journal={ArXiv},
  year={2024},
  volume={abs/2403.12466},
  url={https://api.semanticscholar.org/CorpusID:268531824}
}
``` -->
