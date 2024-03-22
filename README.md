# Few-shot Object Localization

![image](https://github.com/Ryh1218/FSOL/blob/main/assets/FSOL.png)

## Start
### Dependencies
```
pip install -r requirements.txt
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
1. Copy 'test_data', 'train_data' to data/ShanghaiTech/part_A
2. Run gen_gt_density.py

For ShanghaiTech partB:
1. Copy 'test_data', 'train_data' to data/ShanghaiTech/part_B
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
1. Copy 'CARPK/CARPK_devkit/data/Images' to data/CARPK_devkit/
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

Then, you can run `sh train.sh #GPU_NUM #GPU_ID` to train the FSOL model. For example, training with one GPU and ID 0 should be `sh train.sh 1 0`. For FSC-147 dataset, you can run `sh eval.sh #GPU_NUM #GPU_ID` for evaluation and `sh test.sh #GPU_NUM #GPU_ID` for testing. For other datasets, you can run `sh eval.sh #GPU_NUM #GPU_ID` for testing.

We suggest you to train the model on single GPU.


## Results
All of the following results are experimented on one NVIDIA RTX 3090 with one support sample provided.

| **Dataset**       | **F1($\sigma$=5)** | **AP($\sigma$=5)** | **AR($\sigma$=5)** | **F1($\sigma$=10)** | **AP($\sigma$=10)** | **AR($\sigma$=10)** |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| FSC-147 | 53.4 | 55.5 | 51.4 | 70.0 |	72.7 | 67.4 |
| Shanghai A | 52.4 | 58.4 | 47.6 | 69.6 | 77.6 | 63.1 |
| Shanghai B | 67.2 | 75.5 | 60.5 | 78.0 | 88.4 | 70.9 |
| CARPK | 81.84 | 80.9 | 82.8 | 93.46 | 92.38 | 94.56 |

## Thanks
This code is based on SAFECount (https://github.com/zhiyuanyou/SAFECount) and FIDTM (https://github.com/dk-liang/FIDTM). Many thanks for your code implementation.