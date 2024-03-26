## Motion-to-Matching: A Mixed Paradigm for 3D Single Object Tracking (RAL 2023)

This is the official code release of "Motion-to-Matching: A Mixed Paradigm for 3D Single Object Tracking"

![](https://github.com/LeoZhiheng/MTM-Tracker/blob/main/Picture/Quantitative_results.png)

## Abstract

3D single object tracking with LiDAR points is an important task in the computer vision field. Previous methods usually adopt the matching-based or motion-centric paradigms to estimate the current target status. However, the former is sensitive to the similar distractors and the sparseness of point clouds due to relying on appearance matching, while the latter usually focuses on short-term motion clues (eg. two frames) and ignores the long-term motion pattern of target. To address these issues, we propose a mixed paradigm with two stages, named **MTM-Tracker**, which combines motion modeling with feature matching into a single network. Specifically, in the first stage, we exploit the continuous historical boxes as motion prior and propose an encoder-decoder structure to locate target coarsely. Then, in the second stage, we introduce a feature interaction module to extract motion-aware features from consecutive point clouds and match them to refine target movement as well as regress other target states. Extensive experiments validate that our paradigm achieves competitive performance on large-scale datasets (70.9\% in KITTI and 51.70\% in NuScenes).

## Method

![](https://github.com/LeoZhiheng/MTM-Tracker/blob/main/Picture/MTM-Tracker.png)

## Performance

![](https://github.com/LeoZhiheng/MTM-Tracker/blob/main/Picture/Performance.png)

## Setup
### Installation
```
conda create -n mtm python=3.8 -y
conda activate mtm

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# please refer to https://github.com/traveller59/spconv
pip install spconv-cu113

git clone https://github.com/LeoZhiheng/MTM-Tracker.git
cd MTM-Tracker/
pip install -r requirements.txt

python setup.py develop

cd ltr/ops/deformattn/
python setup.py develop
cd ../../../
```

### Dataset preparation
Download the dataset from [KITTI Tracking](https://www.cvlibs.net/datasets/kitti/) and organize the downloaded files as follows:

```
MTM-Tracker                                           
|-- data                                     
|   |-- kitti                                                                          
│   │   └── training
│   │       ├── calib
│   │       ├── label_02
│   │       └── velodyne
```

## QuickStart
### Train
For training, you can customize the training by modifying the parameters in the yaml file of the corresponding model, such as '**CLASS_NAMES**'.

After configuring the yaml file, run the following command to parser the path of config file.

```
cd tools/
bash scripts/dist_train.sh
```
