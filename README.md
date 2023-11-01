# Code_EvaDB

#### Python packages:
1. Install [EvaDB](https://evadb.readthedocs.io/en/stable/source/overview/getting-started.html) and [norfair](https://github.com/hanryxu/norfair) in python environment

2. Install some other packages for color_NN model of [AICITY](https://github.com/zef1611/AIC23_NLRetrieval_HCMIU_CVIP) according to [requirements.txt](https://github.com/zef1611/AIC23_NLRetrieval_HCMIU_CVIP/blob/main/requirements.txt)
<br>

#### Checkpoints:
1. Download [color_NN model](https://drive.google.com/drive/folders/1J6zSRS7ubWinO9BxKIt7e8lI2Z1zD6g2), and save it like `1_query_stateless_property/data/checkpoints/CLIP_recognition_color/checkpoint_recognition_color.pth`
<br>

#### Video data:
1. Download [Banff](link), [Jackson](link) and [Southampton](link), and put them in directory `Three_Datasets`. Ensure that `Three_Datasets` and `Code_EvaDB` are at the same level.

2. Download ua_detrac.mp4 as `wget -nc "https://www.dropbox.com/s/k00wge9exwkfxz6/ua_detrac.mp4?raw=1" -O ua_detrac.mp4`, and put it in `1_query_stateless_property/`
<br>

#### How to run:
1. Example for query red cars: 
`cd ./1_query_stateless_property/`
`python query_red_car.py`
