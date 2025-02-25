# End-to-End Pose-Action Recognition via Implicit Pose Encoding and Multi-Scale Skeleton Modeling
Official implementation of the paper (End-to-End Pose-Action Recognition via Implicit Pose Encoding and Multi-Scale Skeleton Modeling).

## Abstract
Human action recognition from videos is a cornerstone in multimedia and computer vision, with applications spanning human behavior analysis, video surveillance, and intelligent transportation systems. Traditional RGB-based methods suffer from low robustness to lighting changes and high computational costs. In contrast, skeleton-based approaches exhibit enhanced robustness by excluding background noise. However, existing skeleton-based methods often adopt a two-stage pipeline, involving separate pose estimation and action recognition models, leading to error propagation and amplification. To address these challenges, we propose an end-to-end deep framework that unifies pose estimation and skeleton action recognition within a joint optimization architecture. Our framework eliminates explicit intermediate pose decoding by preserving implicit pose representations through Gaussian probability encoding, thereby avoiding quantization errors and enabling differentiable optimization. Building upon STGCN++, we introduce a multi-scale network, MS-STGCN++, to enhance multi-scale feature exchange. Additionally, a lightweight visual semantic branch integrates discarded low-level features with skeleton motion patterns via cross-attention, providing complementary scene context. Extensive experiments on NTU-60, NTU-120, and N-UCLA datasets demonstrate that our framework outperforms the two-stage baseline in both pose estimation (achieving an mAP improvement of 2.2% on NTU-60 Xview) and action recognition accuracy (increasing accuracy by 0.8% on NTU-60 Xview), offering a practical solution for integrated pose-action recognition.

## Installation
Note that python3 is required and is used unless specified. You should install required python packages before you continue:

```shell
pip3 install -r requirements.txt
```

## Data Preparation
Since the license of the NTU RGB+D 60 and 120 datasets do not allow further distribution, derivation or generation, we cannot release the processed datasets publicly. You can download the official data [here](https://rose1.ntu.edu.sg/dataset/actionRecognition/). 

Then, video frames from NTU should be also manually extracted. A Python [script](tools\extract_ntu_frames.py) is provided to help in this task.

A pre-trained Lite-HRNet model checkpoint should be loaded before training our model. The pre-trained checkpoints can be downloaded [here](https://drive.google.com/file/d/1ZewlvpncTvahbqcCFb-95C3NHet30mk5/view?usp=sharing). 

## Training & Testing

You can use following commands for training and testing. Basically, we support distributed training on a single server with multiple GPUs.

```shell
# Training and testing
python exp\train.py
```

The config file are located in [here](core\config\baseline\baseline_cv.yaml), by default it's trains on NTU-60 dataset under cross-view (cs) protocol, you can adjust the experiment by customizing the configurations. 
Here are some tips:
```shell
DATA_DIR: # to specify your data root
GPUS: # to choose the gpu id for conducting experiments. You need at least 2 gpus by default for training.
PRETRAINED: # to specify your Lite-HRNet checkpoint root
MODEL:
  MULTI_FUSE: # to enable the visual branch. False by default.
TRAIN:
  EVAL_MODE: # to specify the evaluation protocol ('cs' for cross-subject or 'cv' for cross-view). 'cv' by default.
  END_EPOCH: # to specify the num epochs. 100 by default.
```