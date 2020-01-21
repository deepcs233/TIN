# Temporal Interlacing Network [pdf](http://thesouthfrog.com/about.me/Shengju_files/TIN.pdf) [arxiv]

Shao Hao, [Shengju Qian](http://thesouthfrog.com/about.me/) and [Yu Liu](http://liuyu.us/)

## Overview

![framework](http://thesouthfrog.com/about.me/Shengju_files/TIN.png)

We present Temporal Interlacing Network(TIN), a simple interlacing operator. TIN performs equivalently to the regularized temporal convolution network (r-TCN), but gains 4% more accuracy with 6x less latency on 6 challenging benchmarks. These results push the state-of-the-art performances of video understanding by a considerable margin. 

Meanwhile, TIN severs as the key component of the winning entry in ICCV 2019 - Multi Moments in Time challenge. You can find more detailed solutions in the [report](http://moments.csail.mit.edu/challenge2019/efficient_challenge_report.pdf).  [Challenge Website](http://moments.csail.mit.edu/results2019.html)


## Prerequisites

The code is built with following libraries:

- [PyTorch](https://pytorch.org/) 0.4(switch to branch 0.4), 1.0 or higher
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [tqdm](https://github.com/tqdm/tqdm.git)
- [sklearn](https://github.com/scikit-learn/scikit-learn)
- [scikit-learn](https://scikit-learn.org/stable/)

For extracting frames from video data, you may need [ffmpeg](https://www.ffmpeg.org/).

### Data Preparation
Please refer to [TSN](https://github.com/yjxiong/temporal-segment-networks) and [TSM](https://github.com/mit-han-lab/temporal-shift-module#prerequisites) repo for the detailed guide of data pre-processing. We provide the example pre-processing scripts for Kinetics and Something-Something datasets.

![cuda](http://thesouthfrog.com/about.me/Shengju_files/cuda.png)

## Get Started(CUDA operator)
We provide the CUDA operator of TIN to provide acceleration. The pure Python version runs about 1.1s on 1080TI and the optimized code version runs about 0.7s. You may first build the provided library before getting started. 

```bash
cd cuda_shift
bash make.sh
```

### Train
Here we provide several examples to train with this repo at Something-Something v1 dataset:

** Train from scratch (with imagenet pretrain) ** 
```bash
python -u main.py something RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.02 --lr_type cos  --epochs 40 --use_warmup --warmup_epochs 1 \
     --store_name "tin_example"  \
     --batch-size 48 -j 4 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
     --tin --shift_div=4 --gpus 0 1 2 3 4 5 6 7 --workers 36  \
```

** Train from pretrained model  **
```bash
python -u main.py something RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.008 --lr_type cos  --epochs 40 --use_warmup --warmup_epochs 1 \
     --store_name "tin_example"  \
     --batch-size 48 -j 4 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
     --tin --shift_div=4 --gpus 0 1 2 3 4 5 6 7 --workers 36  \
     --tune_from=pretrained/xxx.pth.tar \
```

** Resume training **
```bash
python -u main.py something RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.008 --lr_type cos  --epochs 40 --use_warmup --warmup_epochs 1 \
     --store_name "tin_example"  \
     --batch-size 48 -j 4 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
     --tin --shift_div=4 --gpus 0 1 2 3 4 5 6 7 --workers 36  \
     --resume=True \
     --resume_path=checkpoint/xx/xxx.pth.tar \
```

### Test
Here we provide several examples to test with this repo at Something-Something v1 dataset:

** Test with center crop and 1 temporal clips (same as training setting)**
```bash
python -u  ignore test.py something RGB \
     --arch resnet50 --num_segments 8 --print-freq 50 --npb \
     --batch-size 24 -j 5 --dropout 0.8 --consensus_type=avg  \
     --tin --shift_div=4 \
     --temporal_clips=1 --random_crops=1 \
     --scale_size 256 --crop_size 224 \
     --gpus 0 1 2 3 4 5 6 7 \
     --resume_path=checkpoint/xxx/ckpt.best.pth.tar \
```

** Test with Full resolution, 2 temporal clips and 3 spatial crops **
```bash
python -u  ignore test.py something RGB \
     --arch resnet50 --num_segments 8 --print-freq 50 --npb \
     --batch-size 24 -j 5 --dropout 0.8 --consensus_type=avg  \
     --tin --shift_div=4 \
     --temporal_clips=2  --random_crops=3 \
     --scale_size 256 --crop_size 256 \
     --gpus 0 1 2 3 4 5 6 7 \
     --resume_path=checkpoint/xxx/ckpt.best.pth.tar \
```


### Acknowledge 

This code is based on the original TSN and TSM codebased. Thanks to their great work!

## Citation

If you use our code, please consider citing our paper or challenge report:

```
```

