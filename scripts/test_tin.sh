python -u -W ignore test.py something RGB \
     --arch resnet50 --num_segments 8 --print-freq 50 --npb \
     --batch-size 64 -j 5 --dropout 0.8 --consensus_type=avg  \
     --tin --shift_div=4 \
     --temporal_clips=2 \
     --random_crops=1 \
     --scale_size 256 --crop_size 256 \
     --gpus 0 1 2 3 4 5 6 7 \
     --resume_path=checkpoint/xxx/ckpt.best.pth.tar \
