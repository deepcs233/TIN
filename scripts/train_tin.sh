python -u main.py something RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.008 --lr_type cos  --epochs 36 --store_name "tin_example"  \
     --batch-size 48 -j 4 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
     --tin --shift_div=4 --gpus 0 1 2 3 4 5 6 7 --workers 36  \
     --use_warmup --warmup_epochs 1 \
     --resume \
     --resume_path=checkpoint/xxx/ckpt.best.pth.tar \
