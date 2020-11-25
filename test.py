# Code for "Temporal Interlacing Network"
# Hao Shao, Shengju Qian, Yu Liu
# shaoh19@mails.tsinghua.edu.cn, sjqian@cse.cuhk.edu.hk, yuliu@ee.cuhk.edu.hk

import os
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import accuracy, save_bias
from ops.temporal_shift import make_temporal_pool
from utils import *

best_prec1 = 0

def main():
    global args, best_prec1, TRAIN_SAMPLES
    args = parser.parse_args()

    num_class, args.train_list, args.val_list, args.test_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.modality)
    if os.path.exists(args.test_list):
        args.val_list = args.test_list


    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                temporal_pool=args.temporal_pool,
                non_local=args.non_local,
                tin=args.tin)


    crop_size = args.crop_size
    scale_size = args.scale_size
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]

    print(args.gpus)
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    if os.path.isfile(args.resume_path):
        print(("=> loading checkpoint '{}'".format(args.resume_path)))
        checkpoint = torch.load(args.resume_path)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(("=> loaded checkpoint '{}' (epoch {})"
               .format(args.evaluate, checkpoint['epoch'])))
    else:
        print(("=> no checkpoint found at '{}'".format(args.resume_path)))


    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    if args.random_crops == 1:
       crop_aug = GroupCenterCrop(args.crop_size)
    elif args.random_crops == 3:
       crop_aug = GroupFullResSample(args.crop_size, args.scale_size, flip=False)
    elif args.random_crops == 5:
       crop_aug = GroupOverSample(args.crop_size, args.scale_size, flip=False)
    else:
       crop_aug = MultiGroupRandomCrop(args.crop_size, args.random_crops),


    test_dataset = TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   multi_class=args.multi_class,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(args.scale_size)),
                       crop_aug,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample,
                   test_mode=True,
                   temporal_clips=args.temporal_clips)


    test_loader = torch.utils.data.DataLoader(
            test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test(test_loader, model, args.start_epoch)

def test(val_loader, model, epoch):
    batch_time = AverageMeter(args.print_freq)
    top1 = AverageMeter(args.print_freq)
    top5 = AverageMeter(args.print_freq)
    mAPs = AverageMeter(args.print_freq)

    # switch to evaluate mode
    model.eval()

    dup_samples = args.random_crops * args.temporal_clips 
    end = time.time()
    total_num = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if i % 50 ==0: print('Test Complete: %d / %d' % (i, len(val_loader)))
            input = input.cuda()
            target = target.cuda()

            sizes = input.shape
            input = input.view((sizes[0] * dup_samples, -1, sizes[2], sizes[3]))

            # compute output
            output = model(input)
            sizes = output.shape
            output = output.view((sizes[0] // dup_samples, -1, sizes[1]))
            output = torch.nn.functional.softmax(output, 2)
            output = torch.mean(output, 1)

            num = input.size(0)
            total_num += num
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.multi_class:
                from ops.calculate_map import calculate_mAP
                mAP = calculate_mAP(output.data, target)
                mAPs.update(mAP, num)
            else:
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                top1.update(prec1.item(), num)
                top5.update(prec5.item(), num)


        if args.multi_class:
            final_mAP = mAPs.avg
            output = (' * Map {:.3f}\t total_num={}'.format(final_mAP, total_num))
        else:
            output = (' * Prec@1 {:.3f}\t Prec@5 {:.3f}\ttotal_num={}'.format(top1.avg, top5.avg, total_num))
        print(output)

    if args.multi_class:
        return mAPs.avg
    else:
        return top1.avg

if __name__ == '__main__':
    main()
