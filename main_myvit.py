import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import glob
from contextlib import nullcontext

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from vision_transformer import DINOHead, FusionHead
from dataset import MyImageFolder
import losses


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small_fd', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'deit_tiny', 'deit_small', 'vit_small_fd'],
                        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--arch_img', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'deit_tiny', 'deit_small'],
                        help="""Name of architecture to train. For quick experiments with ViTs,
            we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--arch_skt', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'deit_tiny', 'deit_small'],
                        help="""Name of architecture to train. For quick experiments with ViTs,
            we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.07, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=5, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=256, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=5, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--global_crops_number', type=int, default=1, help="""Number of global
        views to generate. When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_number', type=int, default=0, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='../dataset', type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default="../output_fd", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=1, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--dataset', default='sketchy', type=str, choices=['sketchy', 'tuberlin', 'quickdraw'])
    parser.add_argument('--split', default='zeroshot', type=str, choices=['zeroshot', 'random'])
    parser.add_argument('--resume_pretrain', default=False, type=utils.bool_flag, help='load model from the pretrained checkpoint')
    parser.add_argument('--disable_dropout', default=False, type=utils.bool_flag,
                        help='disable dropout layers and path dropout')
    parser.add_argument('--skt_factor', default=2, type=int, help='in every batch, sketches/images = 1/skt_factor')
    parser.add_argument('--token_num', default=2, type=int, choices=[2, 3], help='number of tokens')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--use_align_uniform', type=utils.bool_flag, default=True, help='use alignment and uniformity or not')
    parser.add_argument('--align_uniform_weight', type=float, default=2.0,
                        help="weight of alignment and uniformity.")
    parser.add_argument('--align_uniform_epochs', type=int, default=20,
                        help="Number of epochs to reach the weight of alignment and uniformity.")
    parser.add_argument('--early_schedule_epochs', type=int, default=0,
                        help="Number of epochs to reach the specified weight for weight decay and learning rate.")
    return parser


def read_classes(path):
    with open(path) as fp:
        classes = fp.read().splitlines()
        classes = sorted(classes)
    return classes


def prepare_dataset(args):
    all_classes = read_classes(os.path.join(args.data_path, args.dataset, 'split', 'all_classes.txt'))
    test_classes = read_classes(os.path.join(args.data_path, args.dataset, 'split', args.split + '_classes.txt'))
    train_classes = np.setdiff1d(all_classes, test_classes)
    train_classes = sorted(train_classes)

    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.global_crops_number,
        args.local_crops_number,
    )

    if args.dataset == 'sketchy':
        outliers = {'airplane': ['n02691156_359-5.png', 'n02691156_24584-2.png'],
                    'sheep': ['n02411705_6501-2.png'],
                    'horse': ['n02374451_388-1.png', 'n02374451_10809-3.png'],
                    'lion': ['n02129165_6087-1.png'],
                    'spider': ['n01772222_8541-2.png'],
                    'turtle': ['n01669191_4429-3.png', 'n01669191_5861-4.png'],
                    'alarm_clock': ['n02694662_3449-5.png'],
                    'bear': ['n02131653_7633-3.png'],
                    'mouse': ['n02330245_8823-4.png']}
    else:
        outliers = None

    dataset_train_skt = MyImageFolder(os.path.join(args.data_path, args.dataset, 'sketch'), train_classes,
                                      outlier=outliers,
                                      transform=transform)
    dataset_train_img = MyImageFolder(os.path.join(args.data_path, args.dataset, 'photo'), train_classes,
                                      outlier=None,
                                      transform=transform)
    sampler_skt = torch.utils.data.DistributedSampler(dataset_train_skt, shuffle=True)
    sampler_img = torch.utils.data.DistributedSampler(dataset_train_img, shuffle=True)
    data_loader_train_skt = torch.utils.data.DataLoader(
        dataset_train_skt,
        sampler=sampler_skt,
        batch_size=args.batch_size_per_gpu // args.skt_factor,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    data_loader_train_img = torch.utils.data.DataLoader(
        dataset_train_img,
        sampler=sampler_img,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print("Data loaded: there are {} images.".format(len(dataset_train_img)))
    print("Data loaded: there are {} sketches.".format(len(dataset_train_skt)))
    print("Data loaded: there are {} train classes.".format(len(train_classes)))
    print("Data loaded: there are {} test classes.".format(len(test_classes)))
    return data_loader_train_img, data_loader_train_skt, len(train_classes)


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    data_loader_img, data_loader_skt, args.num_classes = prepare_dataset(args)

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth  # not used
            token_num=args.token_num
        )
        teacher_img = vits.__dict__[args.arch_img](patch_size=args.patch_size)
        teacher_skt = vits.__dict__[args.arch_skt](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    else:
        print(f"Unknow architecture: {args.arch}")

    teacher_img.train(False)
    teacher_skt.train(False)
    if args.disable_dropout:
        student.train(False)
    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.FDMultiCropWrapper(student,
                                       DINOHead(embed_dim, args.out_dim, use_bn=args.use_bn_in_head,
                                                norm_last_layer=args.norm_last_layer),
                                       DINOHead(embed_dim, args.out_dim, use_bn=args.use_bn_in_head,
                                                norm_last_layer=args.norm_last_layer),
                                       FusionHead(embed_dim, args.num_classes, use_bn=args.use_bn_in_head,
                                                  norm_last_layer=args.norm_last_layer))

    teacher_img = utils.MultiCropWrapper(
        teacher_img,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    teacher_skt = utils.MultiCropWrapper(
        teacher_skt,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher_img, teacher_skt = student.cuda(), teacher_img.cuda(), teacher_skt.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher_img = nn.SyncBatchNorm.convert_sync_batchnorm(teacher_img)
        teacher_skt = nn.SyncBatchNorm.convert_sync_batchnorm(teacher_skt)

        # we need DDP wrapper to have synchro batch norms working...
        teacher_img = nn.parallel.DistributedDataParallel(teacher_img, device_ids=[args.gpu])
        teacher_skt = nn.parallel.DistributedDataParallel(teacher_skt, device_ids=[args.gpu])
        teacher_img_without_ddp = teacher_img.module
        teacher_skt_without_ddp = teacher_skt.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_img_without_ddp = teacher_img
        teacher_skt_without_ddp = teacher_skt
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher_img.parameters():
        p.requires_grad = False
    for p in teacher_skt.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + args.global_crops_number,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()
    lsce_loss = losses.LabelSmoothLossDINO(args.local_crops_number + args.global_crops_number, 0.1)
    unif_loss = losses.uniform_loss
    ca_loss = losses.CenterAlignment(args.local_crops_number+args.global_crops_number, args.num_classes).cuda()
    ia_loss = losses.InstanceAlignment(args.local_crops_number+args.global_crops_number)

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups_lr(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, min(len(data_loader_img), len(data_loader_skt)),
        warmup_epochs=args.warmup_epochs,
        early_schedule_epochs=args.early_schedule_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, min(len(data_loader_img), len(data_loader_skt)),
        early_schedule_epochs=args.early_schedule_epochs,
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    loss_schedule = utils.cosine_scheduler_loss(0, args.align_uniform_weight, args.epochs, min(len(data_loader_img), len(data_loader_skt)),
                                                warmup_epochs=1, early_schedule_epochs=args.align_uniform_epochs)
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, min(len(data_loader_img), len(data_loader_skt)))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    if not args.resume_pretrain:
        ckp_name = \
            sorted(glob.glob(os.path.join(args.output_dir, 'checkpoint-*.pth')), key=lambda name: int(name[-8:-4]))[
                -1]
        utils.restart_from_checkpoint(
            ckp_name,
            run_variables=to_restore,
            student=student,
            teacher_img=teacher_img,
            teacher_skt=teacher_skt,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            dino_loss=dino_loss,
            ca_loss=ca_loss,
        )
    else:
        ckp_img_name = os.path.join(args.output_dir, "checkpoint_pretrain_img.pth")
        ckp_skt_name = os.path.join(args.output_dir, "checkpoint_pretrain_skt.pth")
        ckp_stu_name = os.path.join(args.output_dir, "checkpoint_pretrain_img.pth")
        # ckp_stu_name = os.path.join(args.output_dir, "checkpoint_pretrain_skt.pth")

        # ckp_img_name = os.path.join(args.output_dir, "checkpoint_pretrain_img_16.pth")
        # ckp_skt_name = os.path.join(args.output_dir, "checkpoint_pretrain_skt_16.pth")
        # ckp_stu_name = os.path.join(args.output_dir, "checkpoint_pretrain_img_16.pth")

        utils.restart_from_checkpoint_fd(
            ckp_img_name, ckp_skt_name, ckp_stu_name,
            teacher_img, teacher_skt, student, run_variables=to_restore
        )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader_img.sampler.set_epoch(epoch)  # before training, shuffle datasets
        data_loader_skt.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher_img, teacher_skt, teacher_img_without_ddp,
                                      teacher_skt_without_ddp, dino_loss, lsce_loss, unif_loss, ca_loss, ia_loss,
                                      data_loader_img, data_loader_skt, optimizer,
                                      lr_schedule, wd_schedule, loss_schedule, momentum_schedule,
                                      epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher_img': teacher_img.state_dict(),
            'teacher_skt': teacher_skt.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
            'ca_loss': ca_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and (epoch+1) % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint-{epoch+1:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher_img, teacher_skt, teacher_img_without_ddp, teacher_skt_without_ddp, dino_loss,
                    lsce_loss, unif_loss, ca_loss, ia_loss, data_loader_img, data_loader_skt,
                    optimizer, lr_schedule, wd_schedule,
                    loss_schedule, momentum_schedule, epoch, fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    # ======= preset some args for gradient accumulation ===============
    sub_batch_size = int(args.batch_size_per_gpu / args.gradient_accumulation_steps)
    res = args.batch_size_per_gpu % args.gradient_accumulation_steps
    if res != 0:
        grad_acc_steps = args.gradient_accumulation_steps + math.ceil(res / sub_batch_size)
    else:
        grad_acc_steps = args.gradient_accumulation_steps
    for it, ((images, lab_img), (sketches, lab_skt)) in enumerate(
            metric_logger.log_every(zip(data_loader_img, data_loader_skt), 10, header,
                                    min(len(data_loader_img), len(data_loader_skt)))):
        loss_tmp = 0

        # update weight decay and learning rate according to their schedule
        it = min(len(data_loader_img), len(data_loader_skt)) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            if i == 0 or i == 1:
                param_group['lr'] = lr_schedule[it] * 0.1
            else:
                param_group["lr"] = lr_schedule[it]
            if i == 0 or i == 2:  # only the first group is regularized; look at get_params_groups for details
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        for i in range(grad_acc_steps):
            # if i is not the last iteration, no sync for ddp model, i.e., student
            my_context = student.no_sync if (i + 1) % grad_acc_steps != 0 else nullcontext
            with my_context():
                # move images to gpu
                end_i = min(args.batch_size_per_gpu, (i + 1) * sub_batch_size)
                sub_images = [im[i * sub_batch_size: end_i].cuda(non_blocking=True) for im in images]
                sub_lab_img = lab_img[i * sub_batch_size: end_i].cuda(non_blocking=True)
                # teacher and student forward passes + compute dino loss
                with torch.cuda.amp.autocast(fp16_scaler is not None):
                    teacher_output_img = teacher_img(sub_images)
                    student_output_img_d, student_output_img_f, student_output_img_f_fea = student(sub_images, 'img')
                    loss = dino_loss(student_output_img_d, teacher_output_img, epoch, 'img')
                    if args.global_crops_number > 1:
                        loss += (lsce_loss(student_output_img_f, sub_lab_img) +
                                unif_loss(student_output_img_f_fea) * loss_schedule[it] * (0.5 / args.align_uniform_weight) +
                                ca_loss(student_output_img_f_fea, sub_lab_img, 'img') * loss_schedule[it] +
                                ia_loss(student_output_img_f_fea))
                    else:
                        loss += lsce_loss(student_output_img_f, sub_lab_img)
                        if args.use_align_uniform:
                            loss += (unif_loss(student_output_img_f_fea) * loss_schedule[it] * (0.5 / args.align_uniform_weight) +
                                     ca_loss(student_output_img_f_fea, sub_lab_img, 'img') * loss_schedule[it])
                    loss /= (grad_acc_steps + grad_acc_steps / args.skt_factor)
                    loss_tmp += loss.item()

                    if fp16_scaler is None:
                        loss.backward()
                    else:
                        fp16_scaler.scale(loss).backward()

                    if (i+1) % args.skt_factor == 0:
                        sub_sketches = [sk[(i//args.skt_factor) * sub_batch_size: ((i+1) // args.skt_factor) * sub_batch_size].cuda(non_blocking=True) for sk in sketches]
                        sub_lab_skt = lab_skt[(i//args.skt_factor) * sub_batch_size: ((i+1) // args.skt_factor)*sub_batch_size].cuda(non_blocking=True)
                        teacher_output_skt = teacher_skt(sub_sketches)
                        student_output_skt_d, student_output_skt_f, student_output_skt_f_fea = student(sub_sketches, 'skt')
                        loss = dino_loss(student_output_skt_d, teacher_output_skt, epoch, 'skt')
                        if args.global_crops_number > 1:
                            loss += (lsce_loss(student_output_skt_f, sub_lab_skt) +
                                    unif_loss(student_output_skt_f_fea) * loss_schedule[it] * (0.5 / args.align_uniform_weight) +
                                    ca_loss(student_output_skt_f_fea, sub_lab_skt, 'skt') * loss_schedule[it] +
                                    ia_loss(student_output_skt_f_fea))
                        else:
                            loss += lsce_loss(student_output_skt_f, sub_lab_skt)
                            if args.use_align_uniform:
                                loss += (unif_loss(student_output_skt_f_fea) * loss_schedule[it] * (0.5 / args.align_uniform_weight) +
                                         ca_loss(student_output_skt_f_fea, sub_lab_skt, 'skt') * loss_schedule[it])
                        loss /= (grad_acc_steps + grad_acc_steps / args.skt_factor)
                        loss_tmp += loss.item()
                        if fp16_scaler is None:
                            loss.backward()
                        else:
                            fp16_scaler.scale(loss).backward()

                if not math.isfinite(loss.item()):
                    print("Loss is {} at {}th grad_acc at {}th iter of {}th epoch,"
                          " stopping training".format(loss.item(), i, it, epoch), force=True)
                    sys.exit(1)

                # student update
                param_norms = None
                if (i + 1) % grad_acc_steps == 0:
                    if fp16_scaler is None:
                        if args.clip_grad:
                            param_norms = utils.clip_gradients(student, args.clip_grad)
                        utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
                        optimizer.step()
                        optimizer.zero_grad()
                    else:
                        if args.clip_grad:
                            fp16_scaler.unscale_(optimizer)
                            # unscale the gradients of optimizer's assigned params in-place
                            param_norms = utils.clip_gradients(student, args.clip_grad)
                        utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
                        fp16_scaler.step(optimizer)
                        fp16_scaler.update()
                        optimizer.zero_grad()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_tmp)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(w_uni_ca=loss_schedule[it])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum_img=0.9, center_momentum_skt=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum_img = center_momentum_img
        self.center_momentum_skt = center_momentum_skt
        self.ncrops = ncrops
        # TODO: Calculate offline centers for sketches and images
        self.register_buffer("center_skt", torch.zeros(1, out_dim))
        self.register_buffer("center_img", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch, modality='img'):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        if modality == 'img':
            teacher_out = F.softmax((teacher_output - self.center_img) / temp, dim=-1)
        elif modality == 'skt':
            teacher_out = F.softmax((teacher_output - self.center_skt) / temp, dim=-1)
        else:
            teacher_out = F.softmax(teacher_output / temp, dim=-1)
            print('no specified modality, withdraw centering!')
        teacher_out = teacher_out.detach().chunk(self.ncrops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq and self.ncrops > 1:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output, modality)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output, modality='img'):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        if modality == 'img':
            self.center_img = self.center_img * self.center_momentum_img + batch_center * (1 - self.center_momentum_img)
        elif modality == 'skt':
            self.center_skt = self.center_skt * self.center_momentum_img + batch_center * (1 - self.center_momentum_skt)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number):
        '''
        :param global_crops_scale: (0.4, 1.0) or (0.14, 1.0)
        :param local_crops_scale:  (0.05, 0.4)
        :param global_crops_number: 1 or 2
        :param local_crops_number: 0 or others
        '''
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_crops_number = global_crops_number
        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        for i in range(self.global_crops_number):
            crops.append(eval('self.global_transfo' + str(i + 1) + '(image)'))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.dataset+'_'+args.split)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
