import os
import sys
import pickle
import argparse

import glob
import torch
from torch import nn
import torch.utils.data
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import models as torchvision_models
from torchvision import transforms as pth_transforms
import numpy as np
import time
import datetime
import hashlib

import utils
import vision_transformer as vits
from dataset import MyImageFolder
from metric import sake_metric, MAP

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, len(data_loader) // 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples).clone()
        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            print(feats.shape)
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


def read_classes(path):
    with open(path) as fp:
        classes = fp.read().splitlines()
        classes = sorted(classes)
    return classes


class ReturnIndexDataset(MyImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


def load_features(features):
    test_features_img = torch.from_numpy(features['test_features_img'])
    test_features_skt = torch.from_numpy(features['test_features_skt'])
    test_labels_img = torch.from_numpy(features['test_labels_img'])
    test_labels_skt = torch.from_numpy(features['test_labels_skt'])

    return test_features_img, test_features_skt, test_labels_img, test_labels_skt


@torch.no_grad()
def sbir_retrieval(model, args):
    start_r = time.time()
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize((args.imsize, args.imsize)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    all_classes = read_classes(os.path.join(args.data_path, args.dataset, 'split', 'all_classes.txt'))
    test_classes = read_classes(os.path.join(args.data_path, args.dataset, 'split', args.split + '_classes.txt'))
    if args.use_train:
        train_classes = np.setdiff1d(all_classes, test_classes)
        train_classes = sorted(train_classes)
        test_classes = train_classes

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
    dataset_test_skt = ReturnIndexDataset(os.path.join(args.data_path, args.dataset, 'sketch'), test_classes,
                                          outlier=outliers,
                                          transform=transform)
    dataset_test_img = ReturnIndexDataset(os.path.join(args.data_path, args.dataset, 'photo'), test_classes,
                                          outlier=None,
                                          transform=transform)
    data_loader_test_skt = torch.utils.data.DataLoader(
        dataset_test_skt,
        batch_size=64,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_test_img = torch.utils.data.DataLoader(
        dataset_test_img,
        batch_size=64,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(
        f" test: {len(dataset_test_img)} imgs {len(dataset_test_skt)} skts {len(test_classes)} test classes")

    ############################################################################
    # Step 1: extract features
    test_features_img = extract_features(model, data_loader_test_img, args.use_cuda, multiscale=args.multiscale).cpu()
    test_features_skt = extract_features(model, data_loader_test_skt, args.use_cuda, multiscale=args.multiscale).cpu()

    # print(test_features_img.shape, test_features_skt.shape)

    test_labels_img = torch.tensor([s[-1] for s in dataset_test_img.samples]).long().cpu()
    test_labels_skt = torch.tensor([s[-1] for s in dataset_test_skt.samples]).long().cpu()

    with open(os.path.join(args.output_dir, "features.pkl"), 'wb') as f:
        pickle.dump({'test_features_img': test_features_img.numpy(),
                     'test_features_skt': test_features_skt.numpy(),
                     'test_labels_img': test_labels_img.numpy(),
                     'test_labels_skt': test_labels_skt.numpy()}, f)

    # evaluation
    return_dict = {}
    if utils.get_rank() == 0:  # only rank 0 will work from now on
        if not args.intra_modal:
            # normalize features
            test_features_img = nn.functional.normalize(test_features_img, dim=1, p=2)
            test_features_skt = nn.functional.normalize(test_features_skt, dim=1, p=2)

            ############################################################################
            # Step 2: similarity
            sim = torch.mm(test_features_skt, test_features_img.T)
            k = {'map': args.mapk, 'precision': args.preck}

            ############################################################################
            # Step 3: evaluate
            start = time.time()
            mAP, precision, similarity, str_sim = sake_metric(test_features_img.numpy(),
                                                              test_labels_img.numpy(),
                                                              test_features_skt.numpy(),
                                                              test_labels_skt.numpy(), sim.numpy(), k)
            print('map{}: {:.4f} prec{}: {:.4f}'.format(k['map'], np.mean(mAP), k['precision'], precision))
            print('using time: ', time.time() - start)
            return_dict['s2i_prec@' + str(k['precision'])] = precision.item()
        else:
            test_features_img = nn.functional.normalize(test_features_img, dim=1, p=2)
            ############################################################################
            # Step 2: similarity for images
            sim = torch.mm(test_features_img, test_features_img.T)
            k = {'map': args.mapk, 'precision': args.preck}

            ############################################################################
            # Step 3: evaluate for images
            start = time.time()
            mAP, precision, similarity, str_sim = sake_metric(test_features_img.numpy(),
                                                              test_labels_img.numpy(),
                                                              test_features_img.numpy(),
                                                              test_labels_img.numpy(),
                                                              sim.numpy(), k)

            print('map{}: {:.4f} prec{}: {:.4f}'.format(k['map'], np.mean(mAP), k['precision'], precision))
            print('using time: ', time.time() - start)
            return_dict['i2i_prec@' + str(k['precision'])] = precision.item()

            ############################################################################
            # Step 2: similarity for sketches
            test_features_skt = nn.functional.normalize(test_features_skt, dim=1, p=2)
            sim = torch.mm(test_features_skt, test_features_skt.T)
            k = {'map': args.mapk, 'precision': args.preck}

            ############################################################################
            # Step 3: evaluate for sketches
            start = time.time()
            mAP, precision, similarity, str_sim = sake_metric(test_features_skt.numpy(),
                                                              test_labels_skt.numpy(),
                                                              test_features_skt.numpy(),
                                                              test_labels_skt.numpy(),
                                                              sim.numpy(), k)
            print('map{}: {:.4f} prec{}: {:.4f}'.format(k['map'], np.mean(mAP), k['precision'], precision))
            print('using time: ', time.time() - start)
            return_dict['s2s_prec@' + str(k['precision'])] = precision.item()
    print('retrieval done in {} seconds'.format(time.time() - start_r))
    return return_dict


if __name__ == '__main__':
    nowTime = datetime.datetime.now().strftime('%m-%d-%H-%M')
    logfilename = "../log/train_log_file-" + nowTime + ".txt"
    errfilename = '../log/train_err_file-' + nowTime + '.txt'
    sys.stdout = utils.LoggerTxt(logfilename)
    sys.stderr = utils.LoggerTxt(errfilename)

    parser = argparse.ArgumentParser('Image Retrieval on revisited Paris and Oxford')
    parser.add_argument('--data_path', default='../dataset', type=str)
    parser.add_argument('--dataset', default='sketchy', type=str, choices=['sketchy', 'tuberlin', 'quickdraw'])
    parser.add_argument('--split', default='zeroshot', type=str, choices=['zeroshot', 'random'])
    parser.add_argument('--multiscale', default=False, type=utils.bool_flag)
    parser.add_argument('--imsize', default=224, type=int, help='Image size')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag)
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture',
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                                + torchvision_archs + torch.hub.list("facebookresearch/xcit"))
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--output_dir', default="../output", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--mapk', default=None, type=int, help='mAP@k')
    parser.add_argument('--preck', default=None, type=int, help='prec@k')
    parser.add_argument('--intra_modal', default=False, type=utils.bool_flag, help='intra-modal retrieval')
    parser.add_argument('--use_train', default=False, type=utils.bool_flag, help='use train split to do retrieve')
    parser.add_argument('--check_sketch_dino', default=False, type=utils.bool_flag, help='evaluate sketch_dino model')
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.dataset + '_' + args.split)
    utils.init_distributed_mode(args)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        # reset fc as identity for resnet50
        model.fc = nn.Identity()
        print(model)
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    if args.use_cuda:
        model.cuda()
    model.eval()

    # load pretrained weights
    if args.pretrained_weights == '':
        if args.arch == "vit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain.pth"
        elif args.arch == "vit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain.pth"
        elif args.arch == "xcit_small_12_p16":
            url = "dino_xcit_small_12_p16_pretrain.pth"
        elif args.arch == "xcit_small_12_p8":
            url = "dino_xcit_small_12_p8_pretrain.pth"
        elif args.arch == "xcit_medium_24_p16":
            url = "dino_xcit_medium_24_p16_pretrain.pth"
        elif args.arch == "xcit_medium_24_p8":
            url = "dino_xcit_medium_24_p8_pretrain.pth"
        elif args.arch == "resnet50":
            url = "dino_resnet50_pretrain.pth"
        else:
            url = 'fadagagasdgadsgadsagsd'
        if os.path.exists(os.path.join('../pretrained_ViT/DINO', url)):
            args.pretrained_weights = os.path.join('../pretrained_ViT/DINO', url)
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size,
                                  args.landmark)
    # do sbir retrieval
    if not args.check_sketch_dino:
        sbir_retrieval(model, args, only_prec=False, use_sake=False)
    else:
        md5 = hashlib.md5()
        p_log = []
        ckp_names = sorted(glob.glob(os.path.join(args.output_dir, 'checkpoint-*.pth')),
                           key=lambda name: int(name[-8:-4]))
        start_idx = -1
        dict_args = vars(args)
        unique_config = ''
        for i in sorted(dict_args.keys()):
            unique_config += (str(i) + str(dict_args[i]))
        md5.update(unique_config.encode('utf-8'))
        unique_config = md5.hexdigest()

        if os.path.isfile(os.path.join(args.output_dir, 'skt_dino_log.pkl')):
            with open(os.path.join(args.output_dir, 'skt_dino_log.pkl'), 'rb') as f:
                p_log = pickle.load(f)
                same_config_log = [i for i in p_log if i['unique_config'] == unique_config]
                same_config_log = sorted(same_config_log, key=lambda x: x['model_name'])
                last_check_model = same_config_log[-1]['mode_name']
                start_idx = ckp_names.index(last_check_model)
        for ckp_name in ckp_names[start_idx + 1:]:
            utils.load_pretrained_weights(model, ckp_name, args.checkpoint_key, args.arch,
                                          args.patch_size,
                                          args.landmark)
            return_dict = sbir_retrieval(model, args, only_prec=True)
            return_dict2 = {**dict_args, **return_dict}
            return_dict2['unique_config'] = unique_config
            return_dict2['model_name'] = ckp_name.split('/')[-1]
            p_log.append(return_dict2)
            with open(os.path.join(args.output_dir, 'skt_dino_log.pkl'), 'wb') as f:
                pickle.dump(p_log, f)

    dist.barrier()
