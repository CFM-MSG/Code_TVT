import os
import sys
import time
import math
import random
import shutil
import datetime
import subprocess
from collections import defaultdict, deque

import numpy as np
import torch
from torch import nn
from PIL import Image
import torch.distributed as dist
from PIL import ImageFilter, ImageOps


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size, landmark=False):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if landmark:
            url = "dino_vitsmall16_googlelandmark_pretrain/dino_vitsmall16_googlelandmark_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif model_name == "xcit_small_12_p16":
            url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
        elif model_name == "xcit_small_12_p8":
            url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
        elif model_name == "xcit_medium_24_p16":
            url = "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
        elif model_name == "xcit_medium_24_p8":
            url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
        elif model_name == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")


def load_pretrained_weights_fd(model, pretrained_weights, checkpoint_key):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("There is no reference weights available for this model => We use random weights.")


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        print('=> there is no checkpoint for the specified file')
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def restart_from_checkpoint_fd(ckp_img_path, ckp_skt_path, ckp_stu_path, teacher_img, teacher_skt, student, run_variables=None):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_img_path) or not os.path.isfile(ckp_skt_path) or not os.path.isfile(ckp_stu_path):
        print('=> there is no checkpoint for the specified file')
        return
    print("Found checkpoint at {}, {}, {}".format(ckp_img_path, ckp_skt_path, ckp_stu_path))

    # open checkpoint file
    checkpoint_img = torch.load(ckp_img_path, map_location="cpu")
    checkpoint_skt = torch.load(ckp_skt_path, map_location='cpu')
    checkpoint_stu = torch.load(ckp_stu_path, map_location='cpu')

    # load state_dict for three model
    def load_state_dict(checkpoint, model, ckp_path, ckp_name):
        try:
            if ckp_name == 'student' and (hasattr(model.module, 'head_skt') or hasattr(model.module, 'head_fusion')):
                state_dict = {'module.'+k: v for k, v in checkpoint['teacher'].items()}
                # vit + skt distill token + fusion token
                head_state_dict = {k: v for k, v in state_dict.items() if 'head.' in k}

                model_dict = model.state_dict()
                trash_vars = [k for k in state_dict.keys() if k not in model_dict.keys()]
                print('trashed vars from resume dict:')
                print(trash_vars)

                resume_dict = {k: v for k, v in state_dict.items() if k in model_dict}

                # address head_skt and head_fusion
                for k in head_state_dict.keys():
                    if hasattr(model.module, 'head_skt'):
                        resume_dict[k.replace('head.', 'head_skt.')] = head_state_dict[k]
                    if hasattr(model.module, 'head_fusion'):
                        if 'last_layer' not in k:
                            resume_dict[k.replace('head.', 'head_fusion.')] = head_state_dict[k]
                # address tokens and pos_emb
                cls_token = state_dict['module.backbone.cls_token']
                pos_embed = state_dict['module.backbone.pos_embed']
                resume_dict['module.backbone.dist_img_token'] = cls_token
                if hasattr(model.module.backbone, 'fusion_token'):
                    resume_dict['module.backbone.fusion_token'] = cls_token
                if hasattr(model.module.backbone, 'dist_skt_token'):
                    resume_dict['module.backbone.dist_skt_token'] = cls_token
                extra_pos = []
                for i in range(model.module.backbone.token_num-1):
                    extra_pos.append(pos_embed[:, 0].unsqueeze(0))
                extra_pos.append(pos_embed)
                resume_dict['module.backbone.pos_embed'] = torch.cat(extra_pos, dim=1)
                lost_vars = [k for k in model_dict.keys() if k not in resume_dict.keys()]
                print('lost vars from resume dict:')
                print(lost_vars)
                model_dict.update(resume_dict)
                msg = model.load_state_dict(model_dict)
            else:
                msg = model.load_state_dict(checkpoint['teacher'], strict=False)
            print("=> loaded '{}' from checkpoint '{}' with msg {}".format(ckp_name, ckp_path, msg))
        except TypeError:
            try:
                msg = model.load_state_dict(checkpoint['teacher'])
                print("=> loaded '{}' from checkpoint: '{}'".format(ckp_name, ckp_path))
            except ValueError:
                print("=> failed to load '{}' from checkpoint: '{}'".format(ckp_name, ckp_path))
    load_state_dict(checkpoint_img, teacher_img, ckp_img_path, 'teacher_img')
    load_state_dict(checkpoint_skt, teacher_skt, ckp_skt_path, 'teacher_skt')
    load_state_dict(checkpoint_stu, student, ckp_stu_path, 'student')


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0, early_schedule_epochs=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    if early_schedule_epochs == 0:
        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        schedule = np.concatenate((warmup_schedule, schedule))
    else:
        iters = np.arange(early_schedule_epochs*niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        remainder = np.array([final_value]*((epochs - early_schedule_epochs) * niter_per_ep))
        schedule = np.concatenate((warmup_schedule, schedule, remainder))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def cosine_scheduler_loss(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, early_schedule_epochs=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.array([base_value] * warmup_iters)

    if early_schedule_epochs == 0:
        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        schedule = np.concatenate((warmup_schedule, schedule))
    else:
        iters = np.arange(early_schedule_epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        remainder = np.array([final_value]*((epochs - early_schedule_epochs) * niter_per_ep))
        schedule = np.concatenate((warmup_schedule, schedule, remainder))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, len_iter=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        if len_iter is None:
            len_iter = len(iterable)
        space_fmt = ':' + str(len(str(len_iter))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len_iter - 1:
                eta_seconds = iter_time.global_avg * (len_iter - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len_iter, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len_iter, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len_iter))


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '50000'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)


class FDMultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head, head_skt, head_fusion):
        super(FDMultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.token_num = backbone.token_num
        self.head = head  # default for distill_img_token
        if self.token_num == 3:
            self.head_skt = head_skt
        self.head_fusion = head_fusion


    def forward(self, x, modality='img'):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output_dist, output_fusion = 0, torch.empty(0).to(x[0].device), torch.empty(0).to(x[0].device)
        head_skt = self.head if self.token_num != 3 else self.head_skt
        for end_idx in idx_crops:
            if self.token_num == 3:
                _out_img, _out_skt, _out_fusion = self.backbone(torch.cat(x[start_idx: end_idx]))
            else:
                _out_dist, _out_fusion = self.backbone(torch.cat(x[start_idx: end_idx]))
                _out_img, _out_skt = _out_dist, _out_dist

            # accumulate outputs
            if modality == 'img':
                output_dist = torch.cat((output_dist, _out_img))
            elif modality == 'skt':
                output_dist = torch.cat((output_dist, _out_skt))
            else:
                print('You must specifiy the modality!')
                sys.exit(-1)
            output_fusion = torch.cat((output_fusion, _out_fusion))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        f_cls, f_fea = self.head_fusion(output_fusion, return_both=True)
        if modality == 'img':
            return self.head(output_dist), f_cls, f_fea
        elif modality == 'skt':
            return head_skt(output_dist), f_cls, f_fea
        else:
            print('You must specifiy the modality!')
            sys.exit(-1)

    def features(self, x, modality='img', return_idx=0):
        features = []
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output_dist, output_fusion = 0, torch.empty(0).to(x[0].device), torch.empty(0).to(x[0].device)
        head_skt = self.head if self.token_num != 3 else self.head_skt
        for end_idx in idx_crops:
            if self.token_num == 3:
                _out_img, _out_skt, _out_fusion = self.backbone(torch.cat(x[start_idx: end_idx]))
            else:
                _out_dist, _out_fusion = self.backbone(torch.cat(x[start_idx: end_idx]))
                _out_img, _out_skt = _out_dist, _out_dist

            # accumulate outputs
            if modality == 'img':
                output_dist = torch.cat((output_dist, _out_img))
            elif modality == 'skt':
                output_dist = torch.cat((output_dist, _out_skt))
            else:
                print('You must specifiy the modality!')
                sys.exit(-1)
            output_fusion = torch.cat((output_fusion, _out_fusion))
            start_idx = end_idx
        features.append(output_fusion)
        features.append(output_dist)
        features.append(self.head_fusion(output_fusion, return_fea=True))
        # Run the head forward on the concatenated features.
        if modality == 'img':
            features.append(self.head(output_dist, return_fea=True))
        elif modality == 'skt':
            features.append(head_skt(output_dist, return_fea=True))
        else:
            print('You must specifiy the modality!')
            sys.exit(-1)
        features[1] = (features[0] + features[1]) / 2
        features[3] = (features[3] + features[2]) / 2
        features[3] /= features[3].norm(p=2, dim=1, keepdim=True)
        return features[return_idx]


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def get_params_groups_lr(model):
    slow_fn_regularized = []
    slow_fn_not_regularized = []
    fast_lr_regularized = []
    fast_lr_not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            if name.endswith('.bias') or len(param.shape) == 1:
                slow_fn_not_regularized.append(param)
            else:
                slow_fn_regularized.append(param)
        else:
            if name.endswith('.bias') or len(param.shape) == 1:
                fast_lr_not_regularized.append(param)
            else:
                fast_lr_regularized.append(param)

    return [{'params': slow_fn_regularized}, {'params': slow_fn_not_regularized, 'weight_decay': 0.}, {'params': fast_lr_regularized},
            {'params': fast_lr_not_regularized, 'weight_decay': 0.}]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


class PCA():
    """
    Class to  compute and apply PCA.
    """
    def __init__(self, dim=256, whit=0.5):
        self.dim = dim
        self.whit = whit
        self.mean = None

    def train_pca(self, cov):
        """
        Takes a covariance matrix (np.ndarray) as input.
        """
        d, v = np.linalg.eigh(cov)
        eps = d.max() * 1e-5
        n_0 = (d < eps).sum()
        if n_0 > 0:
            d[d < eps] = eps

        # total energy
        totenergy = d.sum()

        # sort eigenvectors with eigenvalues order
        idx = np.argsort(d)[::-1][:self.dim]
        d = d[idx]
        v = v[:, idx]

        print("keeping %.2f %% of the energy" % (d.sum() / totenergy * 100.0))

        # for the whitening
        d = np.diag(1. / d**self.whit)

        # principal components
        self.dvt = np.dot(d, v.T)

    def apply(self, x):
        # input is from numpy
        if isinstance(x, np.ndarray):
            if self.mean is not None:
                x -= self.mean
            return np.dot(self.dvt, x.T).T

        # input is from torch and is on GPU
        if x.is_cuda:
            if self.mean is not None:
                x -= torch.cuda.FloatTensor(self.mean)
            return torch.mm(torch.cuda.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)

        # input if from torch, on CPU
        if self.mean is not None:
            x -= torch.FloatTensor(self.mean)
        return torch.mm(torch.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)

def multi_scale(samples, model):
    v = None
    for s in [1, 1/2**(1/2), 1/2]:  # we use 3 different scales
        if s == 1:
            inp = samples.clone()
        else:
            inp = nn.functional.interpolate(samples, scale_factor=s, mode='bilinear', align_corners=False)
        feats = model(inp).clone()
        if v is None:
            v = feats
        else:
            v += feats
    v /= 3
    v /= v.norm(p=2, dim=1, keep_dim=True)
    return v


def multi_scale_fd(samples, model, modality, return_idx):
    v = None
    for s in [1, 1/2**(1/2), 1/2]:  # we use 3 different scales
        if s == 1:
            inp = samples.clone()
        else:
            inp = nn.functional.interpolate(samples, scale_factor=s, mode='bilinear', align_corners=False)
        feats = model.features(inp, modality, return_idx).clone()
        if v is None:
            v = feats
        else:
            v += feats
    v /= 3
    v /= v.norm(p=2, dim=1, keepdim=True)
    return v


def reduce_mean(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


def ITQ(V, n_iter):
    # Main function for  ITQ which finds a rotation of the PCA embedded data
    # Input:
    #     V: nxc PCA embedded data, n is the number of images and c is the code length
    #     n_iter: max number of iterations, 50 is usually enough
    # Output:
    #     B: nxc binary matrix
    #     R: the cxc rotation matrix found by ITQ
    # Publications:
    #     Yunchao Gong and Svetlana Lazebnik. Iterative Quantization: A
    #     Procrustes Approach to Learning Binary Codes. In CVPR 2011.
    # Initialize with a orthogonal random rotation initialize with a orthogonal random rotation

    bit = V.shape[1]
    np.random.seed(n_iter)
    R = np.random.randn(bit, bit)
    U11, S2, V2 = np.linalg.svd(R)
    R = U11[:, :bit]

    # ITQ to find optimal rotation
    for iter in range(n_iter):
        Z = np.matmul(V, R)
        UX = np.ones((Z.shape[0], Z.shape[1])) * -1
        UX[Z >= 0] = 1
        C = np.matmul(np.transpose(UX), V)
        UB, sigma, UA = np.linalg.svd(C)
        R = np.matmul(UA, np.transpose(UB))

    # Make B binary
    B = UX
    B[B < 0] = 0

    return B, R


def compressITQ(Xtrain, Xtest, n_iter=50):

    # compressITQ runs ITQ
    # Center the data, VERY IMPORTANT
    Xtrain = Xtrain - np.mean(Xtrain, axis=0, keepdims=True)
    Xtest = Xtest - np.mean(Xtest, axis=0, keepdims=True)

    # PCA
    C = np.cov(Xtrain, rowvar=False)
    l, pc = np.linalg.eigh(C, 'U')
    idx = l.argsort()[::-1]
    idx = idx[:64]
    pc = pc[:, idx]
    XXtrain = np.matmul(Xtrain, pc)
    XXtest = np.matmul(Xtest, pc)

    # ITQ
    _, R = ITQ(XXtrain, n_iter)

    Ctrain = np.matmul(XXtrain, R)
    Ctest = np.matmul(XXtest, R)

    Ctrain = Ctrain > 0
    Ctest = Ctest > 0

    return Ctrain, Ctest

