import argparse
import json
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from glob import glob

import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop, triplet_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, random_add_speckle_noise_pt, random_add_saltpepper_noise_pt, bivariate_Gaussian

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

from datsr.data.transforms import augment, mod_crop, totensor, random_crop


def parse_args_paired_testing(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="./configs/sr_test.yaml", type=str)
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo", help="The name of the wandb project to log to.")

    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=8, type=int)
    parser.add_argument("--lora_rank_vae", default=8, type=int)

    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--chop_size", type=int, default=128, choices=[512, 256, 128], help="Chopping forward.")
    parser.add_argument("--chop_stride", type=int, default=96, help="Chopping stride.")
    parser.add_argument("--padding_offset", type=int, default=32, help="padding offset.")

    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) 
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)

    parser.add_argument("--align_method", type=str, default="wavelet")
    
    parser.add_argument("--pos_prompt", type=str, default="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.")
    parser.add_argument("--neg_prompt", type=str, default="oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth")

    # training details
    parser.add_argument("--output_dir", type=str, default='output/')
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    # --------------------------------------------------------------------------------
    parser.add_argument("--is_image", default=True)
    parser.add_argument("--fusion_blocks", default="full")
    parser.add_argument("--save_lr_pred_ref", default = True)

    # details about the model architecture
    parser.add_argument("--sd_path", default = "your_code_path/model/stabilityai/sd-turbo")
    parser.add_argument("--de_net_path", default = "your_code_path/assets/mm-realsr/de_net.pth", type=str)
    parser.add_argument("--pretrained_backbone_path", type=str, default="your_code_path/model/zhangap/S3Diff/s3diff.pkl")
    parser.add_argument("--pretrained_ref_gen_path", default = "your_code_path/model/adarefsr.pkl", type=str)
    parser.add_argument("--dape_path", default = "your_code_path/model/DAPE/DAPE.pth", type=str)
    parser.add_argument("--ram_path", default = "your_code_path/model/xinyu1205/recognize-anything/ram_swin_large_14m.pth", type=str)
    
    parser.add_argument("--proj_name", default = "reference-sr")
    parser.add_argument("--exp_name", default = "version-xxx")
    parser.add_argument("--deg_file_path", default="params_realesrgan.yml", type=str)
    
    parser.add_argument("--num_references", default=1, type=int)
    parser.add_argument("--learnable_tokens", default=16, type=int)
    
    parser.add_argument("--test_resolution", default = 512, type=int)
    parser.add_argument("--test_warmup_step", default=10, type=int)
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


class PlainDataset(data.Dataset):
    """Modified dataset based on the dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(PlainDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        if 'image_type' not in opt:
            opt['image_type'] = 'png'

        # support multiple type of data: file path and meta data, remove support of lmdb
        self.lr_paths = []
        if 'lr_path' in opt:
            if isinstance(opt['lr_path'], str):
                self.lr_paths.extend(sorted(
                    [str(x) for x in Path(opt['lr_path']).glob('*.png')] +
                    [str(x) for x in Path(opt['lr_path']).glob('*.jpg')] +
                    [str(x) for x in Path(opt['lr_path']).glob('*.jpeg')]
                ))
            else:
                self.lr_paths.extend(sorted([str(x) for x in Path(opt['lr_path'][0]).glob('*.'+opt['image_type'])]))
                if len(opt['lr_path']) > 1:
                    for i in range(len(opt['lr_path'])-1):
                        self.lr_paths.extend(sorted([str(x) for x in Path(opt['lr_path'][i+1]).glob('*.'+opt['image_type'])]))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        lr_path = self.lr_paths[index]

        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                lr_img_bytes = self.file_client.get(lr_path, 'gt')
            except (IOError, OSError) as e:
                # logger = get_root_logger()
                # logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__()-1)
                lr_path = self.lr_paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1

        img_lr = imfrombytes(lr_img_bytes, float32=True)
        
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lr = img2tensor([img_lr], bgr2rgb=True, float32=True)[0]

        return_d = {'lr': img_lr, 'lr_path': lr_path}
        return return_d

    def __len__(self):
        return len(self.lr_paths)


def lr_proc(config, batch, device):
    im_lr = batch['lr'].cuda()
    im_lr = im_lr.to(memory_format=torch.contiguous_format).float()    

    ori_lr = im_lr

    im_lr = F.interpolate(
            im_lr,
            size=(im_lr.size(-2) * config.sf,
                  im_lr.size(-1) * config.sf),
            mode='bicubic',
            )

    im_lr = im_lr.contiguous() 
    im_lr = im_lr * 2 - 1.0
    im_lr = torch.clamp(im_lr, -1.0, 1.0)

    ori_h, ori_w = im_lr.size(-2), im_lr.size(-1)

    pad_h = (math.ceil(ori_h / 64)) * 64 - ori_h
    pad_w = (math.ceil(ori_w / 64)) * 64 - ori_w
    im_lr = F.pad(im_lr, pad=(0, pad_w, 0, pad_h), mode='reflect')

    return im_lr.to(device), ori_lr.to(device), (ori_h, ori_w)


# todo 参考reference图像生成的测试代码实现
def lr_proc_with_reference(config, batch, device):
    img_in = batch['gt']    # in [0, 1] # 注意输入需要是BGR
    img_ref = batch['ref']
    
    # 转化为numpy
    img_in = img_in.squeeze(axis = 0).permute(1, 2, 0) if isinstance(img_in, torch.Tensor) else img_in  # to h, w, c 
    img_ref = img_ref.squeeze(axis = 0).permute(1, 2, 0) if isinstance(img_ref, torch.Tensor) else img_ref  # to h, w, c
    
    img_in = img_in.numpy() if isinstance(img_in, torch.Tensor) else img_in
    img_ref = img_ref.numpy() if isinstance(img_ref, torch.Tensor) else img_ref
    
    # 输入需要是BGR三通道
    img_in = img_in[:,:,::-1]
    img_ref = img_ref[:,:,::-1]
    
    scale = config['scale']
    
    print(img_in.shape)
    gt_h, gt_w, _ = img_in.shape
    # 
    img_in_pil = img_in * 255
    img_in_pil = Image.fromarray(
        cv2.cvtColor(img_in_pil.astype(np.uint8), cv2.COLOR_BGR2RGB))
    
    
    # todo here to degration
    # downsample image using PIL bicubic kernel
    lq_h, lq_w = gt_h // scale, gt_w // scale
    img_in_lq = img_in_pil.resize((lq_w, lq_h), Image.BICUBIC)

    # bicubic upsample LR
    img_in_up = img_in_lq.resize((gt_w, gt_h), Image.BICUBIC)

    img_in_lq = cv2.cvtColor(np.array(img_in_lq), cv2.COLOR_RGB2BGR)
    img_in_lq = img_in_lq.astype(np.float32) / 255.
    img_in_up = cv2.cvtColor(np.array(img_in_up), cv2.COLOR_RGB2BGR)
    img_in_up = img_in_up.astype(np.float32) / 255.
    
    # BGR to RGB, HWC to CHW, numpy to tensor
    img_in, img_in_lq, img_in_up, img_ref = totensor(  # noqa: E501
        [img_in, img_in_lq, img_in_up, img_ref],
        bgr2rgb=True,   # 已经是rgb了 
        float32=True)
    
    return_dict = { # [0, 1]区间，注意模型的输入是在[-1, 1]区间，因此还需要标准化一下
        'img_in': img_in,
        'img_in_lq': img_in_lq,
        'img_in_up': img_in_up,
        'img_ref': img_ref,
    }

    return return_dict



def lr_degration_proc_with_reference(config, batch, device):
    
    img_in_up, img_in, img_in_lq  = degradation_proc(config, batch, device)   # lq.to(device), gt.to(device), ori_lq.to(device)
    
    # img_in = batch['gt']    # in [0, 1] # 注意输入需要是BGR
    img_ref = batch['ref']
    
    
    # todo here to degration
    
    
    # BGR to RGB, HWC to CHW, numpy to tensor
    img_in, img_in_lq, img_in_up, img_ref = totensor(  # noqa: E501
        [img_in, img_in_lq, img_in_up, img_ref],
        bgr2rgb=True,   # 已经是rgb了 
        float32=True)
    
    return_dict = {
        'img_in': img_in,
        'img_in_lq': img_in_lq,
        'img_in_up': img_in_up,
        'img_ref': img_ref,
    }

    return return_dict



def crop_images(img_gt, img_ref = None, crop_pad_size = 512, bilinear=True):
    """
    对 img_gt 和 img_ref 使用相同的裁剪方法，首先按比例放大到接近目标尺寸，然后中心裁剪。
    
    :param img_gt: 高质量图像，形状为 (H, W, C) 或 (H, W)
    :param img_ref: 参考图像，形状为 (H, W, C) 或 (H, W)
    :param crop_pad_size: 裁剪的目标尺寸
    :param bilinear: 是否使用双线性插值进行缩放
    :return: 裁剪后的 img_gt 和 img_ref
    """
    # 获取图像的高度和宽度
    h_gt, w_gt = img_gt.shape[0:2]
    
    # 先处理 img_gt，按照原逻辑裁剪
    if h_gt > crop_pad_size or w_gt > crop_pad_size:
        top = int((h_gt - crop_pad_size) // 2)
        left = int((w_gt - crop_pad_size) // 2)
        if bilinear:
            img_gt = cv2.resize(img_gt, (crop_pad_size, crop_pad_size), interpolation=cv2.INTER_LINEAR)
        else:
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

    if img_ref is not None:    
        h_ref, w_ref = img_ref.shape[0:2]

        # 计算 img_ref 缩放比例 r
        if h_ref > w_ref:
            new_w = 512
            new_h = int(512 * (h_ref / w_ref))

            # 缩放 img_ref
            img_ref_resized = cv2.resize(img_ref, (new_w, new_h), interpolation=cv2.INTER_LINEAR if bilinear else cv2.INTER_NEAREST)
            
        else:
            new_h = 512
            new_w = int(512 * (w_ref / h_ref))
            # 缩放 img_ref
            img_ref_resized = cv2.resize(img_ref, (new_w, new_h), interpolation=cv2.INTER_LINEAR if bilinear else cv2.INTER_NEAREST)

        # 计算裁剪区域，进行 center crop
        top_ref = (new_h - crop_pad_size) // 2
        left_ref = (new_w - crop_pad_size) // 2
        img_ref = img_ref_resized[top_ref:top_ref + crop_pad_size, left_ref:left_ref + crop_pad_size, ...]

        return img_gt, img_ref
    
    return img_gt

def degradation_proc(configs, batch, device, val=False, use_usm=False, resize_lq=True, random_size=False):
    
    """Degradation pipeline, modified from Real-ESRGAN:
    https://github.com/xinntao/Real-ESRGAN
    """

    jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
    usm_sharpener = USMSharp().cuda()  # do usm sharpening

    im_gt = batch['gt'].cuda()
    if use_usm:
        im_gt = usm_sharpener(im_gt)
    im_gt = im_gt.to(memory_format=torch.contiguous_format).float()
    kernel1 = batch['kernel1'].cuda()
    kernel2 = batch['kernel2'].cuda()
    sinc_kernel = batch['sinc_kernel'].cuda()

    ori_h, ori_w = im_gt.size()[2:4]

    # ----------------------- The first degradation process ----------------------- #
    # blur
    out = filter2D(im_gt, kernel1)
    # random resize
    updown_type = random.choices(
            ['up', 'down', 'keep'],
            configs.degradation['resize_prob'],
            )[0]
    if updown_type == 'up':
        scale = random.uniform(1, configs.degradation['resize_range'][1])
    elif updown_type == 'down':
        scale = random.uniform(configs.degradation['resize_range'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scale, mode=mode)
    # add noise
    gray_noise_prob = configs.degradation['gray_noise_prob']
    if random.random() < configs.degradation['gaussian_noise_prob']:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=configs.degradation['noise_range'],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob,
            )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=configs.degradation['poisson_scale_range'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)
    # JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*configs.degradation['jpeg_range'])
    out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
    out = jpeger(out, quality=jpeg_p)

    # ----------------------- The second degradation process ----------------------- #
    # blur
    if random.random() < configs.degradation['second_blur_prob']:
        out = filter2D(out, kernel2)
    # random resize
    updown_type = random.choices(
            ['up', 'down', 'keep'],
            configs.degradation['resize_prob2'],
            )[0]
    if updown_type == 'up':
        scale = random.uniform(1, configs.degradation['resize_range2'][1])
    elif updown_type == 'down':
        scale = random.uniform(configs.degradation['resize_range2'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(
            out,
            size=(int(ori_h / configs.sf * scale),
                  int(ori_w / configs.sf * scale)),
            mode=mode,
            )
    # add noise
    gray_noise_prob = configs.degradation['gray_noise_prob2']
    if random.random() < configs.degradation['gaussian_noise_prob2']:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=configs.degradation['noise_range2'],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob,
            )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=configs.degradation['poisson_scale_range2'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False,
            )

    # JPEG compression + the final sinc filter
    # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
    # as one operation.
    # We consider two orders:
    #   1. [resize back + sinc filter] + JPEG compression
    #   2. JPEG compression + [resize back + sinc filter]
    # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
    if random.random() < 0.5:
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out,
                size=(ori_h // configs.sf,
                      ori_w // configs.sf),
                mode=mode,
                )
        out = filter2D(out, sinc_kernel)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*configs.degradation['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
    else:
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*configs.degradation['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out,
                size=(ori_h // configs.sf,  # todo 下采样构造LQ数据的方法
                      ori_w // configs.sf),
                mode=mode,
                )
        out = filter2D(out, sinc_kernel)

    # clamp and round
    im_lq = torch.clamp(out, 0, 1.0)

    # random crop
    gt_size = configs.degradation['gt_size']
    im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, configs.sf)
    lq, gt = im_lq, im_gt
    ori_lq = im_lq

    if resize_lq:
        lq = F.interpolate(
                lq,
                size=(gt.size(-2),
                      gt.size(-1)),
                mode='bicubic',
                )

    if random.random() < configs.degradation['no_degradation_prob'] or torch.isnan(lq).any():
        lq = gt

    # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
    lq = lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
    lq = lq * 2 - 1.0 # TODO 0~1?
    gt = gt * 2 - 1.0

    if random_size:
        lq, gt = randn_cropinput(lq, gt)

    lq = torch.clamp(lq, -1.0, 1.0)

    return lq.to(device), gt.to(device), ori_lq.to(device)


def randn_cropinput(lq, gt, base_size=[64, 128, 256, 512]):
    cur_size_h = random.choice(base_size)
    cur_size_w = random.choice(base_size)
    init_h = lq.size(-2)//2
    init_w = lq.size(-1)//2
    lq = lq[:, :, init_h-cur_size_h//2:init_h+cur_size_h//2, init_w-cur_size_w//2:init_w+cur_size_w//2]
    gt = gt[:, :, init_h-cur_size_h//2:init_h+cur_size_h//2, init_w-cur_size_w//2:init_w+cur_size_w//2]
    assert lq.size(-1)>=64
    assert lq.size(-2)>=64
    return [lq, gt]