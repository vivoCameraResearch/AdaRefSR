import torch.utils.data as data
from PIL import Image

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datsr.utils import FileClient
from omegaconf import OmegaConf

import torch
import torchvision.transforms as T
from torchvision import transforms
tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

class CUFED5ValidDataset(data.Dataset):
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
        super(CUFED5ValidDataset, self).__init__()
        
        self.opt = opt  # 输入是一个config
        self.io_backend_opt = opt['io_backend']
        self.file_client = None

        if 'crop_size' in opt:
            self.process_size = opt['crop_size']
        else:
            self.process_size = 512
        if 'image_type' not in opt:
            opt['image_type'] = 'png'
            
        self.rscale = opt['sf']
        

    
        self.ref_dir = opt["ref_dir"]
        self.lq_dir = opt["lq_dir"]
        self.gt_dir = opt["gt_dir"]
        
        self.image_names = os.listdir(self.lq_dir)
        

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)


        image_name = self.image_names[index]
        
        img_id = os.path.basename(image_name).split('_')[0]
        lr_path = os.path.join(self.lq_dir, image_name)
        
        
        gt_path = os.path.join(self.gt_dir, img_id+"0_HR.png")
        
        # 处理lr image
        img_lr = Image.open(lr_path).convert("RGB")
        ori_width, ori_height = img_lr.size
        rscale = self.rscale     
            
        if ori_width < self.process_size//rscale or ori_height < self.process_size//rscale:
            scale = (self.process_size//rscale)/min(ori_width, ori_height)
            tmp_image = img_lr.resize((int(scale*ori_width), int(scale*ori_height)))
            img_lr = tmp_image
            
        img_lr = img_lr.resize((img_lr.size[0]*rscale, img_lr.size[1]*rscale)) 
        img_lr = img_lr.resize((img_lr.size[0]//8*8, img_lr.size[1]//8*8))
        width, height = img_lr.size # 没有经过padding的size！！！

        # ----------------- 处理reference image and gt_image --------------------------------
        if 'cufed5' in lr_path:
            ref_im_path = os.path.join(self.ref_dir, img_id + "_1.png")
            img_ref = Image.open(ref_im_path).convert("RGB")
            
            gt_path = os.path.join(self.gt_dir, img_id + "_0_HR.png")
            img_gt = Image.open(gt_path).convert("RGB")
        else:
            raise NotImplementedError

        ref_width, ref_height = img_ref.size
        img_ref = img_ref.resize((img_ref.size[0]//8*8, img_ref.size[1]//8*8))
        # todo：合并reference和LR image到同样的尺寸，方便后续处理
        img_lr = tensor_transforms(img_lr)
        img_ref = tensor_transforms(img_ref)
        ref_height, ref_width = img_ref.shape[-2:]   # 没有经过padding的size！

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = tensor_transforms(img_gt)
        img_lr = img_lr
        img_ref = img_ref
        # to tensor and input 
        img_lr = torch.clamp(img_lr, 0, 1.0)
        img_ref = torch.clamp(img_ref, 0, 1.0)

        # height与width
        return_d = {'lr': img_lr, 'ref': img_ref, 'gt': img_gt, 'lr_path': lr_path, 'lr_name': image_name, "height": height, "width": width, "ref_height": ref_height, "ref_width": ref_width}
        return return_d

    def __len__(self):
        return len(self.image_names)
