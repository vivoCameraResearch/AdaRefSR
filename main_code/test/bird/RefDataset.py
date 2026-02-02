import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
from datsr.utils import FileClient
from omegaconf import OmegaConf

import torch
import torchvision.transforms as T
import torch.utils.data as data
from torchvision import transforms
from datasets import load_dataset

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dino_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

from torchvision.transforms.functional import to_pil_image

class RefTestDataset(data.Dataset):
    def __init__(self, opt):
        super(RefTestDataset, self).__init__()
        
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
    
        self.data_dir = opt["data_dir"]
        # load json file
        
        
        self.dataset = load_dataset(
            'json',                    # 数据格式为 JSON
            data_files=self.data_dir,      # 所有 .json 文件路径
            cache_dir='./cache',        # 设置缓存目录
            split='train'               # 使用默认训练数据
        )
        

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        item = self.dataset[index]
        
        lr_path = item["lq_path"]
        ref_path = item["ref_path"]
        gt_path = item["gt_path"]
        image_name = os.path.basename(lr_path)


        # 处理lr image
        img_lr = Image.open(lr_path).convert("RGB")
        img_ref = Image.open(ref_path).convert("RGB")
        img_gt = Image.open(gt_path).convert("RGB")
        
        target_width, target_height = img_gt.size
        # resize
        img_lr = img_lr.resize((self.process_size, self.process_size))
        img_ref = img_ref.resize((self.process_size, self.process_size))

        img_lr = tensor_transforms(img_lr).unsqueeze(0)
        img_ref = tensor_transforms(img_ref).unsqueeze(0)
        
        img_lr_ram = ram_transforms(img_lr)[0]
        img_ref_ram = ram_transforms(img_ref)[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = tensor_transforms(img_gt)
        img_lr = img_lr.squeeze(0)
        img_ref = img_ref.squeeze(0)
        
        
        # to tensor and input 
        img_lr = torch.clamp(img_lr, 0, 1.0)
        img_ref = torch.clamp(img_ref, 0, 1.0)
        
        return_d = {'lr': img_lr, 'ref': img_ref, 'gt': img_gt, 'lr_path': lr_path, 'lr_name': image_name, "height": target_height, "width": target_width, 'lr_ram': img_lr_ram, 'ref_ram': img_ref_ram}
        
        return return_d

    def __len__(self):
        return len(self.dataset)

