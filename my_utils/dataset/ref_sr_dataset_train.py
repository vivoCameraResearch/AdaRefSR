import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F

# 将项目的根目录加入到 Python 的搜索路径
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
# --------------- added -----------------------------

from my_utils.dataset.realesrgan import RealESRGAN_degradation
from datasets import load_dataset
import random
from basicsr.data.transforms import augment
from basicsr.utils import DiffJPEG, USMSharp, img2tensor, tensor2img
from my_utils.dataset.utils.reference_pairs_utils import choose_area_crop_size, random_crop_area, random_crop_just, crop_images, random_crop_
import json

from torchvision import transforms
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


class ReferenceOnlineContrastDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None, opt=None):
        super().__init__()
        self.args = args
        self.split = split
        self.opt = opt
        self.data_dirs = opt["data_dir"]    # lsdir and other
        self.crop_pad_size = opt["crop_pad_size"]
        self.neg_prob = opt.get("neg_prob", 0.0)
        self.other_reference_ratio = opt.get("other_reference_ratio", 0.0)
        
        face_num = opt["face_num"]
        div_num = opt["div_num"]

        if split == 'train':
            self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')

        self.dataset = []
        self.ref_list = []
        self.face_ref_list = []
        for data_txt in self.data_dirs:
            if "CelebRef" in data_txt:
                with open(data_txt, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    random.seed(42)
                    random.shuffle(data)  # 打乱顺序
                    data = data[:face_num]  # 截取指定数量

                    for item in data:
                        self.dataset.append(item["gt_path"])
                        self.ref_list.append(item["ref_path"])
                        self.face_ref_list.append(item["ref_path"])
            else:
                with open(data_txt, "r", encoding="utf-8") as f:
                    data = [line for line in f.readlines()]
                    data = data[:div_num]
                    for line in data:
                        self.dataset.append(line.strip())
                        self.ref_list.append(None)
                        
    def __len__(self):
        return len(self.dataset)

    def random_exclude(self, low, high, exclude):
        choices = np.delete(np.arange(low, high), exclude)
        return int(np.random.choice(choices))

    # crop 
    def crop_and_preprocess(self, img_np):
        h, w = img_np.shape[1:]
        area_size, crop_size, ratio = choose_area_crop_size(h, w)
        if ratio > 0:
            img_gt, img_ref = random_crop_area(img_np, area_size, crop_size)
        else:
            img_gt, img_ref = random_crop_just(img_np, 512)
        img_gt, img_ref = img_gt.transpose(1, 2, 0) / 255.0, img_ref.transpose(1, 2, 0) / 255.0
        img_gt, img_ref = crop_images(img_gt, img_ref, self.crop_pad_size, bilinear=True)
        return img_gt, img_ref

    def __getitem__(self, idx):
        item_path = self.dataset[idx]   # self.dataset -> image path or reference given
        item_ref = self.ref_list[idx]
        

        ori_img = Image.open(item_path).convert('RGB')
        ori_np = np.array(ori_img).transpose(2, 0, 1)

        # 负样本替换 reference
        random_choice = random.random()

        if item_ref == None:    # for dataset DIV2K...
            img_gt, img_ref = self.crop_and_preprocess(ori_np)  # get crop image from one input
            if random_choice < self.neg_prob: # 0.1
                neg_idx = self.random_exclude(0, len(self.dataset), idx)
                neg_path = self.dataset[neg_idx]
                neg_img = Image.open(neg_path).convert('RGB')
                neg_np = np.array(neg_img).transpose(2, 0, 1)
                _, img_ref = self.crop_and_preprocess(neg_np)

            # 注入其他参考图像
            elif random_choice < self.neg_prob + self.other_reference_ratio:  # 0.1
                rand_idx = self.random_exclude(0, len(self.dataset), idx)
                other_img = Image.open(self.dataset[rand_idx]).convert('RGB')
                other_np = np.array(other_img).transpose(2, 0, 1)
                img_ref = random_crop_(other_np, crop_size=self.crop_pad_size)
                img_ref = img_ref.transpose(1, 2, 0).astype(np.float32) / 255.0

        else:
            # 选择reference图像
            is_retrieval = False
            img_ref_path = item_ref
            
            img_gt = ori_np
            img_ref = Image.open(img_ref_path).convert('RGB')
            img_ref = np.array(img_ref).transpose(2, 0, 1)
            img_gt, img_ref = img_gt.transpose(1, 2, 0) / 255.0, img_ref.transpose(1, 2, 0) / 255.0
            img_gt, img_ref = crop_images(img_gt, img_ref, self.crop_pad_size, bilinear=True)

        # 数据增强
        img_ref, _ = augment(img_ref, hflip=True, rotation=False, return_status=True)
        # 模拟退化过程
        img_gt, img_lq = self.degradation.degrade_process(img_gt, resize_bak=True)  # to tensor [0, 1]! b, c, h, w
        img_ref = img2tensor([img_ref], bgr2rgb=False, float32=True)[0]    # b, c, h, w
        img_gt, img_lq = img_gt.squeeze(0), img_lq.squeeze(0)

        
        # get ram model input
        img_lq_ram = ram_transforms(img_lq)
        img_gt_ram = ram_transforms(img_gt) # ram 接受的输入是[0, 1]之间
        img_ref_ram = ram_transforms(img_ref)
        

        img_lq = F.normalize(img_lq, mean=[0.5], std=[0.5]).squeeze(0)
        img_gt = F.normalize(img_gt, mean=[0.5], std=[0.5]).squeeze(0)
        img_ref = F.normalize(img_ref, mean=[0.5], std=[0.5]).squeeze(0)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'ref': img_ref,
            'img_path': item_path,
            'lq_ram': img_lq_ram, 
            'gt_ram': img_gt_ram,
            'ref_ram': img_ref_ram,
        }
        


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from my_utils.training_utils import parse_args_paired_training, degradation_proc, degradation_proc_valid

    # args = parse_args_paired_training()
    
    args = {}
    
    deg_file_path = "params_realesrgan.yml"
    args["deg_file_path"] = deg_file_path
    base_config = "/home/u2120220604/wangyuan/code/retrieval/s3diff-pipeline/configs/src_2/train_one_step_sr_reference_online_lsdir.yaml"
    config = OmegaConf.load(base_config)
    
    dataset = ReferenceOnlineContrastDataset(opt = config.train, args = args, split = 'train')
    
    
    for idx in range(len(dataset)):
        item = dataset[idx]
        lq = item["lq"]
        ref = item["ref"]
        lq_ram = item["lq_ram"]
        ref_ram = item["ref_ram"]
        
        print(lq_ram.shape)
        print(lq_ram.max())
        
    