
import torch
from os import path as osp
import os
import sys
import os.path as osp

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from main_code.test.face.RefDataset import RefTestDataset


from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from main_code.model.gen_model import GenModel, torch_dfs
from main_code.model.ref_model import RefModel

from main_code.model.anymate_anyone.reference_attention import ReferenceNetAttention
from my_utils.wavelet_color import adain_color_fix, wavelet_color_fix


from omegaconf import OmegaConf
import argparse
import transformers

import torch.nn.functional as F


from torchvision.transforms import ToTensor, ToPILImage
from my_utils.testing_utils import parse_args_paired_testing
from main_code.model.de_net import DEResNet
from accelerate import Accelerator


import torch

from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention import BasicTransformerBlock as _BasicTransformerBlock
from main_code.model.anymate_anyone.attn_processor_valid_high import ReferenceAttnProcessorWithZeroConvolution

from ram.models.ram import ram
from ram.models.ram_lora import ram as ram_deg
from ram import inference_ram as inference
tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])
ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def set_reference_processor(unet, args):
    # 获取需要处理的模块
    # referencenet是一个权重
    # 为需要加入controlnet的Unet mid and up block加入新的reference模块
    fusion_blocks = args.fusion_blocks

    if fusion_blocks == "midup":
        attn_modules_unet = [m for m in (torch_dfs(unet.mid_block) + torch_dfs(unet.up_blocks))
                             if isinstance(m, BasicTransformerBlock) or isinstance(m, _BasicTransformerBlock)]


    elif fusion_blocks == "full":
        attn_modules_unet = [m for m in torch_dfs(unet)
                             if isinstance(m, BasicTransformerBlock) or isinstance(m, _BasicTransformerBlock)]
    else:
        raise ValueError(f"Unknown fusion_blocks mode: {fusion_blocks}")

    for module_unet in attn_modules_unet:
        if isinstance(module_unet, BasicTransformerBlock):
            module_unet.attn_ref.set_processor(ReferenceAttnProcessorWithZeroConvolution())
            
    return unet

def prepare_models(args, device, accelerator):
    
    # initialize degradation estimation network

    pretrained_backbone_path = args.pretrained_backbone_path
    pretrained_ref_gen_path = args.pretrained_ref_gen_path

    sd_path = args.sd_path
    net_sr = GenModel(sd_path = sd_path, pretrained_backbone_path=pretrained_backbone_path, pretrained_ref_gen_path = pretrained_ref_gen_path, args = args)
    net_ref = RefModel(sd_path = sd_path)
        
    # wrap model!!!
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            print("----enable_xformers_memory_efficient_attention!!!---")
            net_sr.unet.enable_xformers_memory_efficient_attention()
            net_ref.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")
            
    net_sr.unet = set_reference_processor(net_sr.unet, args)

    is_image = args.is_image
    fusion_blocks = args.fusion_blocks

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    reference_control_writer = ReferenceNetAttention(net_ref.unet, do_classifier_free_guidance=False, mode='write', fusion_blocks=fusion_blocks, is_image=is_image, dtype=weight_dtype)
    reference_control_reader = ReferenceNetAttention(net_sr.unet, do_classifier_free_guidance=False, mode='read', fusion_blocks=fusion_blocks, is_image=is_image, dtype=weight_dtype)
    
    
    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de.eval()
    net_sr.set_eval()
    net_ref.set_eval()
    

    net_sr.to(accelerator.device, dtype=weight_dtype) 
    net_ref.to(accelerator.device, dtype=weight_dtype)
    net_de = net_de.to(accelerator.device, dtype=weight_dtype)
    
    # net_sr, net_ref = accelerator.prepare(net_sr, net_ref)
    # net_de = accelerator.prepare(net_de)

    model_vlm = ram(
        pretrained=args.ram_path,
        image_size=384,
        vit='swin_l'
    )
    model_vlm.eval().cuda()  # 建议推理时加上 .cuda()
    model_vlm.to("cuda", dtype=torch.float16)
    
    
    model_vlm_deg = ram_deg(
        pretrained=args.ram_path,
        pretrained_condition=args.dape_path,
        image_size=384,
        vit='swin_l'
    )
    model_vlm_deg.eval().cuda()  # 建议推理时加上 .cuda()
    model_vlm_deg.to("cuda", dtype=torch.float16)
                    
    
    return net_sr, net_ref, reference_control_writer, reference_control_reader, net_de, model_vlm, model_vlm_deg
    
def main(args, device):
    
    dataset_config = OmegaConf.load(args.base_config)
    dataset_val = RefTestDataset(opt = dataset_config.dataset)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=1)
    
    
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # get models
    net_sr, net_ref, reference_control_writer, reference_control_reader, net_de, model_vlm, model_vlm_deg = prepare_models(args, device, accelerator)
    
    
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True


    valid_image_dir_full = os.path.join(args.output_dir, args.exp_name, "full_images")
    if not osp.exists(valid_image_dir_full):
        os.makedirs(valid_image_dir_full)
    
    valid_image_dir_pred = os.path.join(args.output_dir, args.exp_name, "pred_images")
    if not osp.exists(valid_image_dir_pred):
        os.makedirs(valid_image_dir_pred)
    
    ref_image_dir_pred = os.path.join(args.output_dir, args.exp_name, "ref_images")
    if not osp.exists(ref_image_dir_pred):
        os.makedirs(ref_image_dir_pred)
        
    print(">>> Accelerator precision:", accelerator.mixed_precision)
    
    # for validation!!!
    for step, batch_val in tqdm(enumerate(dl_val)):
        x_src = batch_val["lr"].to(accelerator.device, dtype = weight_dtype)
        ref_img_val = batch_val["ref"].to(accelerator.device, dtype = weight_dtype)
        x_tgt = batch_val["gt"].to(accelerator.device, dtype = weight_dtype)

        x_src_norm = x_src * 2 - 1.0
        x_src_norm = torch.clamp(x_src_norm, -1.0, 1.0)

        img_ref_norm = ref_img_val * 2 - 1.0
        img_ref_norm = torch.clamp(img_ref_norm, -1.0, 1.0)
        
        x_src_ram = batch_val["lr_ram"].to(accelerator.device, dtype = weight_dtype)
        x_ref_ram = batch_val["ref_ram"].to(accelerator.device, dtype = weight_dtype)
        
        print("----- 1. x_src_norm shape --------")
        print(x_src_norm.shape)
        
        print(x_src.dtype)
        print("x_src_norm.dtype:", x_src_norm.dtype)  # 要是 float16 才行

        
        B, C, H, W = x_src.shape
        assert B == 1, "Use batch size 1 for eval."

        with torch.no_grad():
            deg_score = net_de(x_src.detach())
            print("deg_score dtype:", deg_score.dtype)

            caption_ref = inference(x_ref_ram.to(dtype=torch.float16), model_vlm)
            vlm_prompt_ref = [f'{each_caption}' for each_caption in caption_ref]

            caption_src = inference(x_src_ram.to(dtype=torch.float16), model_vlm_deg)
            vlm_prompt_src = [f'{each_caption}' for each_caption in caption_src]

            print("----- 2. x_ref_norm shape --------")
            print(img_ref_norm.shape)
            

            ref_predict = net_ref(img_ref_norm.detach(), prompt = vlm_prompt_ref)
            reference_control_reader.update(reference_control_writer, dtype=weight_dtype)   
            
            
            # print("ref_predict dtype:", ref_predict.dtype)
            

            x_tgt_pred = accelerator.unwrap_model(net_sr)(x_src_norm, deg_score, prompt = vlm_prompt_src)
            print("x_tgt_pred dtype:", x_tgt_pred.dtype)

            
            reference_control_reader.clear()
            reference_control_writer.clear()
        
        x_src = x_src.cpu().detach().float()
        x_tgt = x_tgt.cpu().detach().float()
        x_tgt_pred = x_tgt_pred.cpu().detach().float() * 0.5 + 0.5
        ref_img_val = ref_img_val.cpu().detach().float()  # 本身就在 [0, 1]
        ref_predict = ref_predict.cpu().detach().float() * 0.5 + 0.5
        
        
        if args.align_method == 'nofix':
            x_tgt_pred = x_tgt_pred
        else:
            x_src_img = transforms.ToPILImage()(x_src[0].cpu().detach())
            x_tgt_img = transforms.ToPILImage()(x_tgt_pred[0].cpu().detach())
            
            if args.align_method == 'wavelet':
                x_tgt_pred = wavelet_color_fix(x_tgt_img, x_src_img)
            elif args.align_method == 'adain':
                x_tgt_pred = adain_color_fix(x_tgt_img, x_src_img)
        to_tensor = ToTensor()
        x_tgt_pred = to_tensor(x_tgt_pred).unsqueeze(0)
        
                
        # save reference image!
        if args.align_method == 'nofix':
            ref_predict = ref_predict
        else:
            ref_img_val = transforms.ToPILImage()(ref_img_val[0].cpu().detach())
            ref_img_pred = transforms.ToPILImage()(ref_predict[0].cpu().detach())
            if args.align_method == 'wavelet':
                ref_predict = wavelet_color_fix(ref_img_pred, ref_img_val)
            elif args.align_method == 'adain':
                ref_predict = adain_color_fix(ref_img_pred, ref_img_val)
        ref_predict = to_tensor(ref_predict).unsqueeze(0)
        ref_img_val = to_tensor(ref_img_val).unsqueeze(0)
        
        # 去掉pad！！！ （根据ori_height, ori_width），然后再将图像resize到img_gt的情况
        
        
        image_name = batch_val["lr_name"][0]
        # 获取填充值
        height, width = batch_val["height"], batch_val["width"]

        gt_height, gt_width = x_tgt[0].shape[1:]
        x_tgt_pred = x_tgt_pred[0]    
        # rescale to 

        # 将张量转换为 PIL 图像
        # todo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # ---------------------------- todo -------------------------------
        
        # 现在的x_tgt_pred经过了padding操作，需要去除padding
        x_tgt_pred = x_tgt_pred
        output_pil = transforms.ToPILImage()(x_tgt_pred)
        output_pil = output_pil.resize((gt_width, gt_height), Image.BICUBIC)
        
        image_name_out = image_name
        outf = os.path.join(valid_image_dir_pred, image_name_out)
        output_pil.save(outf)
        
        if args.save_lr_pred_ref:
            x_tgt = x_tgt[0]    # [c, h1, w1]
            x_src_unpadded = x_src[0]    # [c, h2, w2]
            ref_img_val_unpadded = ref_img_val[0]    # [c, h2, w2]
            x_tgt_unpadded = x_tgt_pred
            # resize to x_tgt size!!!
            
            # 获取 x_tgt 的尺寸
            h1, w1 = x_tgt.shape[1], x_tgt.shape[2]

            # 将 x_src_unpadded 调整到 x_tgt 的尺寸
            x_src_resized = F.interpolate(x_src_unpadded.unsqueeze(0), size=(h1, w1), mode='bilinear', align_corners=False).squeeze(0)
            x_tgt_pred_resized = F.interpolate(x_tgt_unpadded.unsqueeze(0), size=(h1, w1), mode='bilinear', align_corners=False).squeeze(0)
            # 将 ref_img_val_unpadded 调整到 x_tgt 的尺寸
            ref_img_val_resized = F.interpolate(ref_img_val_unpadded.unsqueeze(0), size=(h1, w1), mode='bilinear', align_corners=False).squeeze(0)
            
            combined = torch.cat([x_src_resized, x_tgt_pred_resized, x_tgt, ref_img_val_resized], dim=2)
            output_pil = transforms.ToPILImage()(combined)
            outf = os.path.join(valid_image_dir_full, image_name)
            output_pil.save(outf)
            
        print(f"image {image_name} saved at {outf}")
                

if __name__ == "__main__":
    
    args = parse_args_paired_testing()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    main(args, device)