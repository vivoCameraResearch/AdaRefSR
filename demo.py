import torch
import os
import argparse
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from main_code.model.gen_model import GenModel
from main_code.model.ref_model import RefModel
from main_code.model.de_net import DEResNet
from main_code.model.anymate_anyone.reference_attention import ReferenceNetAttention

from ram.models.ram import ram
from ram.models.ram_lora import ram as ram_deg
from ram import inference_ram as inference
    
from my_utils.wavelet_color import wavelet_color_fix
from accelerate import Accelerator
from omegaconf import OmegaConf

# 导入原始的参数解析函数
from my_utils.testing_utils import parse_args_paired_testing

# 基础配置
ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def run_demo(lq_path, ref_path, output_path, args):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

    # 1. 加载基础 SR 模型
    net_sr = GenModel(sd_path=args.sd_path, pretrained_backbone_path=args.pretrained_backbone_path, 
                      pretrained_ref_gen_path=args.pretrained_ref_gen_path, args=args)
    net_ref = RefModel(sd_path=args.sd_path)
    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    
    # --- VLM 模块条件加载 ---
    model_vlm = None
    model_vlm_deg = None


    model_vlm = ram(pretrained=args.ram_path, image_size=384, vit='swin_l').eval().to(device, dtype=torch.float16)
    model_vlm_deg = ram_deg(pretrained=args.ram_path, pretrained_condition=args.dape_path, image_size=384, vit='swin_l').eval().to(device, dtype=torch.float16)

    # 设置 Attention 处理器 (AICG)
    reference_control_writer = ReferenceNetAttention(net_ref.unet, mode='write', fusion_blocks=args.fusion_blocks, is_image=True, dtype=weight_dtype)
    reference_control_reader = ReferenceNetAttention(net_sr.unet, mode='read', fusion_blocks=args.fusion_blocks, is_image=True, dtype=weight_dtype)

    net_sr.to(device, dtype=weight_dtype).eval()
    net_ref.to(device, dtype=weight_dtype).eval()
    net_de.to(device, dtype=weight_dtype).eval()

    # 2. 预处理图像
    lq_img = Image.open(lq_path).convert("RGB")
    ref_img = Image.open(ref_path).convert("RGB")
    lq_img = lq_img.resize((lq_img.size[0]//8*8, lq_img.size[1]//8*8)) 
    ref_img = ref_img.resize((ref_img.size[0]//8*8, ref_img.size[1]//8*8))
    orig_w, orig_h = lq_img.size
    
    transform = transforms.ToTensor()
    x_src = transform(lq_img).unsqueeze(0).to(device, dtype=weight_dtype)
    x_ref = transform(ref_img).unsqueeze(0).to(device, dtype=weight_dtype)

    # 3. 推理过程
    with torch.no_grad():
        deg_score = net_de(x_src)
        
        from ram import inference_ram as inference
        prompt_ref = inference(ram_transforms(x_ref).to(device, dtype=torch.float16), model_vlm)
        prompt_src = inference(ram_transforms(x_src).to(device, dtype=torch.float16), model_vlm_deg)


        net_ref(x_ref * 2 - 1.0, prompt=prompt_ref)
        reference_control_reader.update(reference_control_writer, dtype=weight_dtype)
        
        prediction = net_sr(x_src * 2 - 1.0, deg_score, prompt=prompt_src)
        
        reference_control_reader.clear()
        reference_control_writer.clear()

    # 4. 后处理
    pred_img = transforms.ToPILImage()((prediction[0] * 0.5 + 0.5).clamp(0, 1).cpu())
    if args.get('align_method', 'wavelet') == 'wavelet':
        pred_img = wavelet_color_fix(pred_img, lq_img)
    
    pred_img.resize((orig_w, orig_h), Image.BICUBIC).save(output_path)
    print(f"✅ Result saved to: {output_path}")

if __name__ == "__main__":
    demo_parser = argparse.ArgumentParser(add_help=False) # 关闭 help 防止冲突
    demo_parser.add_argument("--config", type=str, default="./configs/demo_config.yaml")
    demo_parser.add_argument("--lq_path", type=str, required=True)
    demo_parser.add_argument("--ref_path", type=str, required=True)
    demo_parser.add_argument("--output_path", type=str, default="./assets/pic/result.png")

    demo_args, unknown = demo_parser.parse_known_args()

    import sys
    sys.argv = [sys.argv[0]] + unknown 
    args = parse_args_paired_testing()

    base_cfg = OmegaConf.create(vars(args))
    
    if os.path.exists(demo_args.config):
        yaml_cfg = OmegaConf.load(demo_args.config)
        base_cfg = OmegaConf.merge(base_cfg, yaml_cfg)
        print(f">>> Loaded YAML config from {demo_args.config}")

    # 将 demo 参数合并进去
    final_cfg = OmegaConf.merge(base_cfg, OmegaConf.create(vars(demo_args)))

    # 5. 运行推理
    run_demo(final_cfg.lq_path, final_cfg.ref_path, final_cfg.output_path, final_cfg)