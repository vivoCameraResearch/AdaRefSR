import sys
import os
import os.path as osp

# 将项目的根目录加入到 Python 的搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import transformers

from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from diffusers.models.attention import BasicTransformerBlock

from main_code.model.de_net import DEResNet
from main_code.model.gen_model import GenModel
from main_code.model.ref_model import RefModel
from main_code.model.anymate_anyone.reference_attention import ReferenceNetAttention
from main_code.model.anymate_anyone.attn_processor import ReferenceAttnProcessorWithZeroConvolution


from my_utils.training_utils import parse_args_paired_training
from my_utils.dataset.ref_sr_dataset_train import ReferenceOnlineContrastDataset

from torch.utils.tensorboard import SummaryWriter
from mmcv.utils import get_logger
from ram.models.ram import ram
from ram.models.ram_lora import ram as ram_deg
from ram import inference_ram as inference

import wandb
wandb.login()

# todo 1_1: 加入attn_ref 
def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

def register_unet_sr_model(unet_model, block_wise = False):
    for name, layer in unet_model.named_modules():
        if isinstance(layer, BasicTransformerBlock):
            if hasattr(layer, 'attn_ref'):
                layer.attn_ref.processor = ReferenceAttnProcessorWithZeroConvolution()    
    return unet_model


def main(args):

    # init and save configs
    config = OmegaConf.load(args.base_config)
    
    if not osp.exists(osp.dirname(args.log_file)):
        os.makedirs(osp.dirname(args.log_file), exist_ok=True)
    
    logger = get_logger(
        name=args.log_name,
        log_file=args.log_file,  # 日志保存路径
        log_level='INFO',  # 日志级别
    )
    
    run = wandb.init(project=args.proj_name, name=args.exp_name)
    
    
    writer = SummaryWriter(log_dir=args.summer_writer)  # logs 文件夹将存储 TensorBoard 数据
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    # initialize degradation estimation network
    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de = net_de.cuda()
    net_de.eval()

    net_sr = GenModel(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae, sd_path=args.sd_path, pretrained_backbone_path=args.pretrained_backbone_path,pretrained_ref_gen_path = args.pretrained_ref_gen_path,  args = args)
    net_ref = RefModel(sd_path=args.sd_path)
    
    net_sr.set_train()
    net_ref.set_eval()
    
    reference_control_writer = ReferenceNetAttention(net_ref.unet, do_classifier_free_guidance=False, mode='write', fusion_blocks=args.fusion_blocks, is_image=args.is_image)
    reference_control_reader = ReferenceNetAttention(net_sr.unet, do_classifier_free_guidance=False, mode='read', fusion_blocks=args.fusion_blocks, is_image=args.is_image)
    
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

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_sr.unet.enable_xformers_memory_efficient_attention()
            net_ref.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    net_sr.unet = register_unet_sr_model(net_sr.unet)

    if args.gradient_checkpointing:
        net_sr.unet.enable_gradient_checkpointing()
        

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gan_disc_type == "vagan":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(cv_type='dino', output_type='conv_multi_level', loss_type=args.gan_loss_type, device="cuda")
    else:
        raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")

    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)
    
    # todo modify here for experiment 4
    layers_to_opt = []

    for n, _p in net_sr.unet.named_parameters():
        if "attn_ref" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
                
    # todo 修改一下
    dataset_train = ReferenceOnlineContrastDataset(split = "train", args =args, opt = config.train)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)    

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles, power=args.lr_power)

    # Prepare everything with our `accelerator`.
    net_sr, net_ref, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        net_sr, net_ref, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc
    )
    net_de, net_lpips = accelerator.prepare(net_de, net_lpips)
    # # renorm with image net statistics
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_sr.to(accelerator.device, dtype=weight_dtype)
    
    net_ref.to(accelerator.device, dtype=weight_dtype)
    net_de.to(accelerator.device, dtype=weight_dtype)
    net_disc.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False
    
    global_step = 0
    for epoch in range(0, args.num_training_epochs):
        for training_step, batch in enumerate(dl_train):
            if global_step >= args.max_train_steps:
                exit()
            
            l_acc = [net_sr, net_disc]
            with accelerator.accumulate(*l_acc):
                # todo batch当中需要获取reference图像
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(accelerator.device)
                x_src = batch["lq"] # 范围在[-1, 1]之间, 注意是[B, C, H, W]大小
                x_tgt = batch["gt"] # 范围在[-1, 1]之间
                x_ref = batch["ref"] # 范围在[-1, 1]之间
                
                x_tgt_ram = batch["gt_ram"]
                x_ref_ram = batch["ref_ram"]
                
                B, C, H, W = x_src.shape
                with torch.no_grad():
                    x_src_de_norm = x_src * 0.5 + 0.5
                    deg_score = net_de(x_src_de_norm.detach()).detach()    # 获取一个得分（应该是分类器，得到一个分

                    caption_ref = inference(x_ref_ram.to(dtype=torch.float16), model_vlm)
                    vlm_prompt_ref = [f'{each_caption}' for each_caption in caption_ref]
                    
                    caption_gt = inference(x_tgt_ram.to(dtype=torch.float16), model_vlm)
                    vlm_prompt_gt = [f'{each_caption}' for each_caption in caption_gt]
                
                
                net_ref(x_ref.detach(), vlm_prompt_ref)
                reference_control_reader.update(reference_control_writer)
                
                x_tgt_pred = net_sr(x_src.detach(), deg_score, vlm_prompt_gt)

                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.detach().float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.detach().float()).mean() * args.lambda_lpips

                loss = loss_l2 + loss_lpips
                
                """
                Generator loss: fool the discriminator
                """
                lossG = net_disc(x_tgt_pred, for_G=True).mean() * args.lambda_gan
                loss += lossG
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                """
                Discriminator loss: fake image vs real image
                """             
                
                # real image
                lossD_real = net_disc(x_tgt.detach(), for_real=True).mean() * args.lambda_gan
                accelerator.backward(lossD_real.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                
                
                # fake image
                lossD_fake = net_disc(x_tgt_pred.detach(), for_real=False).mean() * args.lambda_gan
                accelerator.backward(lossD_fake.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                lossD = lossD_real + lossD_fake
                
                
                # clear bank!
                reference_control_reader.clear()
                reference_control_writer.clear()
            
            
            if accelerator.sync_gradients:

                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    logger.info(
                        f"global step: {global_step} "
                        f"training step: {training_step} "
                        f"loss: {loss.item():.2f} "
                        f"loss_l2: {loss_l2.item():.2f} "
                        f"loss_lpips: {loss_lpips.item():.2f} "
                        f"lossG: {lossG.item():.2f} "
                        f"lossD: {lossD.item():.2f}"
                    )
                    
                    run.log({
                        "Loss/total": loss.item(),
                        "Loss/l2": loss_l2.item(),
                        "Loss/lpips": loss_lpips.item(),
                        "Loss/G": lossG.item(),
                        "Loss/D": lossD.item(),
                    }, step=global_step)
                    
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"epoch_{global_step}.pkl")
                        accelerator.unwrap_model(net_sr).save_model(outf)
    writer.close()


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
