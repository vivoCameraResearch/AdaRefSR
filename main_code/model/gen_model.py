import re
import sys
p = "src/"
sys.path.append(p)

import copy
import numpy as np
from tqdm import tqdm
from basicsr.archs.arch_util import default_init_weights
from peft import LoraConfig
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention import BasicTransformerBlock as _BasicTransformerBlock

from main_code.model.anymate_anyone.attn_processor import ReferenceAttnProcessorWithZeroConvolution
from main_code.model.model import make_1step_sched, my_lora_fwd

def get_layer_number(module_name):
    base_layers = {
        'down_blocks': 0,
        'mid_block': 4,
        'up_blocks': 5
    }

    if module_name == 'conv_out':
        return 9

    base_layer = None
    for key in base_layers:
        if key in module_name:
            base_layer = base_layers[key]
            break

    if base_layer is None:
        return None

    additional_layers = int(re.findall(r'\.(\d+)', module_name)[0]) #sum(int(num) for num in re.findall(r'\d+', module_name))
    final_layer = base_layer + additional_layers
    return final_layer




# todo 1_1: 加入attn_ref 
def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

def initiate_ori_s3diff(unet, vae, lora_rank_unet, lora_rank_vae, target_modules_unet, target_modules_vae, neg_module = "attn_ref"):
    vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
        target_modules=target_modules_vae)
    vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
    
    target_modules_unet_new = set()
    
    print(target_modules_unet)
    
    neg_module = "attn_ref"

    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in target_modules_unet:
            if pattern in n and neg_module not in n:
                module_name = ".".join(n.split(".")[:-1])  # 去掉最后的 weight/bias
                module_name = module_name.replace(".weight","")
                target_modules_unet_new.add(module_name)
    
    
    target_modules_unet_new = list(target_modules_unet_new)           
    unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
        target_modules=target_modules_unet_new
    )
    unet.add_adapter(unet_lora_config)  # 在目标层当中加入lora网络
    return unet, vae, target_modules_unet_new


def add_reference(unet, args, ori_unet):
    # 获取需要处理的模块
    # referencenet是一个权重
    # 为需要加入controlnet的Unet mid and up block加入新的reference模块
    fusion_blocks = args.fusion_blocks

    if fusion_blocks == "midup":
        attn_modules_unet = [m for m in (torch_dfs(unet.mid_block) + torch_dfs(unet.up_blocks))
                             if isinstance(m, BasicTransformerBlock) or isinstance(m, _BasicTransformerBlock)]
        attn_modules_ori = [m for m in (torch_dfs(ori_unet.mid_block) + torch_dfs(ori_unet.up_blocks))
                            if isinstance(m, BasicTransformerBlock) or isinstance(m, _BasicTransformerBlock)]

    elif fusion_blocks == "full":
        attn_modules_unet = [m for m in torch_dfs(unet)
                             if isinstance(m, BasicTransformerBlock) or isinstance(m, _BasicTransformerBlock)]
        attn_modules_ori = [m for m in torch_dfs(ori_unet)
                            if isinstance(m, BasicTransformerBlock) or isinstance(m, _BasicTransformerBlock)]
    else:
        raise ValueError(f"Unknown fusion_blocks mode: {fusion_blocks}")

    for module_unet, module_ori in zip(attn_modules_unet, attn_modules_ori):
        if isinstance(module_unet, BasicTransformerBlock):
            # 从 ori_unet 拷贝 attn1 到 attn_ref
            module_unet.attn_ref = copy.deepcopy(module_ori.attn1)

            # 添加 zero_linear
            inner_dim = module_unet.attn_ref.inner_dim
            out_dim = module_unet.attn_ref.out_dim
            module_unet.attn_ref.zero_linear = nn.Linear(inner_dim, out_dim)
            num_learnable_tokens = args.learnable_tokens
            module_unet.attn_ref.learnable_token = nn.Parameter(torch.randn(num_learnable_tokens, inner_dim))

            nn.init.zeros_(module_unet.attn_ref.zero_linear.weight)
            nn.init.zeros_(module_unet.attn_ref.zero_linear.bias)
            module_unet.attn_ref.set_processor(ReferenceAttnProcessorWithZeroConvolution())
            
    return unet


class GenModel(torch.nn.Module):
    def __init__(self, sd_path=None, pretrained_backbone_path=None, lora_rank_unet=32, lora_rank_vae=16, block_embedding_dim=64, pretrained_ref_gen_path = None, args = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").cuda()
        self.sched = make_1step_sched(sd_path)

        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")   
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")  # 更改一下Unet网络的输入
     
        
        target_modules_vae = r"^encoder\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"
        
        target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
        ]   # unet当中的所有模块，包含了to_k等可以调整的网络都会使用lora
        
        # ---------------------------------------
        self.get_s3diff_other_models(block_embedding_dim,lora_rank_vae , lora_rank_unet)
        
        
        if pretrained_backbone_path is not None:
            print(f"Initializing model with pretrained weights {pretrained_backbone_path}")
            self.unet, self.vae = self.load_ori_lora_model(pretrained_backbone_path, unet, vae) # 只会保存特定层
        
        # todo version_14_1：加入attn_ref网络
        ori_unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")  # 更改一下Unet网络的输入
        self.unet = add_reference(self.unet, args, ori_unet)

        if pretrained_ref_gen_path is not None:
            print(f"using pretrained model path from {pretrained_ref_gen_path}")
            ref_gen_model = torch.load(pretrained_ref_gen_path)
            self.load_ref_gen_weight(ref_gen_model)

        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae
        self.target_modules_vae = target_modules_vae
        self.target_modules_unet = target_modules_unet

        
        # 注册原有的processor
        self.register_abc_processor(unet, vae)

        self.unet_layer_dict = {name: get_layer_number(name) for name in self.unet_lora_layers}

        self.unet.to("cuda")
        self.vae.to("cuda")

        self.timesteps = torch.tensor([999], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def load_ref_gen_weight(self, model):
        for n, p in self.unet.named_parameters():
            if "attn_ref" in n:
                p.data.copy_(model["state_dict_unet_attn_ref"][n])

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.vae_de_mlp.eval()
        self.unet_de_mlp.eval()
        self.vae_block_mlp.eval()
        self.unet_block_mlp.eval()
        self.vae_fuse_mlp.eval()
        self.unet_fuse_mlp.eval()

        self.vae_block_embeddings.requires_grad_(False)
        self.unet_block_embeddings.requires_grad_(False)

        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        # self.vae.train()

        # 全量调节
        for n, _p in self.unet.named_parameters():
            if "attn_ref" in n:
                _p.requires_grad = True


    def forward(self, c_t, deg_score, prompt, cross_attention_kwargs = None):
        # breakpoint()
        if prompt is not None:
            # encode the text prompt
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        # degradation fourier embedding # todo map 2d to 2d * m dimension d_e, score is control parameters in [0, 1]
        deg_proj = deg_score[..., None] * self.W[None, None, :] * 2 * np.pi # 2, ... * 1, 1, M -> 2, 1, M
        deg_proj = torch.cat([torch.sin(deg_proj), torch.cos(deg_proj)], dim=-1)
        deg_proj = torch.cat([deg_proj[:, 0], deg_proj[:, 1]], dim=-1)  #  2, 2 * M

        # degradation mlp forward
        vae_de_c_embed = self.vae_de_mlp(deg_proj)  # c x c
        unet_de_c_embed = self.unet_de_mlp(deg_proj)    # 分别加入到unet和vae的lora当中：专门学习一种改变模型参数的方式，通过两套参数共同前向（相当于控制了Lora的参数！）

        # block embedding mlp forward
        vae_block_c_embeds = self.vae_block_mlp(self.vae_block_embeddings.weight)
        unet_block_c_embeds = self.unet_block_mlp(self.unet_block_embeddings.weight)    # 一个可以学习的参数  
        vae_embeds = self.vae_fuse_mlp(torch.cat([vae_de_c_embed.unsqueeze(1).repeat(1, vae_block_c_embeds.shape[0], 1), \
            vae_block_c_embeds.unsqueeze(0).repeat(vae_de_c_embed.shape[0],1,1)], -1))  # 
        unet_embeds = self.unet_fuse_mlp(torch.cat([unet_de_c_embed.unsqueeze(1).repeat(1, unet_block_c_embeds.shape[0], 1), \
            unet_block_c_embeds.unsqueeze(0).repeat(unet_de_c_embed.shape[0],1,1)], -1))    # 
        
        # get block id for vae to get MLP(de_c_embed, layer_id)
        for layer_name, module in self.vae.named_modules():
            if layer_name in self.vae_lora_layers:
                split_name = layer_name.split(".")
                if split_name[1] == 'down_blocks': 
                    block_id = int(split_name[2])
                    vae_embed = vae_embeds[:, block_id] # 
                elif split_name[1] == 'mid_block':
                    vae_embed = vae_embeds[:, -2]
                else:
                    vae_embed = vae_embeds[:, -1]
                module.de_mod = vae_embed.reshape(-1, self.lora_rank_vae, self.lora_rank_vae)   # 这里新加入了，所以就和定义的前向function一致

        for layer_name, module in self.unet.named_modules():
            if layer_name in self.unet_lora_layers:
                split_name = layer_name.split(".")

                if split_name[0] == 'down_blocks':
                    block_id = int(split_name[1])
                    unet_embed = unet_embeds[:, block_id]
                elif split_name[0] == 'mid_block':
                    unet_embed = unet_embeds[:, 4]
                elif split_name[0] == 'up_blocks':
                    block_id = int(split_name[1]) + 5
                    unet_embed = unet_embeds[:, block_id]
                else:
                    unet_embed = unet_embeds[:, -1]
                module.de_mod = unet_embed.reshape(-1, self.lora_rank_unet, self.lora_rank_unet)


        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc,cross_attention_kwargs = cross_attention_kwargs).sample
        x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
        x_denoised = x_denoised.to(dtype=self.vae.dtype)
        # print(self.vae.dtype)
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image # , x_denoised

    def save_model(self, outf):
        sd = {}
        sd["state_dict_unet_attn_ref"] = {k: v for k, v in self.unet.state_dict().items() if "attn_ref" in k}
        torch.save(sd, outf)


    # 导入原始网络的权重
    # 这里的逻辑可以分开
    def load_ori_lora_model(self, pretrained_backbone_path, unet, vae):
        sd = torch.load(pretrained_backbone_path, map_location="cpu")    # what is sd["w"]
        
        vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
        vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

        unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
        unet.add_adapter(unet_lora_config) 
        
        _sd_vae = vae.state_dict()
        for k in sd["state_dict_vae"]:
            _sd_vae[k] = sd["state_dict_vae"][k]
        vae.load_state_dict(_sd_vae)
        
        _sd_unet = unet.state_dict()

        # map adapter for model with self-model!!!
        # 预训练好的模型，map一下
        for k in sd["state_dict_unet"]:
            _sd_unet[k] = sd["state_dict_unet"][k]
        unet.load_state_dict(_sd_unet)

        _vae_de_mlp = self.vae_de_mlp.state_dict()
        for k in sd["state_dict_vae_de_mlp"]:
            _vae_de_mlp[k] = sd["state_dict_vae_de_mlp"][k]
        self.vae_de_mlp.load_state_dict(_vae_de_mlp)

        _unet_de_mlp = self.unet_de_mlp.state_dict()
        for k in sd["state_dict_unet_de_mlp"]:
            _unet_de_mlp[k] = sd["state_dict_unet_de_mlp"][k]
        self.unet_de_mlp.load_state_dict(_unet_de_mlp)

        _vae_block_mlp = self.vae_block_mlp.state_dict()
        for k in sd["state_dict_vae_block_mlp"]:
            _vae_block_mlp[k] = sd["state_dict_vae_block_mlp"][k]
        self.vae_block_mlp.load_state_dict(_vae_block_mlp)

        _unet_block_mlp = self.unet_block_mlp.state_dict()
        for k in sd["state_dict_unet_block_mlp"]:
            _unet_block_mlp[k] = sd["state_dict_unet_block_mlp"][k]
        self.unet_block_mlp.load_state_dict(_unet_block_mlp)

        _vae_fuse_mlp = self.vae_fuse_mlp.state_dict()
        for k in sd["state_dict_vae_fuse_mlp"]:
            _vae_fuse_mlp[k] = sd["state_dict_vae_fuse_mlp"][k]
        self.vae_fuse_mlp.load_state_dict(_vae_fuse_mlp)    

        _unet_fuse_mlp = self.unet_fuse_mlp.state_dict()
        for k in sd["state_dict_unet_fuse_mlp"]:
            _unet_fuse_mlp[k] = sd["state_dict_unet_fuse_mlp"][k]
        self.unet_fuse_mlp.load_state_dict(_unet_fuse_mlp)

        self.W = nn.Parameter(sd["w"], requires_grad=False)

        embeddings_state_dict = sd["state_embeddings"]
        self.vae_block_embeddings.load_state_dict(embeddings_state_dict['state_dict_vae_block'])
        self.unet_block_embeddings.load_state_dict(embeddings_state_dict['state_dict_unet_block'])
        
        additional_layers = [self.vae_de_mlp, self.unet_de_mlp, self.vae_block_mlp, self.unet_block_mlp, self.vae_fuse_mlp, self.unet_fuse_mlp]
        
        for layer in additional_layers:
            layer.eval()
            layer.requires_grad_(False)
        
        self.vae_block_embeddings.requires_grad_(False)
        self.unet_block_embeddings.requires_grad_(False)
        self.W.requires_grad_(False)
        
        return unet, vae
    
        
    def register_abc_processor(self, unet, vae):
        self.vae_lora_layers = []
        
        for name, module in vae.named_modules():
            if 'base_layer' in name and "decoder" not in name:
                self.vae_lora_layers.append(name[:-len(".base_layer")])
                
        for name, module in vae.named_modules():
            if name in self.vae_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)

        self.unet_lora_layers = []
        for name, module in unet.named_modules():
            if 'base_layer' in name:
                self.unet_lora_layers.append(name[:-len(".base_layer")])  


        for name, module in unet.named_modules():
            if name in self.unet_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)
                
    def get_s3diff_other_models(self, block_embedding_dim,lora_rank_vae , lora_rank_unet):
        num_embeddings = 64
        self.W = nn.Parameter(torch.randn(num_embeddings), requires_grad=False)

        self.vae_de_mlp = nn.Sequential(
            nn.Linear(num_embeddings * 4, 256), # 
            nn.ReLU(True),
        )

        self.unet_de_mlp = nn.Sequential(
            nn.Linear(num_embeddings * 4, 256),
            nn.ReLU(True),
        )

        self.vae_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )

        self.unet_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )

        self.vae_fuse_mlp = nn.Linear(256 + 64, lora_rank_vae ** 2)
        self.unet_fuse_mlp = nn.Linear(256 + 64, lora_rank_unet ** 2)

        additional_layers = [self.vae_de_mlp, self.unet_de_mlp, self.vae_block_mlp, self.unet_block_mlp, self.vae_fuse_mlp, self.unet_fuse_mlp]
        default_init_weights(additional_layers, 1e-5)
        
        for layer in additional_layers:
            layer.eval()
            layer.requires_grad_(False)

        # vae
        self.vae_block_embeddings = nn.Embedding(6, block_embedding_dim)
        self.unet_block_embeddings = nn.Embedding(10, block_embedding_dim)