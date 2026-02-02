import os
import re
import requests
import sys
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel

from main_code.model.model import make_1step_sched




class RefModel(torch.nn.Module):
    def __init__(self, sd_path=None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").cuda()
        self.sched = make_1step_sched(sd_path)

        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")   
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")  # 更改一下Unet网络的输入

        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.timesteps = torch.tensor([1], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def forward(self, c_t, prompt, other_prompt = None, cross_attention_kwargs = None):
        # breakpoint()
        if cross_attention_kwargs == None:
            cross_attention_kwargs = {}
        
        
        if prompt is not None:
            # encode the text prompt
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        if other_prompt is not None:
            # encode the text prompt
            other_caption_tokens = self.tokenizer(other_prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            other_caption_enc = self.text_encoder(other_caption_tokens)[0]
            cross_attention_kwargs["other_encoder_hidden_states"] = other_caption_enc


        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc,cross_attention_kwargs = cross_attention_kwargs).sample
        x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
        
        x_denoised = x_denoised.to(dtype=self.vae.dtype)
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image # , x_denoised