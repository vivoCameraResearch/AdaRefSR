proj_name="refsr"
learnable_tokens=16
exp_name="final_${learnable_tokens}"

export WANDB_MODE=offline

accelerate launch --num_processes=1 --gpu_ids="0" --main_process_port 29301 train_refsr.py \
    --de_net_path="your_code_path/assets/mm-realsr/de_net.pth" \
    --output_dir="your_code_path/output/${proj_name}/${exp_name}/" \
    --lambda_gan 0.5 \
    --lambda_lpips 5.0 \
    --lambda_l2 2.0 \
    --train_batch_size=1 \
    --enable_xformers_memory_efficient_attention \
    --viz_freq 25 \
    --sd_path="your_code_path/model/stabilityai/sd-turbo" \
    --base_config="your_code_path/configs/train_one_step_sr_reference_online_all.yaml" \
    --resolution=512 \
    --dataloader_num_workers=8 \
    --log_name="reference-sr" \
    --log_file="your_code_path/output/${proj_name}/${exp_name}/logs/reference-sr.log" \
    --pretrained_backbone_path="your_code_path/model/zhangap/S3Diff/s3diff.pkl" \
    --pretrained_ref_path="your_code_path/model/stabilityai/stable-diffusion-2-1-base" \
    --checkpointing_steps=1000 \
    --eval_freq=1000 \
    --summer_writer="your_code_path/output/${proj_name}/${exp_name}/summer_writer" \
    --max_train_steps=100000 \
    --add_spatial_align_net \
    --fusion_blocks="full" \
    --lambda_factor=1 \
    --gradient_accumulation_steps=4 \
    --num_samples_eval=50 \
    --visualize_nums=8 \
    --proj_name=${proj_name} \
    --exp_name=${exp_name} \
    --deg_file_path params_realesrgan.yml
