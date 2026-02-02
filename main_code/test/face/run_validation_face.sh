#!/bin/bash
version_name="face"
exp_name="final"
log_file="your_code_path/output/test/face/${exp_name}/iqa_results.log"

learnable_tokens=16

python ./validation_reference.py \
    --de_net_path="your_code_path/assets/mm-realsr/de_net.pth" \
    --sd_path="your_code_path/model/stabilityai/sd-turbo" \
    --pretrained_backbone_path your_code_path/model/zhangap/S3Diff/s3diff.pkl \
    --pretrained_ref_gen_path="your_code_path/model/adarefsr.pkl" \
    --base_config your_code_path/main_code/test/face/validation.yaml \
    --output_dir your_code_path/output/test/face/ \
    --save_lr_pred_ref True \
    --exp_name ${exp_name} \
    --is_image True \
    --fusion_blocks full \
    --enable_xformers_memory_efficient_attention \
    --mixed_precision fp16 \
    --learnable_tokens=${learnable_tokens}


echo "=== Step ${steps} 质量评估结果 ===" >> $log_file
echo "评估时间: $(date '+%Y-%m-%d %H:%M:%S')" >> $log_file

python ./cal_iqa.py --img_path your_code_path/output/test/face/${exp_name}/pred_images --gt_path your_code_path/data/face/gt >> $log_file 2>&1



