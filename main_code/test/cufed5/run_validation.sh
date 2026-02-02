#!/bin/bash
version_name="cufed5"
exp_name="final"
log_file="your_code_path/output/test/cufed5/${exp_name}/iqa_results.log"

learnable_tokens=16

python ./validation_reference.py \
    --base_config your_code_path/main_code/test/cufed5/validation.yaml \
    --output_dir your_code_path/output/test/cufed5/ \
    --save_lr_pred_ref True \
    --exp_name ${exp_name} \
    --is_image True \
    --fusion_blocks full \
    --enable_xformers_memory_efficient_attention \
    --mixed_precision fp16 \
    --learnable_tokens=${learnable_tokens}

echo "=== Step ${steps} 质量评估结果 ===" >> $log_file
echo "评估时间: $(date '+%Y-%m-%d %H:%M:%S')" >> $log_file

python ./cal_iqa.py --img_path your_code_path/output/test/cufed5/${exp_name}/pred_images --gt_path your_code_path/data/cufed5/Real_Deg/HR >> $log_file 2>&1


