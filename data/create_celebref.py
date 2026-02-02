import os
import json
import random
from itertools import permutations

# 配置参数
BASE_DIR = "your_code_path/data/CelebRef/CelebHQRefForRelease"  # 替换为实际路径
OUTPUT_DIR = "./train_dataset/CelebRef"  # JSON输出路径


# 1. 收集所有ID文件夹
ids = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
train_num = len(ids) - 30  # 假设最后30个ID用于测试和验证
random.seed(42)  # 固定随机种子确保可复现
random.shuffle(ids)

SPLIT_RATIO = {
    "train": train_num,
    "valid": 10,
    "test": 40
}


# 2. 划分ID到不同集合
split_ids = {
    "train": ids[:SPLIT_RATIO["train"]],
    "valid": ids[SPLIT_RATIO["train"]:SPLIT_RATIO["train"]+SPLIT_RATIO["valid"]],
    "test": ids[-SPLIT_RATIO["test"]:]
}

# 3. 生成配对数据
def generate_pairs(id_dir):
    img_paths = [os.path.join(id_dir, f) for f in os.listdir(id_dir) if f.lower().endswith(("png", "jpg", "jpeg"))]
    pairs = []
    for ref, gt in permutations(img_paths, 2):  # 生成所有有序对（不包含自身）
        pairs.append({
            "ref_path": ref,
            "gt_path": gt
        })
    return pairs, len(img_paths)

# 4. 构建JSON文件并统计
def build_json(split_name, split_ids):
    data = []
    total_images = 0
    total_pairs = 0
    for id in split_ids[split_name]:
        id_dir = os.path.join(BASE_DIR, id)
        if not os.path.isdir(id_dir):
            continue
        pairs, img_count = generate_pairs(id_dir)
        data.extend(pairs)
        total_images += img_count
        total_pairs += len(pairs)
    return data, total_images, total_pairs

# 5. 保存JSON文件并统计结果
os.makedirs(OUTPUT_DIR, exist_ok=True)

stats = {}

for split in ["train", "valid", "test"]:
    json_data, img_count, pair_count = build_json(split, split_ids)
    output_path = os.path.join(OUTPUT_DIR, f"{split}.json")
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)
    
    stats[split] = {
        "ids": len(split_ids[split]),
        "images": img_count,
        "pairs": pair_count
    }
    
    print(f"{split}.json 已生成，包含 {len(json_data)} 个配对")

# 6. 输出统计结果到txt文件
stats_path = os.path.join(OUTPUT_DIR, "dataset_stats.txt")
with open(stats_path, "w") as f:
    f.write("===== 数据集统计结果 =====\n\n")
    
    for split in ["train", "valid", "test"]:
        f.write(f"[{split.upper()} SET]\n")
        f.write(f"ID数量: {stats[split]['ids']}\n")
        f.write(f"图像数量: {stats[split]['images']}\n")
        f.write(f"配对数量: {stats[split]['pairs']}\n\n")
    
    # 计算总计
    total_ids = sum(stats[split]["ids"] for split in stats)
    total_images = sum(stats[split]["images"] for split in stats)
    total_pairs = sum(stats[split]["pairs"] for split in stats)
    
    f.write("===== 总计 =====\n")
    f.write(f"ID总数: {total_ids}\n")
    f.write(f"图像总数: {total_images}\n")
    f.write(f"配对总数: {total_pairs}\n")

print(f"统计结果已保存至: {stats_path}")