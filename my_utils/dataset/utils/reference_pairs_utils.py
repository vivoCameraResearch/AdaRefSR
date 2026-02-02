import random
import numpy as np
from PIL import Image
import cv2


def rotate(img, angle_range=(0, 15)):
    """Rotate the image within a specified angle range."""
    angle = random.uniform(*angle_range)
    img = Image.fromarray(img.transpose(1, 2, 0))  # C, H, W -> H, W, C
    rotated_img = img.rotate(angle, resample=Image.BICUBIC, expand=False)
    return np.array(rotated_img).transpose(2, 0, 1)  # H, W, C -> C, H, W

def random_crop_(img, crop_size=1024):
    """Randomly crop a square area from the image."""
    _, H, W = img.shape
    top = random.randint(0, H - crop_size)
    left = random.randint(0, W - crop_size)
    return img[:, top:top + crop_size, left:left + crop_size]

def random_crop_area(img, area_size=2048, crop_size=1024):
    """Randomly crop a region and produce GT and Reference images."""
    _, H, W = img.shape
    hh, ww = H - area_size, W - area_size

    start_h = random.randint(0, hh)
    start_w = random.randint(0, ww)

    # Step 1: Crop a 1024x1024 area
    img_area = img[:, start_h:start_h + area_size, start_w:start_w + area_size]

    # Step 2: Generate GT and Reference
    img_gt = random_crop_(img_area, crop_size=crop_size)
    
    img_area_rotated = rotate(img_area, angle_range=(0, 10))
    img_ref = random_crop_(img_area_rotated, crop_size=crop_size)
    
    
    return img_gt, img_ref

def random_crop_area_with_warp(img, area_size=2048, crop_size=1024):
    """Randomly crop a region and produce GT and Reference images."""
    _, H, W = img.shape
    hh, ww = H - area_size, W - area_size

    start_h = random.randint(0, hh)
    start_w = random.randint(0, ww)

    # Step 1: Crop a 1024x1024 area
    img_area = img[:, start_h:start_h + area_size, start_w:start_w + area_size]

    # Step 2: Generate GT and Reference
    img_gt = random_crop_(img_area, crop_size=crop_size)
    
    # img_area_rotated = rotate(img_area, angle_range=(0, 10))
    img_area_for_ref = random_spatial_augment(img_area)
    img_ref = random_crop_(img_area_for_ref, crop_size=crop_size)
    
    return img_gt, img_ref



def random_crop_just(img, crop_size=512):
    """Randomly crop a region and produce GT and Reference images."""
    # Step 2: Generate GT and Reference
    img_gt = random_crop_(img, crop_size=crop_size)
    # img_area_rotated = rotate(img_area, angle_range=(0, 10))
    # img_ref = random_crop_(img_area_rotated, crop_size=crop_size)
    
    img_ref = random_crop_(img, crop_size=crop_size)
    
    return img_gt, img_ref

def random_crop_warp(img, crop_size=512):
    """Randomly crop a region and produce GT and Reference images."""
    # Step 2: Generate GT and Reference
    img_gt = random_crop_(img, crop_size=crop_size)
    # img_area_rotated = rotate(img_area, angle_range=(0, 10))
    # img_ref = random_crop_(img_area_rotated, crop_size=crop_size)
    img_ref_img = random_spatial_augment(img)
    img_ref = random_crop_(img_ref_img, crop_size=crop_size)
    
    return img_gt, img_ref


def choose_area_crop_size(ori_height, ori_width):
    # 基准面积尺寸和裁剪尺寸
    base_area_size = 1024

    # 计算能够裁剪的最大区域比率
    max_height_ratio = ori_height // base_area_size
    max_width_ratio = ori_width // base_area_size
    ratio = min(max_height_ratio, max_width_ratio)

    # 计算裁剪区域的面积大小
    area_size = ratio * base_area_size
    crop_size = int(area_size * 3 /4)
    
    # if ratio == 0:
    #     area_size = 512, crop_size = 512

    return area_size, crop_size, ratio


def crop_images(img_gt, img_ref = None, crop_pad_size = 512, bilinear=True):
    """
    对 img_gt 和 img_ref 使用相同的裁剪方法，首先按比例放大到接近目标尺寸，然后中心裁剪。
    
    :param img_gt: 高质量图像，形状为 (H, W, C) 或 (H, W)
    :param img_ref: 参考图像，形状为 (H, W, C) 或 (H, W)
    :param crop_pad_size: 裁剪的目标尺寸
    :param bilinear: 是否使用双线性插值进行缩放
    :return: 裁剪后的 img_gt 和 img_ref
    """
    # 获取图像的高度和宽度
    h_gt, w_gt = img_gt.shape[0:2]
    
    # 先处理 img_gt，按照原逻辑裁剪
    if h_gt > crop_pad_size or w_gt > crop_pad_size:
        top = int((h_gt - crop_pad_size) // 2)
        left = int((w_gt - crop_pad_size) // 2)
        if bilinear:
            img_gt = cv2.resize(img_gt, (crop_pad_size, crop_pad_size), interpolation=cv2.INTER_LINEAR)
        else:
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

    if img_ref is not None:    
        h_ref, w_ref = img_ref.shape[0:2]

        # 计算 img_ref 缩放比例 r
        if h_ref > w_ref:
            new_w = 512
            new_h = int(512 * (h_ref / w_ref))

            # 缩放 img_ref
            img_ref_resized = cv2.resize(img_ref, (new_w, new_h), interpolation=cv2.INTER_LINEAR if bilinear else cv2.INTER_NEAREST)
            
        else:
            new_h = 512
            new_w = int(512 * (w_ref / h_ref))
            # 缩放 img_ref
            img_ref_resized = cv2.resize(img_ref, (new_w, new_h), interpolation=cv2.INTER_LINEAR if bilinear else cv2.INTER_NEAREST)

        # 计算裁剪区域，进行 center crop
        top_ref = (new_h - crop_pad_size) // 2
        left_ref = (new_w - crop_pad_size) // 2
        img_ref = img_ref_resized[top_ref:top_ref + crop_pad_size, left_ref:left_ref + crop_pad_size, ...]

        return img_gt, img_ref
    
    return img_gt

# def augment_ref(img_ref):
    


def warp_input(image: np.ndarray) -> np.ndarray:
    """
    对输入图像应用随机仿射变换（包括旋转、缩放、平移和剪切的组合）。

    :param image: 输入图像，NumPy 数组，形状为 (C, H, W)。
    :return: 经过仿射变换后的图像，形状为 (C, H, W)。
    """
    # 1. 形状转换: C, H, W -> H, W, C
    # 如果图像是灰度图 (C=1)，则形状可能只有 (H, W)，需要处理
    if image.ndim == 3:
        H, W = image.shape[1], image.shape[2]
        img_hwc = image.transpose(1, 2, 0)
    elif image.ndim == 2:
        H, W = image.shape[0], image.shape[1]
        img_hwc = image # 灰度图直接使用
    else:
        raise ValueError(f"Unsupported image dimension: {image.ndim}")

    # 2. 定义仿射变换参数范围
    
    # 随机选择中心点（通常是图像中心，但可以略微随机化）
    center_x = W / 2 + random.uniform(-W * 0.05, W * 0.05)
    center_y = H / 2 + random.uniform(-H * 0.05, H * 0.05)
    center = (center_x, center_y)

    # 旋转角度 (例如 -15 到 15 度)
    angle = random.uniform(-15, 15)

    # 缩放比例 (例如 0.8 到 1.2)
    scale = random.uniform(0.8, 1.2)
    
    # 平移量 (例如不超过图像尺寸的 5%)
    tx = random.uniform(-W * 0.05, W * 0.05)
    ty = random.uniform(-H * 0.05, H * 0.05)
    
    # 剪切/倾斜 (Shear) - 通过改变变换矩阵中的点来实现更复杂的视角变化
    # 这里我们简化，使用 OpenCV 的 getRotationMatrix2D 和手动平移来代替全仿射。
    
    # 3. 计算旋转和缩放的变换矩阵 M
    # M_rotate_scale = cv2.getRotationMatrix2D(center, angle, scale)

    # 4. 创建 3x3 仿射变换矩阵
    
    # 定义原图像的三个点（左上、右上、左下）
    pts1 = np.float32([
        [0, 0], 
        [W - 1, 0], 
        [0, H - 1]
    ])
    
    # 定义目标图像的三个点，加入随机的旋转、缩放和平移
    
    # 随机参数 for Perspective Warp (更强的视角变化)
    max_offset_percent = 0.1 # 最大点偏移为图像尺寸的 10%

    # 计算旋转矩阵
    M_rot = cv2.getRotationMatrix2D(center, angle, scale)

    # 应用旋转和缩放
    pts1_homogeneous = np.hstack([pts1, np.ones((3, 1))]) # 转换为齐次坐标 (x, y, 1)
    pts2_rot = np.dot(M_rot, pts1_homogeneous.T).T # 应用旋转矩阵
    
    # 应用平移
    pts2 = pts2_rot.astype(np.float32)
    pts2[:, 0] += tx
    pts2[:, 1] += ty

    # 添加少量随机扰动以模拟剪切/倾斜效果
    pts2[0] += [random.uniform(-W * max_offset_percent, W * max_offset_percent), 
                random.uniform(-H * max_offset_percent, H * max_offset_percent)]
    pts2[1] += [random.uniform(-W * max_offset_percent, W * max_offset_percent), 
                random.uniform(-H * max_offset_percent, H * max_offset_percent)]
    pts2[2] += [random.uniform(-W * max_offset_percent, W * max_offset_percent), 
                random.uniform(-H * max_offset_percent, H * max_offset_percent)]


    # 5. 获取最终的仿射变换矩阵 M
    M = cv2.getAffineTransform(pts1, pts2)
    
    # 6. 应用仿射变换
    # 使用 INTER_LINEAR 进行双线性插值
    warped_img_hwc = cv2.warpAffine(
        img_hwc, 
        M, 
        (W, H), 
        borderMode=cv2.BORDER_REFLECT_101, # 使用反射边界模式填充变换后的空白区域
        flags=cv2.INTER_LINEAR
    )

    # 7. 形状转换: H, W, C -> C, H, W (如果原图是彩色图)
    if image.ndim == 3:
        warped_img = warped_img_hwc.transpose(2, 0, 1)
    else: # 灰度图
        warped_img = warped_img_hwc

    return warped_img



def random_spatial_augment(image: np.ndarray, max_perspective_ratio: float = 0.1, 
                           max_affine_offset_ratio: float = 0.1) -> np.ndarray:
    """
    对输入图像应用随机透视变换，并通过裁剪和缩放确保输出图像完全填充，不留黑边。

    :param image: 输入图像，NumPy 数组，形状为 (C, H, W)。
    :param max_perspective_ratio: 控制透视变换的强度。
    :param max_affine_offset_ratio: 基础平移/缩放的随机偏移强度。
    :return: 经过空间变换后的图像，形状为 (C, H, W)。
    """
    # 1. 形状转换: C, H, W -> H, W, C
    if image.ndim == 3:
        H, W = image.shape[1], image.shape[2]
        img_hwc = image.transpose(1, 2, 0).astype(np.float32)
    elif image.ndim == 2:
        H, W = image.shape[0], image.shape[1]
        img_hwc = image.astype(np.float32)
    else:
        raise ValueError(f"Unsupported image dimension: {image.ndim}")
        
    # 将 HWC 图像转换为 3x3 矩阵，以便处理单通道和多通道图像
    if img_hwc.ndim == 2:
        img_hwc = np.expand_dims(img_hwc, axis=2)


    # 2. 定义原图的四个角点 (用于计算变换矩阵)
    pts_src = np.float32([
        [0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]
    ])
    
    # 3. 定义目标图像的四个角点 (加入随机扰动)
    pts_dst = np.float32(pts_src.copy())
    
    # a) 基础平移 (仿射效果)
    max_dx_affine = W * max_affine_offset_ratio
    max_dy_affine = H * max_affine_offset_ratio
    global_tx = random.uniform(-max_dx_affine, max_dx_affine)
    global_ty = random.uniform(-max_dy_affine, max_dy_affine)
    pts_dst[:, 0] += global_tx
    pts_dst[:, 1] += global_ty
    
    # b) 引入透视失真 (XYZ 视角效果)
    max_dx_perspective = W * max_perspective_ratio
    max_dy_perspective = H * max_perspective_ratio

    # 引入非对称的透视扰动
    pts_dst[0] += [random.uniform(0, max_dx_perspective), random.uniform(0, max_dy_perspective)] 
    pts_dst[1] += [random.uniform(-max_dx_perspective, 0), random.uniform(0, max_dy_perspective)] 
    pts_dst[2] += [random.uniform(-max_dx_perspective, 0), random.uniform(-max_dy_perspective, 0)]
    pts_dst[3] += [random.uniform(0, max_dx_perspective), random.uniform(-max_dy_perspective, 0)] 
    
    # 4. 获取透视变换矩阵 M
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    
    # --- 关键步骤：计算边界并裁剪/缩放 ---
    
    # 将原图角点应用到变换矩阵 M，找出它们在目标坐标系中的位置
    # corners_warped 的形状为 (1, 4, 2)
    corners_warped = cv2.perspectiveTransform(pts_src[np.newaxis, :, :], M)
    
    # 找出变换后图像内容的边界 (最小外接矩形)
    xmin = np.min(corners_warped[:, :, 0])
    xmax = np.max(corners_warped[:, :, 0])
    ymin = np.min(corners_warped[:, :, 1])
    ymax = np.max(corners_warped[:, :, 1])
    
    # 变换后的图像尺寸
    new_w = int(np.ceil(xmax - xmin))
    new_h = int(np.ceil(ymax - ymin))
    
    # 裁剪/平移矩阵 T_crop，将变换后的内容平移到新图像 (0, 0)
    T_crop = np.array([
        [1, 0, -xmin],
        [0, 1, -ymin],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 最终变换矩阵 M_final = T_crop * M
    M_final = T_crop @ M
    
    # 5. 应用最终变换，输出尺寸为 Bounding Box 的尺寸
    warped_img_hwc_cropped = cv2.warpPerspective(
        img_hwc, 
        M_final, 
        (new_w, new_h), 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=0,
        flags=cv2.INTER_LINEAR
    )
    
    # 6. 缩放回原始尺寸 (H, W)，确保图像被完全填充
    if new_w > 0 and new_h > 0:
        warped_img_hwc_resized = cv2.resize(
            warped_img_hwc_cropped, 
            (W, H), 
            interpolation=cv2.INTER_LINEAR
        )
    else:
        # 如果尺寸无效，返回原图 (作为回退机制)
        warped_img_hwc_resized = img_hwc

    # 7. 形状转换并返回 (转换为 C, H, W)
    warped_img_hwc_resized = np.clip(warped_img_hwc_resized, 0, 255).astype(np.uint8)
    if image.ndim == 3:
        warped_img = warped_img_hwc_resized.transpose(2, 0, 1)
    else:
        warped_img = warped_img_hwc_resized[:, :, 0] # 恢复灰度图的维度

    return warped_img