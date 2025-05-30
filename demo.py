import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def save_result(img, flo, output_dir, idx):
    """将图像和光流结果保存为图片文件而不是显示"""
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()

    # 将光流转换为RGB图像
    flo_rgb = flow_viz.flow_to_image(flo)

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 保存原始图像
    img_path = os.path.join(output_dir, f'frame_{idx:04d}.png')
    cv2.imwrite(img_path, img[:, :, ::-1])  # 转换为BGR格式

    # 保存光流图像
    flo_path = os.path.join(output_dir, f'flow_{idx:04d}.png')
    cv2.imwrite(flo_path, flo_rgb[:, :, ::-1])  # 转换为BGR格式

    # 保存合并后的图像
    img_flo = np.concatenate([img, flo_rgb], axis=0)  # 垂直拼接原图和光流
    combined_path = os.path.join(output_dir, f'combined_{idx:04d}.png')
    cv2.imwrite(combined_path, img_flo[:, :, ::-1])  # 转换为BGR格式

    print(f"已保存结果到: {output_dir} (帧 {idx})")

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    # 创建输出目录
    output_dir = os.path.join(args.path, 'flow_results')
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        images = sorted(images)
        for i, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            save_result(image1, flow_up, output_dir, i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)