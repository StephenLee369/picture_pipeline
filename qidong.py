import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from raft import RAFT
#from utils import flow_viz
#from utils.utils import InputPadder
from models1.birefnet import BiRefNet  # 引入前景分割模型
from torchvision import transforms
DEVICE = 'cuda'
def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel
def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image
class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def load_birefnet(model_path):
    """加载BiRefNet前景分割模型"""
    model = BiRefNet.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    model.half()
    return model

def get_foreground_mask(birefnet, image_path):
    """使用BiRefNet获取前景掩码"""
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    input_images = transform_image(image).unsqueeze(0).to('cuda').half()

    # Prediction
    with torch.no_grad():
        pred = birefnet(input_images)[-1].sigmoid().cpu()

    # 转为二值掩码并调整回原始尺寸
    pred = pred[0].squeeze()
    mask = (pred > 0.5).cpu().numpy().astype(np.uint8) * 255
    original_img = Image.open(image_path)
    mask = Image.fromarray(mask).resize(original_img.size)
    return np.array(mask) > 128  # 转为布尔掩码

def analyze_flow(flow, foreground_mask, threshold=1.0):
    """分析光流，计算前景和背景运动指标"""
    flow_np = flow[0].permute(1, 2, 0).cpu().numpy()
    flow_magnitude = np.sqrt(np.sum(flow_np**2, axis=2))

    # 计算前景和背景区域的运动指标
    foreground_flow = flow_magnitude[foreground_mask]
    background_flow = flow_magnitude[~foreground_mask]

    # 前景运动：前景区域平均光流大小
    foreground_motion = np.mean(foreground_flow) if foreground_flow.size > 0 else 0

    # 背景运动：背景区域平均光流大小
    background_motion = np.mean(background_flow) if background_flow.size > 0 else 0

    # 前景运动像素比例
    foreground_moving_pixels = np.sum(foreground_flow > threshold) / foreground_flow.size if foreground_flow.size > 0 else 0

    return {
        'foreground_motion': foreground_motion,
        'background_motion': background_motion,
        'foreground_moving_pixels': foreground_moving_pixels
    }

def is_valid_frame_pair(metrics, foreground_threshold=5.0, background_threshold=0.5, motion_pixels_threshold=0.1):
    """判断帧对是否满足前景运动、背景静止的条件"""
    return (metrics['foreground_motion'] > foreground_threshold and 
            metrics['background_motion'] < background_threshold and
            metrics['foreground_moving_pixels'] > motion_pixels_threshold)
def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)
def save_result(img, flo, output_dir, idx, metrics, is_valid):
    """保存光流分析结果"""
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()

    # 将光流转换为RGB图像
    flo_rgb = flow_to_image(flo)

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 保存合并后的图像
    img_flo = np.concatenate([img, flo_rgb], axis=0)  # 垂直拼接原图和光流

    # 添加运动指标文本
    status_text = f"Valid: {is_valid}"
    motion_text = f"FG Motion: {metrics['foreground_motion']:.2f}, BG Motion: {metrics['background_motion']:.2f}"
    pixels_text = f"FG Moving Pixels: {metrics['foreground_moving_pixels']:.2%}"

    cv2.putText(img_flo, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if is_valid else (0, 0, 255), 2)
    cv2.putText(img_flo, motion_text, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img_flo, pixels_text, (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 保存结果图像
    result_path = os.path.join(output_dir, f'result_{idx:04d}_{"valid" if is_valid else "invalid"}.png')
    cv2.imwrite(result_path, img_flo[:, :, ::-1])  # 转换为BGR格式

    print(f"已保存结果到: {result_path}")
    return result_path

def demo(args):
    # 加载模型
    print('aaaaaaa')
    raft_model = torch.nn.DataParallel(RAFT(args))
    raft_model.load_state_dict(torch.load(args.model))
    raft_model = raft_model.module
    raft_model.to(DEVICE)
    raft_model.eval()

    birefnet = load_birefnet(args.birefnet_model)

    # 创建输出目录
    output_dir = os.path.join(args.path, 'flow_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # 记录有效帧对
    valid_pairs = []

    with torch.no_grad():
        # 获取所有图像并排序
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        images = sorted(images)

        print(f"找到 {len(images)} 张图像，将分析 {len(images)-1} 个相邻帧对")

        for i, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
            print(f"\n分析帧对 {i+1}/{len(images)-1}:")
            print(f"  帧1: {os.path.basename(imfile1)}")
            print(f"  帧2: {os.path.basename(imfile2)}")

            # 加载图像
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            # 填充图像以匹配RAFT模型的要求
            padder = InputPadder(image1.shape)
            padded_image1, padded_image2 = padder.pad(image1, image2)

            # 计算光流
            flow_low, flow_up = raft_model(padded_image1, padded_image2, iters=20, test_mode=True)

            # 获取前景掩码（使用第一帧的掩码作为参考）
            foreground_mask = get_foreground_mask(birefnet, imfile1)

            # 分析光流
            motion_metrics = analyze_flow(flow_up, foreground_mask)

            # 判断帧对是否有效
            valid = is_valid_frame_pair(motion_metrics)
            print(f"{valid}")
            # 保存结果
            result_path = save_result(image1, flow_up, output_dir, i, motion_metrics, valid)

            # 记录有效帧对
            if valid:
                valid_pairs.append({
                    'index': i,
                    'frame1': imfile1,
                    'frame2': imfile2,
                    'metrics': motion_metrics,
                    'result_path': result_path
                })

    # 保存有效帧对列表
    if valid_pairs:
        report_path = os.path.join(output_dir, 'valid_frame_pairs.txt')
        with open(report_path, 'w') as f:
            f.write("有效帧对报告:\n")
            f.write("="*50 + "\n")
            for pair in valid_pairs:
                f.write(f"帧对 #{pair['index']}:\n")
                f.write(f"  帧1: {os.path.basename(pair['frame1'])}\n")
                f.write(f"  帧2: {os.path.basename(pair['frame2'])}\n")
                f.write(f"  前景运动: {pair['metrics']['foreground_motion']:.2f}\n")
                f.write(f"  背景运动: {pair['metrics']['background_motion']:.2f}\n")
                f.write(f"  前景运动像素比例: {pair['metrics']['foreground_moving_pixels']:.2%}\n")
                f.write(f"  结果图像: {os.path.basename(pair['result_path'])}\n")
                f.write("-"*50 + "\n")

        print(f"\n找到 {len(valid_pairs)} 个有效帧对")
        print(f"有效帧对报告已保存到: {report_path}")
    else:
        print("\n未找到满足条件的有效帧对")
def run_on_multiple_subfolders(args):
    from glob import glob
    import os
    
    # 找到所有子目录
    subfolders = [f.path for f in os.scandir(args.multi_dir) if f.is_dir()]
    if not subfolders:
        print(f'在 {args.multi_dir} 下没有找到子文件夹，退出')
        return
    
    print(f'发现 {len(subfolders)} 个子文件夹:')
    for sub in subfolders:
        print(f'    {os.path.basename(sub)}')
    print('='*40)

    for idx, subfolder in enumerate(subfolders):
        print(f'\n=== 处理第 {idx+1}/{len(subfolders)} 个子文件夹：{subfolder} ===\n')
        # 构造新的参数对象
        from copy import deepcopy
        args_one = deepcopy(args)
        args_one.path = subfolder
        # 每个子文件夹的输出会在：子文件夹/flow_analysis 下
        try:
            demo(args_one)
        except Exception as e:
            print(f"处理{subfolder}时出错: {e}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="RAFT模型路径")
    parser.add_argument('--birefnet_model', default="ZhengPeng7/BiRefNet")
    parser.add_argument('--path', help="视频帧文件夹路径")
    parser.add_argument('--small', action='store_true', help='使用小模型')
    parser.add_argument('--mixed_precision', action='store_true', help='使用混合精度')
    parser.add_argument('--alternate_corr', action='store_true', help='使用高效相关实现')
    parser.add_argument(
        '--multi_dir', 
        default=None,
        help="如果指定，该路径下的所有一级子文件夹都会被批量分析"
    )
    args = parser.parse_args()

    if args.multi_dir:
        run_on_multiple_subfolders(args)
    else:
        demo(args)