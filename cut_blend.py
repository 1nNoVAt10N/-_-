# -*-coding:utf-8 -*-

import glob
import numpy as np
import torch
import os
import cv2
from unet_model import UNet

def preprocess_img(img):
    if img is None:
        return np.ones((384, 384, 3)) * 255  # 返回一个全白图像作为默认值

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # CLAHE 处理
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return img


def cut_blend(img_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)
    net.load_state_dict(torch.load(r"./best_model_drive.pth", map_location=device))
    net.eval()

    img = cv2.imread(img_path)
    origin_shape = img.shape
    img = preprocess_img(img)

    # 转为灰度图
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.resize(img_gray, (512, 512))

    # 转换为模型输入格式
    img_tensor = torch.from_numpy(img_gray.reshape(1, 1, 512, 512)).to(device=device, dtype=torch.float32)

    # 预测
    pred = net(img_tensor)
    pred = np.array(pred.data.cpu()[0])[0]

    # 处理预测结果
    pred[pred >= 0.5] = 255
    pred[pred < 0.5] = 0

    # 调整回原尺寸
    pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)

    # **生成彩色叠加图**
    color_mask = np.zeros_like(img)
    color_mask[:, :, 0] = pred  # 仅在红色通道添加血管区域

    # 叠加到原图上
    blended = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)

    return (pred, blended)

if __name__ == "__main__":
    save_dir = r'F:\Unet\unet_42-drive\images\predict_color'
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)
    net.load_state_dict(torch.load(r"F:\Unet\unet_42-drive/best_model_drive.pth", map_location=device))
    net.eval()

    tests_path = glob.glob(r'F:\BFPC\cropped_#Training_Dataset/*.jpg')

    for i, test_path in enumerate(tests_path):
        save_res_path = os.path.join(save_dir, os.path.basename(test_path))

        # 读取原始图像
        img = cv2.imread(test_path)
        origin_shape = img.shape
        img = preprocess_img(img)

        # 转为灰度图
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.resize(img_gray, (512, 512))

        # 转换为模型输入格式
        img_tensor = torch.from_numpy(img_gray.reshape(1, 1, 512, 512)).to(device=device, dtype=torch.float32)

        # 预测
        pred = net(img_tensor)
        pred = np.array(pred.data.cpu()[0])[0]

        # 处理预测结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0

        # 调整回原尺寸
        pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)

        # **生成彩色叠加图**
        color_mask = np.zeros_like(img)
        color_mask[:, :, 0] = pred  # 仅在红色通道添加血管区域

        # 叠加到原图上
        blended = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)

        # 保存彩色叠加结果
        cv2.imwrite(save_res_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        print(f"{i+1}: {test_path} 的预测结果已保存到 {save_res_path}")
        
