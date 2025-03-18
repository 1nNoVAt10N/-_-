import pandas as pd
import cv2
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from ietk import methods
from ietk import util
from ietk.data import IDRiD
#one-hot表示多标签
import os
# OpenCV实现MSR
def normalize_to_01(arr):
 
    arr = arr.astype(np.float64)
    # 计算数组的最小值和最大值
    min_val = np.min(arr)
    max_val = np.max(arr)
    if min_val == max_val:
        normalized_arr = np.zeros(arr.shape, dtype=np.float64)
    else:
        normalized_arr = (arr - min_val) / (max_val - min_val)
 
    return normalized_arr


def map_to_0_255(original_array):
    # 找到数组中的最小值和最大值
    min_value = np.min(original_array)
    max_value = np.max(original_array)
 
    # 检查最大值和最小值是否相同，避免除以零
    if max_value == min_value:
        raise ValueError("所有数值都相同，无法进行映射")
 
    # 初始化映射后的数组
    # mapped_array = []
 
    mapped_array = 255 * ((original_array - min_value) / (max_value - min_value))

    return mapped_array
one_hot = {
    "N":0,
    "D":1,
    "G":2,
    "C":3,
    "A":4,
    "H":5,
    "M":6,
    "O":7,
}

import os
import cv2
import tqdm
import numpy as np
import pandas as pd

def fov_extraction(img):
    # Read the image
  


    red_channel = img[:, :, 2]
    
    # Compute the centerlines
    h, w = red_channel.shape
    Hcenterline, Vcenterline = h // 2, w // 2
    
    # Extract intensity profiles along the centerlines
    horizontal_profile = red_channel[Hcenterline, :]
    vertical_profile = red_channel[:, Vcenterline]
    
    # Compute thresholds
    Hthreshold = np.max(horizontal_profile) * 0.06
    Vthreshold = np.max(vertical_profile) * 0.06
    
    # Identify transitions based on the threshold
    binary_horizontal_profile = (horizontal_profile > Hthreshold).astype(int)
    binary_vertical_profile = (vertical_profile > Vthreshold).astype(int)
    
    diff_horizontal = np.diff(binary_horizontal_profile)
    diff_vertical = np.diff(binary_vertical_profile)
    
    transitions_horizontal = np.where(diff_horizontal != 0)[0]
    transitions_vertical = np.where(diff_vertical != 0)[0]
    
    # Handle cases where no transitions are found
    if len(transitions_horizontal) < 2:
        transitions_horizontal = [0, w - 1]
    if len(transitions_vertical) < 2:
        transitions_vertical = [0, h - 1]
    
    # Use min and max transitions for cropping
    vertical_diff = transitions_vertical[-1] - transitions_vertical[0]
    horizontal_diff = transitions_horizontal[-1] - transitions_horizontal[0]
    
    # Validate transitions
    if horizontal_diff < w * 0.25:
        transitions_horizontal = [0, w - 1]
    if vertical_diff < h * 0.25:
        transitions_vertical = [0, h - 1]
    
    # Crop the image
    cropped_img = img[transitions_vertical[0]:transitions_vertical[-1], transitions_horizontal[0]:transitions_horizontal[-1]]
    

    return cropped_img

class PreprocessAndCache:
    def __init__(self, img_dir, information_file, cache_dir="./ALL"):
        self.img_dir = img_dir
        self.information = pd.read_excel(information_file)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.one_hot = ["N", "D", "G", "C", "A", "H", "M", "O"]
        self._preprocess_and_cache()

    def preprocess_img(self, img_path):
        img = cv2.imread(img_path)
        img = fov_extraction(img)
        if img is None:
            return np.ones((384, 384, 3)) * 255  # 返回一个全白图像作为默认值

        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # CLAHE 处理
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # 高斯模糊+对比度增强
        image_blur = cv2.GaussianBlur(img, (63, 63), sigmaX=10, sigmaY=10)
        img_org_float = img.astype(np.float32)
        img_blur_float = image_blur.astype(np.float32)
        alpha, beta, gamma = 4, -4, 128
        enhanced = alpha * img_org_float + beta * img_blur_float + gamma
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        return enhanced

    def merge_double_imgs(self, left_eye_path, right_eye_path):
        left_img = self.preprocess_img(left_eye_path)
        right_img = self.preprocess_img(right_eye_path)
        return np.concatenate([left_img, right_img], axis=1)

    def _preprocess_and_cache(self):
        for i, row in tqdm.tqdm(self.information.iterrows(), total=len(self.information)):
            left_path = os.path.join(self.img_dir, row['Left-Fundus'])
            right_path = os.path.join(self.img_dir, row['Right-Fundus'])
            cache_path = os.path.join(self.cache_dir, f"{i}.npz")

            if not os.path.exists(cache_path):
                img = self.merge_double_imgs(left_path, right_path)

                # 处理标签
                label = [1 if row.get(key, 0) == 1 else 0 for key in self.one_hot]
                label = np.array(label)

                # 处理诊断关键词
                left_keywords = row["Left-Diagnostic Keywords"] if pd.notna(row["Left-Diagnostic Keywords"]) else "Unknown"
                right_keywords = row["Right-Diagnostic Keywords"] if pd.notna(row["Right-Diagnostic Keywords"]) else "Unknown"

                # 保存 npz 文件
                np.savez_compressed(cache_path, img=img, label=label, 
                                    name=row["Left-Fundus"],
                                    left_keywords=left_keywords, 
                                    right_keywords=right_keywords)

    def __len__(self):
        return len(self.information)

    
class PreprocessAndCache_for_single:
    def __init__(self,left_img=None,right_img=None,text=None,cache_dir="./5K",pre=True):
        #这里的左右图片全为路径
        self.left_img = left_img
        self.right_img = right_img
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.one_hot = ["N", "D", "G", "C", "A", "H", "M", "O"]
        self.pre = pre
        self.text = text
        if self.pre:
            self._preprocess_and_cache()

    def preprocess_img(self, img_path):
        img = cv2.imread(img_path)
        img = fov_extraction(img)
        if img is None:
            return np.ones((384, 384, 3)) * 255  # 返回一个全白图像作为默认值

        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # CLAHE 处理
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # 高斯模糊+对比度增强
        image_blur = cv2.GaussianBlur(img, (63, 63), sigmaX=10, sigmaY=10)
        img_org_float = img.astype(np.float32)
        img_blur_float = image_blur.astype(np.float32)
        alpha, beta, gamma = 4, -4, 128
        enhanced = alpha * img_org_float + beta * img_blur_float + gamma
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        return enhanced



    def merge_double_imgs(self, left_eye_path, right_eye_path):
        left_img = self.preprocess_img(left_eye_path)
        right_img = self.preprocess_img(right_eye_path)
        return np.concatenate([left_img, right_img], axis=1)

    def _preprocess_and_cache(self):
        #获取文件名
        if self.text is None:
            left_img_name = os.path.splitext(os.path.basename(self.left_img))[0]
            right_img_name = os.path.splitext(os.path.basename(self.right_img))[0]
            cache_path = os.path.join(self.cache_dir, f"{left_img_name}_{right_img_name}.npz")
            if not os.path.exists(cache_path):
                img = self.merge_double_imgs(self.left_img, self.right_img) 
                np.savez_compressed(cache_path, img=img)
        else:
            left_img_name = os.path.splitext(os.path.basename(self.left_img))[0]
            right_img_name = os.path.splitext(os.path.basename(self.right_img))[0]
            cache_path = os.path.join(self.cache_dir, f"{left_img_name}_{right_img_name}.npz")
            left_text = self.text["left_text"]
            right_text = self.text["right_text"]
            if not os.path.exists(cache_path):
                img = self.merge_double_imgs(self.left_img, self.right_img) 
                np.savez_compressed(cache_path, img=img,left_keywords=left_text, 
                                    right_keywords=right_text)

    def __getitem__(self, index):
        cache_path = os.path.join(self.cache_dir, f"{index}.npz")
        data = np.load(cache_path)
        return data["img"]


    def __len__(self):
        return 1


if __name__ == "__main__":
    print(one_hot.keys())
    # d1 = PreprocessAndCache("F:\waibao\cropped_5K","F:\Retinal-disease-foundational-model-for-ODIR2019-main\data\ODIR-5K_Training_Annotations(Updated)_V2.xlsx")
    d1 = PreprocessAndCache("F:\BFPC/real_full\cropped","F:\BFPC\combined_file.xlsx")