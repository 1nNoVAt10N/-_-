import pandas as pd
import cv2
import numpy as np
import tqdm
import matplotlib.pyplot as plt
#one-hot表示多标签
import numpy as np
import os
# OpenCV实现MSR




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
class PreprocessAndCache:
    def __init__(self, img_dir, information_file, cache_dir="./5k"):
        self.img_dir = img_dir
        self.information = pd.read_excel(information_file)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.one_hot = ["N", "D", "G", "C", "A", "H", "M", "O"]
        self._preprocess_and_cache()

    def preprocess_img(self,img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            
            if img is None:
                return np.ones((384, 384, 3)) * 255  # 这里可以返回一个全白图像作为默认值
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            print(f"Before CLAHE, img min: {img.min()}, max: {img.max()}")
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            image_blur = cv2.GaussianBlur(img, (63, 63), sigmaX=10, sigmaY=10)
            
            # 转换为浮点类型以进行数学运算
            img_org_float = img.astype(np.float32)
            img_blur_float = image_blur.astype(np.float32)
            alpha = 4
            beta = -4
            gamma = 128
            # 执行加权增强计算
            enhanced = alpha * img_org_float + beta * img_blur_float + gamma
            
            # 限制像素值范围并转换回uint8类型
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)


            return enhanced


    def merge_double_imgs(self, left_eye_path, right_eye_path):
        left_img = self.preprocess_img(left_eye_path)
        right_img = self.preprocess_img(right_eye_path)
        return np.concatenate([left_img, right_img], axis=-1)

    def _preprocess_and_cache(self):
        for i, row in tqdm.tqdm(self.information.iterrows()):
            left_path = f"{self.img_dir}/{row['Left-Fundus']}"
            right_path = f"{self.img_dir}/{row['Right-Fundus']}"
            cache_path = os.path.join(self.cache_dir, f"{i}.npz")
            if not os.path.exists(cache_path):
                img = self.merge_double_imgs(left_path, right_path) 
                label = []
                for idx in one_hot.keys():
                    if row[idx] == 1:
                        label.append(1)
                    else:
                        label.append(0)
                label = np.array(label)
                np.savez_compressed(cache_path, img=img, label=label)

    def __getitem__(self, index):
        alldata = self.information.iterrows()
        row = alldata[index]
        left_path = f"./cropped_#Training_Dataset/{row['Left-Fundus']}"
        right_path = f"./cropped_#Training_Dataset/{row['Right-Fundus']}"
        img = self.merge_double_imgs(left_path, right_path) / 255.0
        label = []
        for idx in one_hot.keys():
            if row[idx] == 1:
                label.append(1)
            else:
                label.append(0)
        label = np.array(label)
        return img,label


    def __len__(self):
        return len(self.information)
    
class PreprocessAndCache_for_single:
    def __init__(self,left_img,right_img,cache_dir="./5K"):
        #这里的左右图片全为路径
        self.left_img = left_img
        self.right_img = right_img
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.one_hot = ["N", "D", "G", "C", "A", "H", "M", "O"]
        self._preprocess_and_cache()

    def preprocess_img(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (384, 384))
            
            if img is None:
                return np.ones((384, 384, 3)) * 255  # 这里可以返回一个全白图像作为默认值
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            print(f"Before CLAHE, img min: {img.min()}, max: {img.max()}")
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            image_blur = cv2.GaussianBlur(img, (63, 63), sigmaX=10, sigmaY=10)
            
            # 转换为浮点类型以进行数学运算
            img_org_float = img.astype(np.float32)
            img_blur_float = image_blur.astype(np.float32)
            alpha = 4
            beta = -4
            gamma = 128
            # 执行加权增强计算
            enhanced = alpha * img_org_float + beta * img_blur_float + gamma
            
            # 限制像素值范围并转换回uint8类型
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)


            return enhanced



    def merge_double_imgs(self, left_eye_path, right_eye_path):
        left_img = self.preprocess_img(left_eye_path)
        right_img = self.preprocess_img(right_eye_path)
        return np.concatenate([left_img, right_img], axis=-1)

    def _preprocess_and_cache(self):
        #获取文件名
        left_img_name,_ = os.path.splitext(self.left_img)
        left_img_name = left_img_name.replace(".","").replace("/","_")
        right_img_name,_ = os.path.splitext(self.right_img)
        right_img_name = right_img_name.replace(".","").replace("/","_")
        cache_path = f"{self.cache_dir}/{left_img_name}_{right_img_name}.npz"
        if not os.path.exists(cache_path):
            img = self.merge_double_imgs(self.left_img, self.right_img) 
            np.savez_compressed(cache_path, img=img)

    def __getitem__(self, index):
        cache_path = os.path.join(self.cache_dir, f"{index}.npz")
        data = np.load(cache_path)
        return data["img"]


    def __len__(self):
        return 1


if __name__ == "__main__":
    print(one_hot.keys())
    d1 = PreprocessAndCache("F:\waibao\cropped_5K","F:\Retinal-disease-foundational-model-for-ODIR2019-main\data\ODIR-5K_Training_Annotations(Updated)_V2.xlsx")
