import pandas as pd
import cv2
import numpy as np
import tqdm

#one-hot表示多标签
import numpy as np
import os
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
    def __init__(self, img_dir, information_file, cache_dir="./cache"):
        self.img_dir = img_dir
        self.information = pd.read_excel(information_file)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.one_hot = ["N", "D", "G", "C", "A", "H", "M", "O"]
        self._preprocess_and_cache()

    def preprocess_img(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return np.ones((256, 256, 3)) * 255  # 这里可以返回一个全白图像作为默认值
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"Before CLAHE, img min: {img.min()}, max: {img.max()}")
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        gamma = 1.0
        img_gamma = np.power(img_clahe / 255.0, gamma) * 255.0
        
        img_gamma = img_gamma.astype(np.uint8)
        return cv2.resize(img_gamma, (256, 256))


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
                img = self.merge_double_imgs(left_path, right_path) / 255.0
                for idx in one_hot.keys():
                    if row[idx] == 1:
                        label = one_hot[idx]
                np.savez_compressed(cache_path, img=img, label=label)

    def __getitem__(self, index):
        cache_path = os.path.join(self.cache_dir, f"{index}.npz")
        data = np.load(cache_path)
        return data["img"],data["label"]


    def __len__(self):
        return len(self.information)


if __name__ == "__main__":
    d1 = PreprocessAndCache("./Training_Dataset","Traning_Dataset.xlsx")
