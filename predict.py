
import torch
import os
from data_preprocessing import PreprocessAndCache_for_single
import numpy as np
import zipfile
from model import BFPCNet1
import pandas as pd
import albumentations as A
from transformers import AutoTokenizer, AutoModel
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
mean = IMAGENET_DEFAULT_MEAN
std = IMAGENET_DEFAULT_STD
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

one_hot_to_name = {
    "0":"正常",
    "1":"糖尿病",
    "2":"青光眼",
    "3":"白内障",
    "4":"AMD",
    "5":"高血压",
    "6":"近视",
    "7":"其他疾病/异常 ",
}
def get_augmentations2():
    return A.Compose([
        A.Normalize(mean, std),
],additional_targets={'right':'image'})
class Predict:
    def __init__(self,model_path,device,visualize=False):
        #模型还没搞好，这部分可以先不用看
        self.model = BFPCNet1(num_classes=8)
        self.model.load_state_dict(torch.load(model_path),strict=False)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("./biobert_model/")
        self.bertmodel = AutoModel.from_pretrained("./biobert_model/") 
        self.transform = get_augmentations2()
        self.visualize = visualize



    def predict(self, left_img=None, right_img=None, texts=None, imgs=None, xlxs=None, mode="single"):
        answers = []
        
        if texts is None:  # 处理无文本情况
            if mode == "single":
                # 仅保留文件名
                left_img_name = os.path.splitext(os.path.basename(left_img))[0]
                right_img_name = os.path.splitext(os.path.basename(right_img))[0]

                process = PreprocessAndCache_for_single(left_img, right_img, cache_dir="./temp_cache")
                cache_path = f"./temp_cache/{left_img_name}_{right_img_name}.npz"

                single_data = np.load(cache_path)
                data = single_data["img"]
                data = torch.tensor(data).to(self.device).permute(2, 0, 1)
                with torch.no_grad():
                    labels = self.model(data)
                labels = labels.squeeze(0)
                print(labels)
                labels = (labels > 0.5).float()
                print(labels)

                for i in range(8):
                    if labels[i] == 1:
                        answers.append(one_hot_to_name[str(i)])

                return answers
            
            elif mode == "batch":
                if xlxs is None:
                    raise ValueError("信息表路径不能为空")
                
                # 读取信息表
                df = pd.read_excel(xlxs)
                batch_results = {}

                for _, row in df.iterrows():
                    left_img_path = os.path.join(imgs, row['Left-Funds'])
                    right_img_path = os.path.join(imgs, row['Right-Funds'])

                    if os.path.exists(left_img_path) and os.path.exists(right_img_path):
                        left_img_name = os.path.splitext(os.path.basename(left_img_path))[0]
                        right_img_name = os.path.splitext(os.path.basename(right_img_path))[0]

                        process = PreprocessAndCache_for_single(left_img_path, right_img_path, cache_dir="./temp_cache")
                        cache_path = f"./temp_cache/{left_img_name}_{right_img_name}.npz"

                        single_data = np.load(cache_path)
                        data = single_data["img"]
                        data = torch.tensor(data).to(self.device)

                        labels = self.model(data)
                        labels = (labels > 0.5).float()

                        img_file_name = f"{left_img_name}_left_right"

                        if img_file_name in batch_results:
                            continue
                        
                        batch_results[img_file_name] = []
                        for i in range(8):
                            if labels[i] > 0.5:
                                batch_results[img_file_name].append(one_hot_to_name[str(i)])

                return batch_results
        
        else:  # 处理带文本情况
            if mode == "single":
                left_img_name = os.path.splitext(os.path.basename(left_img))[0]
                right_img_name = os.path.splitext(os.path.basename(right_img))[0]

                process = PreprocessAndCache_for_single(left_img, right_img, cache_dir="./temp_cache",text=texts)
                cache_path = f"./temp_cache/{left_img_name}_{right_img_name}.npz"

                single_data = np.load(cache_path)
                data = single_data["img"]
                l = data[:, :224, :]
                r = data[:, 224:, :]
                au = self.transform(image = l,right=r)
                data = np.concatenate((au['image'],au['right']), axis=1)

 
                texts = str(single_data["left_keywords"]) + "," + str(single_data["right_keywords"])

                inputs = self.tokenizer(texts, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.bertmodel(**inputs)
                    last_hidden_state = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_dim)
                    cls_embedding = last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_dim)
                    text_embedding = cls_embedding

                data = torch.tensor(data).to(self.device).permute(2, 0, 1).float()
                data = data.unsqueeze(0)
                labels = self.model(data, text_embedding)
                labels = labels.squeeze(0)
                print(labels)
                labels = (labels > 0.5).float()
                print(labels)

                for i in range(8):
                    if labels[i] == 1:
                        answers.append(one_hot_to_name[str(i)])

                return answers





if __name__ == "__main__":
    annotation_path = r"F:\BFPC/real_full/Off-site Test Set\Annotation/off-site test annotation (English).xlsx"
    image_folder = r"F:\BFPC/real_full/Off-site Test Set/Images"  # 存储图像的文件夹
    output_csv = r"F:\BFPC/real_full/Off-site Test Set/predictions.csv"  # 输出的 CSV 文件路径

    # 加载预测模型
    p = Predict("F:\BFPC/final_model_state_dict_with_gate.pth", device="cpu")

    # 读取 Excel 文件
    df = pd.read_excel(annotation_path)

    # 结果存储列表
    results = []

    # 遍历所有行进行预测
    for _, row in df.iterrows():
        left_img_path = os.path.join(image_folder, row["Left-Fundus"])
        right_img_path = os.path.join(image_folder, row["Right-Fundus"])

        # 确保图像文件存在
        if not os.path.exists(left_img_path) or not os.path.exists(right_img_path):
            print(f"警告：未找到图像 {left_img_path} 或 {right_img_path}，跳过...")
            continue

        # 构造文本输入
        text = {
            "left_text": str(row["Left-Diagnostic Keywords"]),
            "right_text": str(row["Right-Diagnostic Keywords"])
        }

        # 进行预测
        predictions = p.predict(left_img_path, right_img_path, texts=text)

        # 存储结果
        results.append([row["ID"]] + [1 if one_hot_to_name[str(i)] in predictions else 0 for i in range(8)])

    # 创建 DataFrame 并保存为 CSV
    columns = ["ID"] + list(one_hot_to_name.values())  # 列名
    df_results = pd.DataFrame(results, columns=columns)
    df_results.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"预测完成，结果已保存至 {output_csv}")


