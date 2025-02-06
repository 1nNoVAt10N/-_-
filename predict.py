from model import ViTModel
import torch
import os
from data_preprocessing import PreprocessAndCache_for_single
import numpy as np
import zipfile

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

class Predict:
    def __init__(self,model_path,device,visualize=False):
        #模型还没搞好，这部分可以先不用看
        #self.model = ViTModel(num_classes=8)
        #self.model.load_state_dict(torch.load(model_path))
        #self.model.to(device)
        #self.model.eval()
        self.device = device

        self.visualize = visualize

    def predict(self,left_img=None,right_img=None,imgs = None,mode = "single"):
        answers = []
        #单个读取双目图片
        #这里左右图片全为路径

        if mode == "single":
            left_img_name,_ = os.path.splitext(left_img)
            left_img_name = left_img_name.replace(".","").replace("/","_")
            right_img_name,_ = os.path.splitext(right_img)
            right_img_name = right_img_name.replace(".","").replace("/","_")
            process = PreprocessAndCache_for_single(left_img,right_img,cache_dir="./temp_cache")
            single_data = np.load(f"./temp_cache/{left_img_name}_{right_img_name}.npz")
            data = single_data["img"]
            data = torch.tensor(data).to(self.device)
            #现在模型还没搞好，先随机输出一个矩阵
            #labels = self.model(data)
            labels = torch.rand(8)
            labels = (labels > 0.5).float()
            for i in range(8):
                if labels[i] > 0.5:
                    answers.append(one_hot_to_name[str(i)])
            return answers
        
        #批量读取双目图片
        if mode == "batch":
            #这里认为多读模式是读取一个压缩包,所以imgs是一个压缩包的路径
            if imgs is None:
                raise ValueError("压缩包路径不能为空")
            
            # 解压压缩包
            with zipfile.ZipFile(imgs, 'r') as zip_ref:
                zip_ref.extractall("./temp_images")  # 提取到临时文件夹
            
            # 获取解压后的图像文件列表
            image_files = [f for f in os.listdir("./temp_images") if f.endswith(('.jpg', '.png'))]
            batch_results = {}
            
            for img_file in image_files:
                left_img = f"./temp_images/{img_file.replace('right', 'left')}"
                right_img = f"./temp_images/{img_file.replace('left', 'right')}"
                
                if os.path.exists(left_img) and os.path.exists(right_img):
                    left_img_name, _ = os.path.splitext(left_img)
                    left_img_name = left_img_name.replace(".", "").replace("/", "_")
                    right_img_name, _ = os.path.splitext(right_img)
                    right_img_name = right_img_name.replace(".", "").replace("/", "_")
                    
                    process = PreprocessAndCache_for_single(left_img, right_img, cache_dir="./temp_cache")
                    single_data = np.load(f"./temp_cache/{left_img_name}_{right_img_name}.npz")
                    data = single_data["img"]
                    data = torch.tensor(data).to(self.device)
                    
                    # labels = self.model(data)
                    labels = torch.rand(8)
                    labels = (labels > 0.5).float()
                    
                    img_file_name,_ = os.path.splitext(img_file)
                    img_file_name = img_file_name.replace(".", "").replace("/", "_")
                    img_file_name = img_file_name.split("_")[0] + "_left_right"
                    if (img_file_name in batch_results.keys()):
                        continue
                    batch_results[img_file_name] = []
                    for i in range(8):
                        if labels[i] > 0.5:
                            batch_results[img_file_name].append(one_hot_to_name[str(i)])

            # 删除临时文件夹及解压的文件
            for file in os.listdir("./temp_images"):
                os.remove(os.path.join("./temp_images", file))
            os.rmdir("./temp_images")
            
            return batch_results




if __name__ == "__main__":
    p = Predict("./models/model_vit_1.pth",device="cpu")
    ans = p.predict("./Training_Dataset/0_left.jpg","./Training_Dataset/0_right.jpg")
    ans2 = p.predict(imgs="hhh.zip",mode="batch")
    print(ans)
    print(ans2)


