from model_with_gate import BFPCNet1
from data_utils import EyeDataset
from data_preprocessing import PreprocessAndCache
from torch.utils.data import DataLoader
import torch
import tqdm
from sklearn import metrics 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    precision_recall_fscore_support,classification_report
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler
from safetensors.torch import load_file
import warnings
warnings.filterwarnings("ignore")


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        """
        Focal Loss for multi-class classification.

        :param alpha: 主要是调整类别不平衡的因子（可选），默认0.25。
        :param gamma: 调整易分类样本的权重，通常取值为2。
        :param reduction: 损失函数的归约方式，'mean', 'sum' 或 'none'。
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        :param inputs: 预测值，通常是网络输出的logits，大小为 (batch_size, num_classes)
        :param targets: 目标标签，大小为 (batch_size,)
        """

        p = torch.sigmoid(inputs)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')  # 计算交叉熵损失
        pt = p * targets + (1 - p) * (1 - targets)

        loss = BCE_loss * ((1 - pt) ** self.gamma)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss



class train_val_test:
    def __init__(self, epoch, lr, batch_size, num_workers, device, model, optimizer, criterion,scheduler):
        self.epoch = epoch
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

    def train(self, train_data, val_data):
        # 训练代码
        self.model.train()
        train_loader = DataLoader(train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

        for epoch in range(self.epoch):
            running_loss = 0.0
            for inputs, texts,targets in tqdm.tqdm(train_loader,desc = "training"):
                inputs, texts,targets = inputs.to(self.device),texts.to(self.device), targets.to(self.device)
                #print(targets)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{self.epoch}], Loss: {running_loss / len(train_loader):.4f}, LR: {self.lr}")

            # 在每个epoch后进行验证
            val_loss, val_accuracy = self.validate(val_loader)
            self.scheduler.step(val_loss)
            print(f"Epoch [{epoch + 1}] Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # 最终返回整个训练过程的最后验证结果
        torch.save(self.model.state_dict(), "final_model_state_dict.pth")
        final_val_loss, final_val_accuracy = self.validate(val_loader)
        return final_val_loss, final_val_accuracy  # 返回最后的验证损失和准确率

    def validate(self, val_dataloader):
        self.model.eval()

        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, texts,target in tqdm.tqdm(val_dataloader,desc = "val"):
                data, texts,target = data.to(self.device), texts.to(self.device),target.to(self.device)
                output = self.model(data)
                
                loss = self.criterion(output, target)
                val_loss += loss.item()
                preds = torch.sigmoid(output)
                #print(preds)
                preds = (preds > 0.5).float()  # 二分类阈值0.5
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        # 计算平均损失
        val_loss /= len(val_dataloader)

        # 转换为 numpy 数组
        all_preds = np.array(all_preds)
        
        all_targets = np.array(all_targets)
        fla_preds = all_preds.flatten()
        fla_targets = all_targets.flatten()

        # 计算其他指标
        accuracy = accuracy_score(fla_targets, fla_preds)
        precision = precision_score(fla_targets, fla_preds, zero_division=0)
        recall = recall_score(fla_targets, fla_preds,  zero_division=0)
        f1 = f1_score(fla_targets, fla_preds,  zero_division=0)
        conf_matrix = confusion_matrix(fla_targets, fla_preds)


        micro_precision = precision_score(all_targets, all_preds, average='micro', zero_division=0)
        micro_recall = recall_score(all_targets, all_preds, average='micro', zero_division=0)
        micro_f1 = f1_score(all_targets, all_preds, average='micro', zero_division=0)

        macro_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        macro_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

        # 打印验证集指标
        print("\nValidation Metrics:")
        print(f"Validation Loss: {val_loss :.4f}")
        print("\nMicro Average:")
        print(f"Precision: {micro_precision:.4f}, Recall: {micro_recall:.4f}, F1: {micro_f1:.4f}")
        print("\nMacro Average:")
        print(f"Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}")
        print("\nBin:")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(conf_matrix)
        _,_,_,score = ODIR_Metrics(all_targets,all_preds)
        print(f"score:{score}")



        print(f"Validation Loss: {val_loss:.8f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")

        for i in range(8):
            print(f"class{i+1}")
            class_pred = []
            class_target=[]
            for j in range(len(all_preds)):
                if (j + 1) % 8 == i + 1:
                    class_pred.append(fla_preds[j])
                    class_target.append(fla_targets[j])
                if (j + 1 ) % 8 == 0:
                    class_pred.append(fla_preds[j])
                    class_target.append(fla_targets[j])

            precision, recall, f1, _ = precision_recall_fscore_support(class_target, class_pred, average='binary',zero_division=0)
            print(f"Precision (binary): {precision:.4f}")
            print(f"Recall (binary): {recall:.4f}")
        


        return val_loss,accuracy

def ODIR_Metrics(gt_data, pr_data):
    th = 0.5
    gt = gt_data.flatten()
    pr = pr_data.flatten()
    kappa = metrics.cohen_kappa_score(gt, pr>th)
    f1 = metrics.f1_score(gt, pr>th, average='micro')
    auc = metrics.roc_auc_score(gt, pr)
    final_score = (kappa+f1+auc)/3.0
    return kappa, f1, auc, final_score


# K-fold交叉验证


# 主程序
if __name__ == "__main__":
    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    dataset = EyeDataset('your_dataset_path')

    # 设置参数
    num_epochs = 30
    batch_size = 16
    lr = 0.0001
    K = 5  # K折交叉验证


