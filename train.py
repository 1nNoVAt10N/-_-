from model import DualResNet
from data_utils import EyeDataset
from data_preprocessing import PreprocessAndCache
from torch.utils.data import DataLoader
import torch
import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,confusion_matrix
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler
import metrics
from safetensors.torch import load_file


def compute_class_weights(labels, num_classes):
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]
    return class_weights

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
        # 对 logits 应用 softmax 得到每个类别的预测概率
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')  # 计算交叉熵损失
        pt = torch.exp(-BCE_loss)  # 计算目标类别的预测概率
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss  # Focal Loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class train_val_test():
    def __init__(self,epoch,lr,batch_size,num_workers,device,model,opitimizer,criterion):
        self.epoch = epoch
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = opitimizer
        self.criterion = criterion
        #lr=lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.patience = 200  # 连续多少个 epoch 验证损失没有改善后停止训练
        self.min_delta = 0.001  # 损失改善的最小阈值
        self.best_val_loss = np.inf  # 初始化为一个较大的值
        self.patience_counter = 0  # 用于计数多少个 epoch 验证损失没有改善
        self.best_model_state = None  # 用于保存最佳模型的参数

    def train(self, train_data, val_data):
        # 在 DataLoader 中使用 WeightedRandomSampler
        train_dataloader = DataLoader(train_data, batch_size=8,shuffle=False, num_workers=4)
        val_dataloader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=4)

        self.model.train()

        for epoch in tqdm.tqdm(range(self.epoch)):
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_dataloader):
                if (data[0][0][0][0] > 1):
                    print(1)
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                
                # 使用FocalLoss计算损失
                loss = self.criterion(output, target)  # target 已经是 one-hot 编码
                #print(target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.epoch}], Step [{batch_idx+1}/{len(train_dataloader)}]")
            print(train_loss / len(train_dataloader))
            val_loss = self.validate(val_dataloader)
            print(f"Validation Loss after epoch {epoch+1}: {val_loss:.8f}")

            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    self.model.load_state_dict(self.best_model_state)
                    break

    def validate(self, val_dataloader):
        self.model.eval()

        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in val_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                #print(output)
                
                loss = self.criterion(output, target)
                val_loss += loss.item()
                preds = (output > 0.5).float()
                print("target:",target)
                print("output:",output)
                

                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
        # 计算平均损失
        val_loss /= len(val_dataloader)

        # 转换为 numpy 数组
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # 计算其他指标
        
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds,  zero_division=0)
        recall = recall_score(all_targets, all_preds,  zero_division=0)
        f1 = f1_score(all_targets, all_preds,  zero_division=0)
        conf_matrix = confusion_matrix(all_targets, all_preds)

        print(f"Validation Loss: {val_loss:.8f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")

        return val_loss

                
        
    def test(self, test_data):
        test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.model.eval()
        all_targets = []
        all_preds = []
        test_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                output = (output > 0.5).float()
                all_preds.extend(output.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())

        test_loss /= len(test_dataloader)
        print(f"Test Loss: {test_loss:.4f}")

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # 计算其他指标
        
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(all_targets, all_preds)

        print(f"Validation Loss: {test_loss:.8f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")

        metrics = {
            "loss": test_loss,
            "accuracy": accuracy if all_targets.ndim == 1 else None,
            "precision": precision if all_targets.ndim > 1 else None,
            "recall": recall if all_targets.ndim > 1 else None,
            "f1": f1 if all_targets.ndim > 1 else None,

        }
        return metrics
    
