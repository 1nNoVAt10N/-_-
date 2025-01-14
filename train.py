from model import ResNetModel
from data_utils import EyeDataset
from data_preprocessing import PreprocessAndCache
from torch.utils.data import DataLoader
import torch
import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np


class train_val_test():
    def __init__(self,epoch,lr,batch_size,num_workers,device,model,opitimizer,criterion):
        self.epoch = epoch
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = opitimizer
        self.criterion = criterion

    def train(self,train_data):
        train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
        self.model.train()
        for epoch in tqdm.tqdm(range(self.epoch)):
            for batch_idx, (data, target) in enumerate(train_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.epoch}], Step [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

    def val(self, val_data):
        val_dataloader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.model.eval()
        all_targets = []
        all_outputs = []
        val_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                val_loss += loss.item()

                # 收集输出和标签用于指标计算
                all_targets.append(target.cpu().numpy())
                all_outputs.append(output.cpu().numpy())

        val_loss /= len(val_dataloader)
        print(f"Validation Loss: {val_loss:.4f}")

        all_targets = np.concatenate(all_targets, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)

        print(all_targets)
        print(all_outputs)
        preds = np.argmax(all_outputs, axis=1)
        accuracy = accuracy_score(all_targets, preds)
        print(f"Validation Accuracy: {accuracy:.4f}")
        precision = precision_score(all_targets, preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, preds, average='weighted', zero_division=0)
        auc = roc_auc_score(all_targets, all_outputs, average='weighted', multi_class='ovr')

        print(f"Validation Precision: {precision:.4f}")
        print(f"Validation Recall: {recall:.4f}")
        print(f"Validation F1 Score: {f1:.4f}")
        print(f"Validation AUC-ROC: {auc:.4f}")
                
        
    def test(self, test_data):
        test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.model.eval()
        all_targets = []
        all_outputs = []
        test_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()

                all_targets.append(target.cpu().numpy())
                all_outputs.append(output.cpu().numpy())

        test_loss /= len(test_dataloader)
        print(f"Test Loss: {test_loss:.4f}")

        all_targets = np.concatenate(all_targets, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)
 
        preds = np.argmax(all_outputs, axis=1)
        accuracy = accuracy_score(all_targets, preds)
        print(f"Test Accuracy: {accuracy:.4f}") 
        precision = precision_score(all_targets, preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, preds, average='weighted', zero_division=0)
        auc = roc_auc_score(all_targets, all_outputs, average='weighted', multi_class='ovr')

        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        print(f"Test AUC-ROC: {auc:.4f}")

        metrics = {
            "loss": test_loss,
            "accuracy": accuracy if all_targets.ndim == 1 else None,
            "precision": precision if all_targets.ndim > 1 else None,
            "recall": recall if all_targets.ndim > 1 else None,
            "f1": f1 if all_targets.ndim > 1 else None,
            "auc": auc if all_targets.ndim > 1 else None,
        }
        return metrics
    
