import os
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

# 自定义的数据集类
class ValveDataset(Dataset):
    def __init__(self, attributes_file, labels_file, transform=None):
        self.attributes = pd.read_csv(attributes_file)
        # 跳过第一行，开始加载实际数据
        self.labels = pd.read_csv(labels_file, header=None, skiprows=1).iloc[:, 0]  # 读取第一列数据作为标签
        self.transform = transform
        # 打印数据集的形状
        print(f"Attributes shape: {self.attributes.shape}")
        print(f"Labels shape: {self.labels.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 原来输出的 sample 形状是 [201]，调整为 [1, 201]，以适应 Conv1d 的输入
        sample = torch.tensor(self.attributes.iloc[idx].values, dtype=torch.float32).unsqueeze(0)  # [1, 201]
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.long)

        if self.transform:
            sample = self.transform(sample)

        return sample, label

# 数据集加载函数
def load_dataset(dataset='ValveDataset', batch_size=100, dataset_path=None, is_cuda=False, num_workers=8, train_ratio=0.8):
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if is_cuda else {}

    if dataset == 'ValveDataset':
        num_classes = 8  # 对应数据集中的类别数量
        attributes_file = os.path.join(dataset_path, 'attributes.csv')
        labels_file = os.path.join(dataset_path, 'label.csv')

        transform = None

        # 加载数据集
        full_dataset = ValveDataset(attributes_file, labels_file, transform=transform)

        # 计算训练集和测试集的大小
        train_size = int(train_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size

        # 分割数据集
        dataset_train, dataset_test = torch.utils.data.random_split(full_dataset, [train_size, test_size])

        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=batch_size, shuffle=False, **kwargs)

    else:
        raise Exception('No valid dataset is specified.')

    return train_loader, test_loader, num_classes
