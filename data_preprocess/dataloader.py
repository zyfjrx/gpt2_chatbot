import torch
import pickle
from torch.utils.data import DataLoader
from parameter_config import *
from dataset import *
import torch.nn.utils.rnn as rnn_utils  # 导入rnn_utils模块，用于处理可变长度序列的填充和排序
params = ParameterConfig()

def load_dataset(train_path,valid_path):
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(valid_path, 'rb') as f:
        valid_data = pickle.load(f)

    train_dataset = MyDataset(train_data, 300)  # 创建训练数据集对象
    val_dataset = MyDataset(valid_data, 300)  # 创建验证数据集对象
    return train_dataset, val_dataset  # 返回训练数据集和验证数据集

def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)
    return input_ids, labels

def get_dataloader(train_path, valid_path):
    train_dataset, val_dataset = load_dataset(train_path, valid_path)

    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True,drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False,drop_last=True, collate_fn=collate_fn)
    return train_loader, val_loader


if __name__ == '__main__':
    # train_dataset, val_dataset = load_dataset('../data/medical_train.pkl', '../data/medical_valid.pkl')
    # print(len(train_dataset))
    train_dataloader, validate_dataloader = get_dataloader('../data/medical_train.pkl', '../data/medical_valid.pkl')
    print(len(train_dataloader))
    # for input_ids, labels in train_dataloader:
    #     print('你好')
    #     print(f'input_ids--->{input_ids.shape}')
    #     # print(f'input_ids--->{input_ids}')
    #     print(f'labels--->{labels.shape}')
    #     print('*' * 80)
    #     break