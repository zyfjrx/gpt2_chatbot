import torch
import pickle
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self,input_list, max_len):
        super().__init__()
        self.input_ids = input_list
        self.max_len = max_len

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        input_ids = input_ids[:self.max_len]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids

if __name__ == '__main__':
    with open('../data/medical_valid.pkl', 'rb') as f:
        train_input_list = pickle.load(f)  # 从文件中加载输入列
    print(len(train_input_list))
    my_dataset = MyDataset(train_input_list, max_len=300)
    print(f'mydataset-->{len(my_dataset)}')
    result = my_dataset[300]
    print(result.shape)