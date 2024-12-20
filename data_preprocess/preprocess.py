import pickle
import os
from tqdm import tqdm
from transformers import BertTokenizerFast
from parameter_config import *

params = ParameterConfig()

def data_preprocess(train_txt_path, train_pkl_path):
    tokenizer = BertTokenizerFast(params.vocab_path)
    sep_id = tokenizer.sep_token_id  # 获取分隔符[SEP]的token ID
    cls_id = tokenizer.cls_token_id  # 获取起始符[CLS]的token ID
    print(f'tokenizer.vocab_size-->{tokenizer.vocab_size}')

    with open(train_txt_path, 'rb') as f:
        data = f.read().decode('utf-8')

    train_data = data.split('\n\n')
    # 记录所有对话长度
    dialogue_len = []
    # 记录所有对话
    dialogue_list = []
    for index,dialogue in enumerate(tqdm(train_data)):
        if "\r\n" in dialogue:
            sequences = dialogue.split("\r\n")
        else:
            sequences = dialogue.split("\n")
        input_ids = [cls_id]
        for sequence in sequences:
            input_ids += tokenizer.encode(sequence,add_special_tokens=False)
            input_ids.append(sep_id)

        dialogue_len.append(len(input_ids))
        dialogue_list.append(input_ids)
    # print(f'dialogue_len--->{dialogue_len}')  # 打印对话长度列表
    # print(f'dialogue_list--->{dialogue_list[-1]}')  # 打印

    with open(train_pkl_path, 'wb') as f:
        pickle.dump(dialogue_list, f)


if __name__ == '__main__':
    train_txt_path = '../data/medical_train.txt'
    train_pkl_path = '../data/medical_train.pkl'
    data_preprocess(train_txt_path, train_pkl_path)