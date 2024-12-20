import torch
import torch.nn.functional as F




def calculate_acc(logit, labels,ignore_index=-100):
    # print(f'logit--->原始值的形状{logit.shape}')
    # print(f'labels--->原始值的形状{labels.shape}')
    logit = logit[:,:-1,:].contiguous().view(-1,logit.shape[-1])
    labels = labels[:,1:].contiguous().view(-1)
    print(f'logit--->变换后的形状{logit.shape}')
    print(f'labels--->变换后的形状{labels.shape}')
    logit = torch.argmax(logit, dim=-1)
    # print(f'logit--->变换后的形状2{logit.shape}')
    # print(logit)
    non_pad_mask = labels.ne(ignore_index)
    #n_correct = logit[non_pad_mask].eq(labels[non_pad_mask]).sum().item()
    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    print(f'n_correct-->{n_correct}')
    n_word = non_pad_mask.sum().item()
    print(f'n_word-->{n_word}')
    return n_correct , n_word



