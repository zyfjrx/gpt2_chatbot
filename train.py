import torch
import os
from datetime import datetime
import transformers
from transformers import GPT2LMHeadModel, GPT2Config, BertTokenizerFast, AdamW
from parameter_config import *
from data_preprocess.dataloader import *
from functions_tools import *

def train_epoch(model,train_dataloader,optimizer, scheduler,epoch, args):
    '''
    :param model: GPT2模型
    :param train_dataloader: 训练数据集
    :param optimizer: 优化器：更新参数
    :param scheduler: 学习率预热
    :param epoch: 当前的轮次
    :param args: 模型配置文件的参数对象
    :return:
    '''
    model.train()
    device = args.device
    ignore_index = args.ignore_index
    epoch_start_time = datetime.now()
    total_loss = 0
    # epoch_correct_num:每个epoch中,output预测正确的word的数量
    # epoch_total_num: 每个epoch中,output预测的word的总数量
    epoch_correct_num, epoch_total_num = 0, 0
    for batch_idx, (input_ids, labels) in enumerate(train_dataloader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        outputs = model(input_ids,labels=labels)
        logits = outputs.logits
        loss = outputs.loss
        batch_correct_num, batch_total_num = calculate_acc(logits, labels,ignore_index=ignore_index)
        # 统计该epoch的预测token正确数与总数
        epoch_correct_num += batch_correct_num
        epoch_total_num += batch_total_num
        # 计算该batch的accuracy
        batch_acc = batch_correct_num / batch_total_num
        total_loss += loss.item()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        if (batch_idx + 1) % args.loss_step == 0:
            print(
                "batch {} of epoch {}, loss {}, batch_acc {}, lr {}".format(
                    batch_idx + 1, epoch + 1, loss.item() * args.gradient_accumulation_steps, batch_acc,
                    scheduler.get_lr()))

        del input_ids, outputs
    # 记录当前epoch的平均loss与accuracy
    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_mean_acc = epoch_correct_num / epoch_total_num
    print("epoch {}, loss {}, predict_acc: {}".format(epoch + 1, epoch_mean_loss, epoch_mean_acc))

    # save model
    if epoch % 10 == 0 or epoch == args.epochs:
        print("saving model for epoch {}".format(epoch + 1))
        model_path = os.path.join(args.save_model_path, 'bj_epoch{}'.format(epoch + 1))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model.save_pretrained(model_path)
        print('epoch {} finished'.format(epoch + 1))
        epoch_finish_time = datetime.now()
        print('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))
    return epoch_mean_loss

def validate_epoch(model, valid_dataloader, epoch, args):
    print("start validation")
    model.eval()
    device = args.device
    epoch_start_time = datetime.now()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (input_ids, labels) in enumerate(valid_dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model(input_ids,labels=labels)
            loss = outputs.loss
            loss = loss.mean()
            total_loss += loss.item()
            del input_ids, outputs
        epoch_mean_loss = total_loss / len(valid_dataloader)
        print("validate epoch {}: loss {}".format(epoch+1, epoch_mean_loss))
        epoch_finish_time = datetime.now()
        print('time for validating one epoch: {}'.format(epoch_finish_time - epoch_start_time))
        return epoch_mean_loss


def train(model,train_dataloader,valid_dataloader,args):
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    optimizer = AdamW(model.parameters(),lr=args.lr,eps=args.eps)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,num_warmup_steps=args.warmup_steps,num_training_steps=t_total
    )
    print("start training")
    # 用于记录每个epoch训练和验证的loss
    train_losses,valid_losses = [],[]
    # 记录验证集的最小loss
    best_val_loss = 10000
    for epoch in range(args.epochs):
        # ========== train ========== #
        train_loss = train_epoch(model,train_dataloader,optimizer,scheduler,epoch,args)
        train_losses.append(train_loss)
        # ========== validate ========== #
        valid_loss = validate_epoch(model,valid_dataloader,epoch,args)
        valid_losses.append(valid_loss)
        # 保存当前困惑度最低的模型，困惑度低，模型的生成效果不一定会越好
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            print('saving current best model for epoch {}'.format(epoch + 1))
            model_path = os.path.join(args.save_model_path, 'min_ppl_model_bj'.format(epoch + 1))
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            model.save_pretrained(model_path)




def main():
    params = ParameterConfig()
    tokenizer = BertTokenizerFast(params.vocab_path)
    if not os.path.exists(params.save_model_path):
        os.mkdir(params.save_model_path)
    if params.pretrained_model:
        model = GPT2LMHeadModel.from_pretrained(params.pretrained_model)
    else:
        model_config = GPT2Config.from_json_file(params.config_json)
        model = GPT2LMHeadModel(config=model_config)
    model.to(params.device)
    assert model.config.vocab_size == tokenizer.vocab_size

    # 计算模型参数数量
    num_parameters = 0
    for param in model.parameters():
        num_parameters += param.numel()
    print(f'模型参数总量：{num_parameters}')

    # 加载模型训练集和验证集
    train_dataloader,validate_dataloader = get_dataloader(params.train_path,params.valid_path)
    print(f'train_dataloader-->{len(train_dataloader)}')
    print(f'validate_dataloader-->{len(validate_dataloader)}')
    train(model, train_dataloader, validate_dataloader, params)


if __name__ == '__main__':
    main()
