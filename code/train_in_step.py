from pytorch_transformers import BertTokenizer, AdamW, BertModel, BertForSequenceClassification, XLNetModel, XLNetTokenizer, XLNetForSequenceClassification
from pytorch_transformers.optimization import WarmupLinearSchedule, WarmupConstantSchedule
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import re
import numpy as np
import torch
from torch import nn, optim
from copy import deepcopy
import click
import logging
from tqdm import tqdm
import os
import re
import json
import time
import random


PAD = 0
PAD_t = '[PAD]'
CLS_t = '[CLS]'
SEP_t = '[SEP]'


def padding_token(x, size):
    if len(x) >= size:
        return x[:size]
    else:
        return x + [PAD_t for i in range(size - len(x))]


def zh_sentence(paragraph):
    for sent in re.findall(u'[^!?。\.\!\?;:：]+[!?。\.\!\?;:：]?', paragraph, flags=re.U):
        yield sent


def preprocess(text_file, label_file, tokenizer, max_length, input_pattern, clean_method,
               max_title_rate, content_head_rate, logger):
    sample_base = random.randint(2000, 4000)
    text_df = pd.read_csv(text_file)
    if label_file:
        lab_df = pd.read_csv(label_file)  # 0: 764, 2: 2932 1: 3659
        text_df = text_df.merge(lab_df, on='id', how='left')
        text_df['label'] = text_df['label'].fillna(-1)
        text_df = text_df[text_df['label'] != -1].reset_index(drop=True)
        text_df['label'] = text_df['label'].astype(int)
    text_df['content'] = text_df['content'].fillna('无')
    text_df['title'] = text_df['title'].fillna('无')
    if clean_method == 'space':
        space_reg = re.compile(r'\s+')
        text_df['content'] = text_df['content'].apply(lambda x: space_reg.sub('', x.replace('\\n', '')))
        text_df['title'] = text_df['title'].apply(lambda x: space_reg.sub('', x.replace('\\n', '')))
    elif clean_method == 'space+notzh':
        han = re.compile(r'[^\u4e00-\u9fff\,，。！？”“]{10,}')
        space_reg = re.compile(r'\s+')
        text_df['content'] = text_df['content'].apply(lambda x: space_reg.sub('', han.sub('', x.replace('\\n', ''))))
        text_df['title'] = text_df['title'].apply(lambda x: space_reg.sub('', han.sub('', x.replace('\\n', ''))))
    output = []

    max_t_n = round(max_length * max_title_rate)
    e_counts = [0, 0]

    for idx in tqdm(range(text_df.shape[0]), 'preprocess'):
        content_str = text_df.loc[idx]['content']
        title_str = text_df.loc[idx]['title']
        id_ = text_df.loc[idx]['id']
        content_token = tokenizer.tokenize(content_str)
        title_token = tokenizer.tokenize(title_str)
        c_t_n = len(content_token)
        t_t_n = len(title_token)
        if c_t_n > 1 and t_t_n > c_t_n and t_t_n > 0.5 * max_length:
            logger.error(f'{title_str}: {content_str}')
            continue
        input_token_type = np.zeros(max_length, dtype=int)
        if input_pattern == 'head-only':
            if c_t_n + t_t_n + 3 > max_length:
                res_n = max_length - t_t_n - 3
                input_token = [CLS_t] + title_token + [SEP_t] + content_token[:res_n] + [SEP_t]
            else:
                res_n = max_length - t_t_n - c_t_n - 3
                input_token = [CLS_t] + title_token + [SEP_t] + content_token + [SEP_t] + \
                    [PAD_t for i in range(res_n)]
            input_token_type[t_t_n+2:] = 1
        elif input_pattern == 'head+tail+dynamic':
            if len(content_token) + len(title_token) + 4 > max_length:
                res_head_n = round((max_length - t_t_n - 4) * content_head_rate)
                res_tail_n = max_length - 4 - t_t_n - res_head_n
                input_token = [CLS_t] + title_token + [SEP_t] + content_token[:res_head_n] + \
                    [SEP_t] + content_token[-res_tail_n:] + [SEP_t]
            else:
                input_token = [CLS_t] + title_token + [SEP_t] + content_token + [SEP_t] + \
                    [PAD_t for i in range(max_length-3-c_t_n-t_t_n)]
            input_token_type[t_t_n+2:] = 1
        elif input_pattern == 'head+tail+fixed':
            content_head_size = round((max_length-max_t_n) * content_head_rate)
            content_tail_size = max_length - 4 - max_t_n - content_head_size
            input_token = [CLS_t] + padding_token(title_token, max_t_n) + [SEP_t] + \
                padding_token(content_token, content_head_size) + [SEP_t] + \
                padding_token(content_token[::-1], content_tail_size)[::-1] + [SEP_t]
            input_token_type[max_t_n+2:] = 1
        elif input_pattern == 'mid+fixed':
            res_n = (max_length - max_t_n - 2)
            if t_t_n > res_n:
                start_idx = round((1 - content_head_rate) * (c_t_n - res_n))
                end_idx = res_n + start_idx
                input_token = [CLS_t] + padding_token(title_token, max_t_n) + [SEP_t] + \
                    content_token[start_idx:end_idx]
            else:
                input_token = [CLS_t] + padding_token(title_token, max_t_n) + [SEP_t] + \
                    padding_token(content_token, res_n)
            input_token_type[max_t_n+2:] = 1
        elif input_pattern == 'mid+dynamic':
            res_n = max_length - t_t_n - 2
            if c_t_n > res_n:
                start_idx = round((1 - content_head_rate) * (c_t_n - res_n))
                end_idx = res_n + start_idx
                input_token = [CLS_t] + title_token + [SEP_t] + \
                    content_token[start_idx:end_idx]
            else:
                input_token = [CLS_t] + title_token + [SEP_t] + \
                    padding_token(content_token, res_n)
            input_token_type[t_t_n+2:] = 1
        elif input_pattern == 'sentence':
            if c_t_n + t_t_n + 3 > max_length:
                content_token = []
                for i, sent in enumerate(zh_sentence(content_str)):
                    if i < 1:
                        continue
                    if max_length - t_t_n - 3 < len(content_token):
                        break
                    content_token += tokenizer.tokenize(sent)
            res_n = max_length - t_t_n - 3
            input_token = [CLS_t] + title_token + [SEP_t] + \
                padding_token(content_token, res_n) + [SEP_t]
            input_token_type[t_t_n+2:] = 1
        else:
            raise ValueError('unkown input pattern')
        if (idx+1) % sample_base == 0:
            logger.info('Token Sample: ' + ' '.join(v+f'_{input_token_type[i]}'for i, v in enumerate(input_token)))
        output.append({
            'input': tokenizer.convert_tokens_to_ids(input_token),
            'input_token_type': input_token_type,
            'target': text_df.loc[idx]['label'] if label_file else -1,
            'id': id_,
        })
    return output


"""data loader"""


class DocData(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def collect_func(x):
    inputs = []
    token_type = []
    targets = []
    for record in x:
        inputs.append(torch.LongTensor(record['input']))
        token_type.append(torch.LongTensor(record['input_token_type']))
        targets.append(record['target'])

    return torch.stack(inputs), torch.stack(token_type), torch.LongTensor(targets)


def collect_test_func(x):
    inputs = []
    token_type = []
    for record in x:
        inputs.append(torch.LongTensor(record['input']))
        token_type.append(torch.LongTensor(record['input_token_type']))
    return torch.stack(inputs), torch.stack(token_type)


def get_data_iter(data, batch_size, collect_func, shuffle=True):
    data = DocData(data)
    data_iter = DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                           collate_fn=collect_func)
    return data_iter


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_train_and_val(data, batch_size, tokenizer, seed):
    train_d, val_d = train_test_split(data, test_size=0.2, random_state=seed)
    train_iter = get_data_iter(train_d, batch_size, collect_func)
    val_iter = get_data_iter(val_d, batch_size*2, collect_func, shuffle=False)
    return train_iter, val_iter


def epoch_acc(model, d_iter, usegpu, eval_step):
    model.eval()
    pred = []
    gt = []
    with torch.no_grad():
        for input, token_type, target in tqdm(d_iter, f'step {eval_step} eval'):
            if usegpu:
                input = input.cuda()
                target = target.cuda()
                token_type = token_type.cuda()
            logits = model.forward(input_ids=input, attention_mask=(input != PAD).long(), token_type_ids=token_type)[0]
            pred.append(logits.softmax(-1).argmax(-1).detach().cpu().data.numpy())
            gt.append(target.detach().cpu().data.numpy())
    gt = np.concatenate(gt)
    pred = np.concatenate(pred)
    return f1_score(gt, pred, labels=[0, 1, 2], average='macro')


"""control"""


def training(model, optimizer, lr_scheduler, train_iter, val_iter, total_step, eval_every, early_stop,
             tokenizer, usegpu, logger):
    best_score = 0
    curr_step = 0
    cum_loss = 0
    cum_step = 0
    decrease_num = 0
    best_model = deepcopy(model.state_dict())
    t = tqdm(total=total_step, desc='training')
    while curr_step < total_step + 1:
        for input, token_type, target in train_iter:
            curr_step += 1
            if curr_step > total_step:
                break
            model.train()
            model.zero_grad()
            if usegpu:
                input = input.cuda()
                target = target.cuda()
                token_type = token_type.cuda()
            loss = model.forward(input_ids=input, attention_mask=(input != PAD).long(),
                                 token_type_ids=token_type, labels=target)[0]
            cum_step += 1
            cum_loss += loss.detach().cpu().data.numpy()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if curr_step % eval_every == 0:
                val_score = epoch_acc(model=model, d_iter=val_iter, usegpu=usegpu, eval_step=curr_step)
                train_loss = cum_loss / cum_step
                logger.info(f'step {curr_step} train loss: {train_loss:.4f}')
                logger.info(f'step {curr_step} score: {val_score:.4f}')
                cum_loss = 0
                cum_step = 0
                if val_score > best_score:
                    best_score = val_score
                    best_model = deepcopy(model.state_dict())
                    decrease_num = 0
                else:
                    decrease_num += 1
                    if decrease_num > early_stop:
                        curr_step = total_step + 2
            t.update(1)
    logger.info(f'best score is {best_score:.4f}')
    return best_model


def predict(model, d_iter, usegpu):
    model.eval()
    pred = []
    with torch.no_grad():
        for input, token_type in tqdm(d_iter, f'predict'):
            if usegpu:
                input = input.cuda()
                token_type = token_type.cuda()
            logits = model.forward(input_ids=input, attention_mask=(input != PAD).long(), token_type_ids=token_type)[0]
            pred.append(logits.softmax(-1).detach().cpu().data.numpy())
    return np.concatenate(pred)


@click.command()
@click.option('--log-in-file', is_flag=True, default=False, help='logger switcher')
@click.option('--lm-type', type=click.Choice(['bert', 'xlnet']), default='bert')
@click.option('--lm-path', help='pretrained bert path.')
@click.option('--data-path', default='../data/')
@click.option('--usegpu', is_flag=True, default=False, help='use gpu')
@click.option('--total-step', type=int, default=10000, help='step number')
@click.option('--eval-every', type=int, default=1000)
@click.option('--early-stop', type=int, default=4)
@click.option('--lr', type=float, default=5e-6)
@click.option('--weight-decay', type=float, default=1e-2)
@click.option('--lr-decay-in-layers', type=float, default=1.0)
@click.option('--wd-decay-in-layers', type=float, default=1.0)
@click.option('--max-length', type=int, default=300)
@click.option('--max-title-rate', type=float, default=0.125)
@click.option('--content-head-rate', type=float, default=0.6)
@click.option('--batch-size', type=int, default=2)
@click.option('--lr-scheduler-type', type=click.Choice(['linear', 'constant']), default='linear')
@click.option('--input-pattern', type=click.Choice(['head-only', 'head+tail+dynamic', 'head+tail+fixed',
                                                    'mid+fixed', 'mid+dynamic', 'sentence']), default='head-only')
@click.option('--clean-method', type=click.Choice(['none', 'space', 'space+notzh']), default='none')
@click.option('--warmup-rate', type=float, default=0.1)
@click.option('--classifier-dropout', type=float, default=0.1)
@click.option('--classifier-active', type=str, default='tanh')
@click.option('--cache-final-model', is_flag=True, default=False)
@click.option('--seed', type=int, default=23)
def main(log_in_file, lm_path, lm_type, data_path, usegpu, total_step, eval_every, early_stop, lr, weight_decay,
         lr_decay_in_layers, wd_decay_in_layers, max_length, max_title_rate, content_head_rate, batch_size, lr_scheduler_type,
         input_pattern, clean_method, warmup_rate, classifier_dropout, classifier_active, cache_final_model, seed):
    arg_name_value_pairs = deepcopy(locals())
    prefix = time.strftime('%Y%m%d_%H%M')
    logger = logging.getLogger('default')
    formatter = logging.Formatter("%(asctime)s %(message)s")
    if log_in_file:
        handler1 = logging.FileHandler(prefix + '.log')
        handler1.setFormatter(formatter)
        handler1.setLevel(logging.DEBUG)
        logger.addHandler(handler1)
    handler2 = logging.StreamHandler()
    handler2.setFormatter(formatter)
    handler2.setLevel(logging.DEBUG)
    logger.addHandler(handler2)
    logger.setLevel(logging.DEBUG)
    for arg_name, arg_value in arg_name_value_pairs.items():
        logger.info(f'{arg_name}: {arg_value}')
    global tokenizer
    if lm_type == 'bert':
        tokenizer = BertTokenizer(os.path.join(lm_path, 'vocab.txt'))
    else:
        tokenizer = XLNetTokenizer(os.path.join(lm_path, 'spiece.model'))
        global PAD, PAD_t, CLS_t, SEP_t
        PAD_t = '<pad>'
        CLS_t = '<cls>'
        SEP_t = '<sep>'
        PAD = tokenizer.convert_tokens_to_ids([PAD_t])[0]
    logger.info(f'padding token is {PAD}')
    processed_train = preprocess(os.path.join(data_path, 'Train_DataSet.csv'),
                                 os.path.join(data_path, 'Train_DataSet_Label.csv'),
                                 tokenizer, max_length, input_pattern, clean_method,
                                 max_title_rate, content_head_rate, logger)
    processed_test = preprocess(os.path.join(data_path, 'Test_DataSet.csv'), False,
                                tokenizer, max_length, input_pattern, clean_method,
                                max_title_rate, content_head_rate, logger)
    logger.info('seed everything and create model')
    seed_everything(seed)
    train_iter, val_iter = create_train_and_val(
        processed_train, batch_size=batch_size, tokenizer=tokenizer, seed=seed)
    no_decay = ['.bias', 'layer_norm.bias', 'layer_norm.weight']
    if lm_type == 'xlnet':
        model = XLNetForSequenceClassification.from_pretrained(lm_path, num_labels=3, summary_last_dropout=classifier_dropout)
        if classifier_active == 'relu':
            model.sequence_summary.activation = nn.ReLU()
        if usegpu:
            model = model.cuda()
        model_layer_names = ['transformer.mask_emb', 'transformer.word_embedding.weight']
        model_layer_names += [f'transformer.layer.{i}.' for i in range(model.config.n_layer)]
        model_layer_names += ['sequence_summary.summary', 'logits_proj']
    else:
        model = BertForSequenceClassification.from_pretrained(
            lm_path, num_labels=3, hidden_dropout_prob=classifier_dropout)
        if classifier_active == 'relu':
            model.bert.pooler.activation = nn.ReLU()
        if usegpu:
            model = model.cuda()
        model_layer_names = ['bert.embeddings']
        model_layer_names += ['bert.encoder.layer.{}.'.format(i) for i in range(model.config.num_hidden_layers)]
        model_layer_names += ['bert.pooler', 'classifier']
    optimizer = optimizer = AdamW(
       [{'params': [p for n, p in model.named_parameters()
                    if layer_name in n and not any(nd in n for nd in no_decay)],
         'lr': lr*(lr_decay_in_layers**i), 'weight_decay': weight_decay*(wd_decay_in_layers**i)}
        for i, layer_name in enumerate(model_layer_names[::-1])] +
       [{'params': [p for n, p in model.named_parameters()
                    if layer_name in n and any(nd in n for nd in no_decay)],
         'lr': lr*(lr_decay_in_layers**i), 'weight_decay': .0}
        for i, layer_name in enumerate(model_layer_names[::-1])])
    if lr_scheduler_type == 'linear':
        lr_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_rate, t_total=total_step)
    elif lr_scheduler_type == 'constant':
        lr_scheduler = WarmupConstantSchedule(optimizer, warmup_steps=warmup_rate)
    else:
        raise ValueError
    best_model = training(
             model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
             train_iter=train_iter, val_iter=val_iter, total_step=total_step,
             tokenizer=tokenizer, usegpu=usegpu, eval_every=eval_every,
             logger=logger, early_stop=early_stop)
    model.load_state_dict(best_model)
    test_iter = get_data_iter(processed_test, batch_size*2, collect_test_func, shuffle=False)
    pred = predict(model, test_iter, usegpu)
    submit = pd.DataFrame()
    submit['id'] = [i['id'] for i in processed_test]
    submit['0'] = pred[:, 0]
    submit['1'] = pred[:, 1]
    submit['2'] = pred[:, 2]
    submit.to_csv('submit.csv', index=False)


if __name__ == '__main__':
    main()
