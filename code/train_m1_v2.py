from pytorch_transformers import BertTokenizer, AdamW, BertModel
from pytorch_transformers.optimization import WarmupLinearSchedule
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import re
import numpy as np
import torch
from torch import nn, optim
import click
import logging
from tqdm import tqdm
import os
import re
import json
import time
import random

"""
note:
val score: ~0.77
"""


"""global setting"""
MAX_LEN = 300
PAD = 0
CLS = 101
SEP = 102
T_N = 40
C_H = 260


"""preprocess"""


def pad_token(x, size):
    if len(x) >= size:
        return x[:size]
    else:
        return x + [PAD for i in range(size - len(x))]


def preprocess(text_file, label_file, tokenizer):
    text_df = pd.read_csv(text_file)
    lab_df = pd.read_csv(label_file)  # 0: 764, 2: 2932 1: 3659
    space_reg = re.compile(r'\s+')
    text_df['content'] = text_df['content'].fillna('')
    text_df['content'] = text_df['content'].apply(lambda x: space_reg.sub('', x))
    text_df['title'] = text_df['title'].fillna('')
    text_df['title'] = text_df['title'].apply(lambda x: space_reg.sub('', x))
    id2lab = dict((lab_df.loc[i]['id'], int(lab_df.loc[i]['label'])) for i in range(lab_df.shape[0]))
    output = []
    for i in tqdm(range(text_df.shape[0]), 'preprocess'):
        e_c = text_df.loc[i]['content']
        e_t = text_df.loc[i]['title']
        e_id = text_df.loc[i]['id']
        if e_id not in id2lab or (len(e_t) + len(e_c)) == 0:
            continue
        e_c_t = tokenizer.encode(e_c)
        e_t_t = tokenizer.encode(e_t)
        p_t = pad_token(e_t_t, T_N)
        p_ch = pad_token(e_c_t, C_H)
        output.append({
            'input': [CLS] + [SEP] + p_t + [SEP] + p_ch,
            'target': id2lab[e_id]
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
    targets = []
    for record in x:
        inputs.append(torch.LongTensor(record['input']))
        targets.append(record['target'])

    return torch.stack(inputs), torch.LongTensor(targets)


def get_data_iter(data, batch_size, collect_func, shuffle=True):
    data = DocData(data)
    data_iter = DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                           collate_fn=collect_func)
    return data_iter


"""epoch train and valicate"""


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


"""model"""


class PoolerClassifier(nn.Module):

    def __init__(self, config, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.batchnorm = nn.BatchNorm1d(config.hidden_size)
        self.active = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 3)

    def forward(self, hidden_state):
        pooled_output = self.pooler(hidden_state)
        pooled_output = self.dropout(self.active(pooled_output))
        logits = self.classifier(pooled_output)
        return logits


class BertClassifier(nn.Module):

    def __init__(self, bert, dropout=0.1):
        super().__init__()
        self.bert = bert
        self.classifier = PoolerClassifier(bert.config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        output = self.bert(input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           position_ids=position_ids,
                           head_mask=head_mask)
        hidden_state = output[0][:, 0]
        logits = self.classifier(hidden_state)
        if labels is not None:
            loss = F.cross_entropy(input=logits, target=labels)
            return loss
        return logits


"""training detail"""


def tensor2ls(x):
    return x.cpu().data.tolist()


def epoch_train(model, optimizer, lr_scheduler, d_iter, usegpu, epoch):
    model.train()
    cum_loss = 0
    cum_step = 0
    for input, target in tqdm(d_iter, f'epoch {epoch:02} training'):
        model.zero_grad()
        token_type = torch.tensor([0] + [1 for i in range(T_N+1)] + [2 for i in range(C_H+1)], dtype=input.dtype)
        if usegpu:
            input = input.cuda()
            target = target.cuda()
            token_type = token_type.cuda()
        token_type = token_type[None, :].expand(input.shape[0], -1)
        loss = model.forward(input_ids=input, attention_mask=input != PAD, token_type_ids=None, labels=target)
        cum_step += 1
        cum_loss += loss.detach().cpu().data.numpy()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
    return cum_loss/cum_step


def epoch_acc(model, d_iter, usegpu, epoch):
    model.eval()
    pred = np.array([])
    gt = np.array([])
    with torch.no_grad():
        for input, target in tqdm(d_iter, f'epoch {epoch:02} training'):
            token_type = torch.tensor([0] + [1 for i in range(T_N+1)] + [2 for i in range(C_H+1)], dtype=input.dtype)
            if usegpu:
                input = input.cuda()
                target = target.cuda()
                token_type = token_type.cuda()
            token_type = token_type[None, :].expand(input.shape[0], -1)
            logits = model.forward(input_ids=input, attention_mask=input != PAD, token_type_ids=None)
            pred = np.append(pred, logits.softmax(-1).argmax(-1).detach().cpu().data.numpy())
            gt = np.append(gt, target.detach().cpu().data.numpy())
    return f1_score(gt, pred, average='macro')


"""control"""


def training(model, optimizer, lr_scheduler, train_iter, val_iter, epoch,
             tokenizer, usegpu, train_dir, logger):
    best_score = 0
    for e in range(epoch):
        epoch_loss = epoch_train(model=model,
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler,
                                 d_iter=train_iter,
                                 epoch=e,
                                 usegpu=usegpu)
        logger.info(f'epcoh {e:02} train loss: {epoch_loss:.4f}')
        epoch_score = epoch_acc(model=model,
                                d_iter=val_iter,
                                usegpu=usegpu,
                                epoch=e)
        logger.info(f'epcoh {e:02} score: {epoch_score:.4f}')
        if epoch_score > best_score:
            best_score = epoch_score
            torch.save({'model': model.state_dict(),
                        'optimizer': model.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict()},
                       os.path.join(train_dir, 'best.cp'))
    torch.save({'model': model.state_dict(),
                'optimizer': model.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()},
               os.path.join(train_dir, 'last.cp'))
    logger.info(f'best score is {best_score:.4f}')


@click.command()
@click.option('--log', is_flag=True, default=False, help='logger switcher')
@click.option('--mpath', help='pretrained language model path.')
@click.option('--usegpu', is_flag=True, default=False, help='use gpu')
@click.option('--epoch', type=int, help='epoch number')
def main(log, mpath, usegpu, epoch):
    work_dir = time.strftime('Baseline_%Y%m%d_%H')
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    logger = logging.getLogger('default')
    handler1 = logging.FileHandler(os.path.join(work_dir, 'training.log'))
    handler2 = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    handler1.setLevel(logging.DEBUG)
    handler2.setLevel(logging.DEBUG)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(logging.DEBUG)
    vocab_path = os.path.join(mpath, 'vocab.txt')
    global tokenizer
    tokenizer = BertTokenizer(vocab_path)
    processed_file = 'processed.json'
    if not os.path.exists(processed_file):
        logger.info('preprocess the raw data.')
        processed = preprocess('../data/Train_DataSet.csv', '../data/Train_DataSet_Label.csv', tokenizer)
        with open(processed_file, 'w') as f:
            json.dump(processed, f)
    else:
        logger.info('using precompute processed file.')
        with open(processed_file) as f:
            processed = json.load(f)
    SEED = 23
    logger.info('seed seed_everything and create model')
    seed_everything(SEED)
    logger.info('creating training iter')
    train_iter, val_iter = create_train_and_val(
        processed, batch_size=16, tokenizer=tokenizer, seed=SEED)
    bert = BertModel.from_pretrained(mpath)
    # bert.embeddings.token_type_embeddings = nn.Embedding(3, bert.config.hidden_size)
    model = BertClassifier(bert, bert.config)
    if usegpu:
        model = model.cuda()
    t_step = epoch*len(train_iter)
    logger.info(f'total step: {t_step}')

    lr = 2e-5
    lr_decay_rate = 0.95

    bert_layer_names = ['bert.embeddings']
    bert_layer_names += ['bert.encoder.layer.{}.'.format(i) for i in range(12)]
    bert_layer_names += ['classifier']
    no_decay = ['.bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer = AdamW(
        [{'params': [p for n, p in model.named_parameters()
                     if layer_name in n and not any(nd in n for nd in no_decay)],
          'lr': lr*(lr_decay_rate**i), 'weight_decay': 1e-2}
         for i, layer_name in enumerate(bert_layer_names[::-1])] +
        [{'params': [p for n, p in model.named_parameters()
                     if layer_name in n and any(nd in n for nd in no_decay)],
          'lr': lr*(lr_decay_rate**i), 'weight_decay': 0}
         for i, layer_name in enumerate(bert_layer_names[::-1])])

    lr_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0.1*t_step,
                                        t_total=t_step)
    training(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
             train_iter=train_iter, val_iter=val_iter, epoch=epoch,
             tokenizer=tokenizer, usegpu=usegpu, train_dir=work_dir,
             logger=logger)


if __name__ == '__main__':
    main()
