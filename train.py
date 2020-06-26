import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bert.seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
from bert.seq2seq.utils import load_bert, load_model_params, load_recent_model


def read_corpus(dir_path, vocab_path):
    sents_src = []
    sents_tgt = []
    word2idx = load_chinese_base_vocab(vocab_path)
    tokenizer = Tokenizer(word2idx)
    files = os.listdir(dir_path) 
    for file1 in files:
        if not os.path.isdir(file1): 
            file_path = dir_path + "/" + file1
            print(file_path)
            if file_path[-3:] != "csv":
                continue
            df = pd.read_csv(file_path)
            for index, row in df.iterrows():
                if type(row[0]) is not str or type(row[3]) is not str:
                    continue
                if len(row[0]) > 8 or len(row[0]) < 2:
                    # 过滤掉题目长度过长和过短的诗句
                    continue
                if len(row[0].split(" ")) > 1:
                    # 说明题目里面存在空格，只要空格前面的数据
                    row[0] = row[0].split(" ")[0]
                encode_text = tokenizer.encode(row[3])[0]
                if word2idx["[UNK]"] in encode_text:
                    continue
                if len(row[3]) == 24 and (row[3][5] == "，" or row[3][5] == "。"):
                    # 五言绝句
                    sents_src.append(row[0] + "##" + "五言绝句")
                    sents_tgt.append(row[3])
                elif len(row[3]) == 32 and (row[3][7] == "，" or row[3][7] == "。"):
                    # 七言绝句
                    sents_src.append(row[0] + "##" + "七言绝句")
                    sents_tgt.append(row[3])
                elif len(row[3]) == 48 and (row[3][5] == "，" or row[3][5] == "。"):
                    # 五言律诗
                    sents_src.append(row[0] + "##" + "五言律诗")
                    sents_tgt.append(row[3])
                elif len(row[3]) == 64 and (row[3][7] == "，" or row[3][7] == "。"):
                    # 七言律诗
                    sents_src.append(row[0] + "##" + "七言律诗")
                    sents_tgt.append(row[3])

    print("诗句共: " + str(len(sents_src)) + "篇")
    return sents_src, sents_tgt


class BertDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """

    def __init__(self, sents_src, sents_tgt, vocab_path):
        # 一般init函数是加载所有数据
        super(BertDataset, self).__init__()
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt
        self.word2idx = load_chinese_base_vocab(vocab_path)
        self.idx2word = {k: v for v, k in self.word2idx.items()}
        self.tokenizer = Tokenizer(self.word2idx)

    def __getitem__(self, i):
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]
        token_ids, token_type_ids = self.tokenizer.encode(src, tgt)
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
        }
        return output

    def __len__(self):

        return len(self.sents_src)


def collate_fn(batch):

    def padding(indice, max_length, pad_idx=0):
        pad_indice = [item + [pad_idx] *
                      max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids_padded


class PoemTrainer:
    def __init__(self):
        # 加载数据
        data_dir = "data"
        self.vocab_path = "corpus/bert-base-chinese-vocab.txt"
        self.sents_src, self.sents_tgt = read_corpus(data_dir, self.vocab_path)
        self.model_name = "bert"
        self.model_path = "weights/bert-base-chinese-pytorch_model.bin"
        self.recent_model_path = ""
        self.model_save_path = "./bert_model.bin"
        self.batch_size = 24
        self.lr = 1e-5
        self.word2idx = load_chinese_base_vocab(self.vocab_path)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        self.bert_model = load_bert(
            self.vocab_path, model_name=self.model_name)
        load_model_params(self.bert_model, self.model_path)
        self.bert_model.to(self.device)
        self.optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(
            self.optim_parameters, lr=self.lr, weight_decay=1e-3)
        dataset = BertDataset(self.sents_src, self.sents_tgt, self.vocab_path)
        self.dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

    def train(self, epoch):
        self.bert_model.train()
        self.iteration(epoch, dataloader=self.dataloader, train=True)

    def save(self, save_path):
        torch.save(self.bert_model.state_dict(), save_path)
        print("{} saved!".format(save_path))

    def iteration(self, epoch, dataloader, train=True):
        total_loss = 0
        start_time = time.time()
        step = 0
        for token_ids, token_type_ids, target_ids in tqdm(dataloader, position=0, leave=True):
            step += 1
            if step % 1000 == 0:
                self.bert_model.eval()
                test_data = ["观棋##五言绝句", "题西林壁##七言绝句",
                             "长安早春##五言律诗", "端午##七言绝句", "端阳安康##五言绝句","静夜思##七言绝句"]
                for text in test_data:
                    print(self.bert_model.generate(
                        text, beam_size=3, device=self.device, is_poem=True))
                self.bert_model.train()

            token_ids = token_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            predictions, loss = self.bert_model(token_ids,
                                                token_type_ids,
                                                labels=target_ids,
                                                device=self.device
                                                )
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        end_time = time.time()
        spend_time = end_time - start_time
        print("epoch is " + str(epoch)+". loss is " +
              str(total_loss) + ". spend time is " + str(spend_time))
        self.save(self.model_save_path)


if __name__ == '__main__':

    trainer = PoemTrainer()
    train_epoches = 30
    for epoch in range(train_epoches):
        trainer.train(epoch)
