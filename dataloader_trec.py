import os
import numpy as np
import pandas as pd
import random
import json
import torch
from pytorch_transformers import *
from torch.utils.data import Dataset, DataLoader
import pickle

class Translator:
    """Backtranslation. Here to save time, we pre-processing and save all the translated data into pickle files.
    """

    def __init__(self, path, transform_type='BackTranslation'):
        # Pre-processed German data
        #with open(path + 'de_2.pkl', 'rb') as f:
            #self.de = pickle.load(f)
        # Pre-processed Russian data
        with open(path + '/ru_trec.pkl', 'rb') as f:
            self.ru = pickle.load(f)

    def __call__(self, ori, idx):
        #out1 = self.de[idx]
        out2 = self.ru[idx]
        return  out2, ori

class imdb_dataset(Dataset):
    def __init__(self, data_path, r, mode, noise_mode,noise_file='', model='/model/zhuoer/bert',max_seq_len=512,pred_sum=[],un_aug=False):
        self.r = r  # noise ratio
        self.mode = mode
        self.noise_mode = noise_mode
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.max_seq_len = max_seq_len
        self.un_aug = un_aug
        self.trans_dist = {}

        if self.un_aug:
            self.aug = Translator(data_path)


        if self.mode == 'test':
            test_df = pd.read_csv(data_path + '/test.csv')
            self.test_texts = np.array([v for v in test_df['review']]) #(10000,)
            self.test_labels = np.array([v for v in test_df['label']])
            self.n_labels = max(self.test_labels) + 1               # 2
        else:
            train_df = pd.read_csv(data_path + '/train.csv')  #
            train_texts = np.array([v for v in train_df['review']])  #
            train_labels = np.array([v for v in train_df['label']])  #
            num_train = train_texts.shape[0]  #40000
            # ADD Noise
            if os.path.exists(noise_file):
                noise_labels = json.load(open(noise_file, "r"))
            else:  # inject noise
                noise_labels = []
                idx = list(range(num_train))  # [0,1,2....39999]
                random.shuffle(idx)  # print(idx)已打乱
                num_noise = int(self.r * num_train)
                noise_idx = idx[:num_noise]  # 有噪音的数据序号
                for i in range(num_train):
                    if i in noise_idx:
                        if self.noise_mode == 'sym':
                            noiselabel_i = random.randint(0,5)  # [0,4] 之间的任何数

                        elif self.noise_mode== 'asym':
                            noiselabel_i = np.random.choice([train_labels[i], (train_labels[i] + 1) % 6], p=[1 - self.r, self.r])
                        noise_labels.append(int(noiselabel_i))
                    else:
                        noise_labels.append(int(train_labels[i]))
                print("save noisy labels to %s ..." % noise_file)

                json.dump(noise_labels, open(noise_file, "w"))

            if self.mode == 'all':
                self.train_texts= train_texts
                self.noise_labels = noise_labels

            else:
                if self.mode == "labeled":
                    pred_idx =np.array(pred_sum).nonzero()[0]

                elif self.mode == "unlabeled":
                    pred_idx = np.where(np.array(pred_sum)==0)[0]

                self.train_texts = train_texts[pred_idx]
                self.noise_labels = np.array(noise_labels)[pred_idx]
                print("%s data has a size of %d" % (self.mode, self.noise_labels.shape[0]))

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding
        return encode_result, length

    # def augment(self, text):
    #     if text not in self.trans_dist:
    #         self.trans_dist[text] = self.de2en.translate(self.en2de.translate(
    #             text,  sampling=True, temperature=0.9),  sampling=True, temperature=0.9)
    #     return self.trans_dist[text]

    def __getitem__(self, index):
        if self.mode=='test':
            text = self.test_texts[index]
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            length = len(tokens)
            encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (self.max_seq_len - len(encode_result))
            encode_result += padding
            return (torch.tensor(encode_result), self.test_labels[index], length)
        if self.mode=='all':
            text = self.train_texts[index]
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            length = len(tokens)
            encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (self.max_seq_len - len(encode_result))
            encode_result += padding
            # to ids
            return (torch.tensor(encode_result), self.noise_labels[index], length)
        if self.mode=='labeled':
            text = self.train_texts[index]
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            length = len(tokens)
            encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (self.max_seq_len - len(encode_result))
            encode_result += padding
            return torch.tensor(encode_result), self.noise_labels[index], length
        elif self.mode=='unlabeled':
            if self.un_aug:
                # text = self.train_texts[index]
                # text_aug = self.augment(text)
                # text_result, text_length = self.get_tokenized(text)
                # text_result2, text_length2 = self.get_tokenized(text_aug)
                # return torch.tensor(text_result), torch.tensor(text_result2)
                u, ori = self.aug(self.train_texts[index], index)
                encode_result_u, length_u = self.get_tokenized(u)
                encode_result_ori, length_ori = self.get_tokenized(ori)
                return torch.tensor(encode_result_u), torch.tensor(encode_result_ori)
            else:
                text = self.train_texts[index]
                tokens = self.tokenizer.tokenize(text)
                if len(tokens) > self.max_seq_len:
                    tokens = tokens[:self.max_seq_len]
                length = len(tokens)
                encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
                padding = [0] * (self.max_seq_len - len(encode_result))
                encode_result += padding
                return torch.tensor(encode_result) ####有待商榷


    def __len__(self):
        if self.mode == 'test':
            return len(self.test_texts)
        else:
            return len(self.train_texts)


class imdb_dataloader():
    def __init__(self, data_path, r, batch_size, num_workers,  noise_mode,noise_file=''):

        self.data_path = data_path
        self.r = r
        self.noise_mode = noise_mode
        self.noise_file = noise_file

        self.batch_size = batch_size
        self.num_workers = num_workers


    def run(self, mode,pred_sum=[],un_aug=False):
        if mode == 'warmup':
            all_dataset = imdb_dataset(data_path=self.data_path,r=self.r,  mode="all", noise_mode= self.noise_mode,noise_file=self.noise_file)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return trainloader

        if mode=='test':
            test_dataset = imdb_dataset(data_path=self.data_path,  r=self.r, mode='test',noise_mode= self.noise_mode)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return test_loader

        if mode=='eval_train':
            eval_dataset = imdb_dataset(data_path=self.data_path, r=self.r, mode='all', noise_mode= self.noise_mode,noise_file=self.noise_file)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return eval_loader

        if mode == 'train':
            labeled_dataset = imdb_dataset(data_path=self.data_path,  r=self.r, mode="labeled",noise_mode= self.noise_mode,noise_file=self.noise_file,\
                                           pred_sum=pred_sum)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)

            unlabeled_dataset = imdb_dataset(data_path=self.data_path, r=self.r,mode="unlabeled",noise_mode= self.noise_mode,\
                                             noise_file=self.noise_file, pred_sum=pred_sum,un_aug=un_aug)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return labeled_trainloader, unlabeled_trainloader