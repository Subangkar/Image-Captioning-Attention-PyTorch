import math
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import glob
import pandas as pd
import numpy as np
import io
from torchvision import transforms
import nltk

from utils_torch import split_data
from utils_torch import padding_tensor


class Flickr8kDataset(Dataset):
    """
    imgname: just image file name
    imgpath: full path to image file
    """

    def __init__(self, dataset_base_path='data/flickr8k/',
                 vocab_set=None, dist='val',
                 startseq="<start>", endseq="<end>", unkseq="<unk>",
                 transformations=None,
                 return_raw=False,
                 device=torch.device('cpu')):
        self.token = dataset_base_path + 'Flickr8k_text/Flickr8k.token.txt'
        self.images_path = dataset_base_path + 'Flicker8k_Dataset/'

        self.dist_list = {
            'train': dataset_base_path + 'Flickr8k_text/Flickr_8k.trainImages.txt',
            'val': dataset_base_path + 'Flickr8k_text/Flickr_8k.devImages.txt',
            'test': dataset_base_path + 'Flickr8k_text/Flickr_8k.testImages.txt'
        }

        self.device = torch.device(device)
        self.torch = torch.cuda if (self.device.type == 'cuda') else torch

        self.return_raw = return_raw

        self.imgpath_list = glob.glob(self.images_path + '*.jpg')
        self.all_imgname_to_caplist = self.__all_imgname_to_caplist_dict()
        self.imgpath_to_caplist = self.__get_imgpath_to_caplist_dict(self.__get_imgpath_list(dist=dist))

        self.transformations = transformations if transformations is not None else transforms.Compose([
            transforms.ToTensor()
        ])

        self.startseq = startseq.strip()
        self.endseq = endseq.strip()
        self.unkseq = unkseq.strip()

        if vocab_set is None:
            self.vocab, self.word2idx, self.idx2word, self.max_len = self.__construct_vocab()
        else:
            self.vocab, self.word2idx, self.idx2word, self.max_len = vocab_set
        self.db = self.get_db()

    def __all_imgname_to_caplist_dict(self):
        captions = open(self.token, 'r').read().strip().split('\n')
        imgname_to_caplist = {}
        for i, row in enumerate(captions):
            row = row.split('\t')
            row[0] = row[0][:len(row[0]) - 2]  # filename#0 caption
            if row[0] in imgname_to_caplist:
                imgname_to_caplist[row[0]].append(row[1])
            else:
                imgname_to_caplist[row[0]] = [row[1]]
        return imgname_to_caplist

    def __get_imgpath_to_caplist_dict(self, img_path_list):
        d = {}
        for i in img_path_list:
            if i[len(self.images_path):] in self.all_imgname_to_caplist:
                d[i] = self.all_imgname_to_caplist[i[len(self.images_path):]]
        return d

    def __get_imgpath_list(self, dist='val'):
        dist_images = set(open(self.dist_list[dist], 'r').read().strip().split('\n'))
        dist_imgpathlist = split_data(dist_images, img=self.imgpath_list, images=self.images_path)
        return dist_imgpathlist

    def __construct_vocab(self):
        words = [self.startseq, self.endseq, self.unkseq]
        max_len = 0
        for key, caplist in self.imgpath_to_caplist.items():
            for cap in caplist:
                cap_words = nltk.word_tokenize(cap)
                words.extend(cap_words)
                max_len = max(max_len, len(cap_words) + 2)
        vocab = sorted(list(set(words)))

        word2idx = {word: index for index, word in enumerate(vocab)}
        idx2word = {index: word for index, word in enumerate(vocab)}

        return vocab, word2idx, idx2word, max_len

    def get_vocab(self):
        return self.vocab, self.word2idx, self.idx2word, self.max_len

    def get_db(self):
        # ----- Forming a df to sample from ------
        l = ["image_id\tcaption\tcaption_length\n"]
        # pil_d = {}
        for imgpath, caplist in self.imgpath_to_caplist.items():
            # pil_d[imgpath[len(self.images_path):]] = Image.open(imgpath).convert('RGB')
            for cap in caplist:
                l.append(
                    f"{imgpath[len(self.images_path):]}\t"
                    f"{cap}\t"  # {self.startseq} {cap} {self.endseq}
                    f"{len(nltk.word_tokenize(cap))}\n")
        img_id_cap_str = ''.join(l)

        df = pd.read_csv(io.StringIO(img_id_cap_str), delimiter='\t')
        return df.to_numpy()

    def __getitem__(self, index: int):
        imgname = self.db[index][0]
        caption = self.db[index][1]
        capt_ln = self.db[index][2]
        if self.return_raw:
            return os.path.join(self.images_path, imgname), caption, capt_ln
        cap_toks = [self.startseq] + nltk.word_tokenize(self.db[index][1]) + [self.endseq]
        img_tens = Image.open(os.path.join(self.images_path, imgname)).convert('RGB')  # self.pil_d[imgname]
        img_tens = self.transformations(img_tens).to(self.device)
        cap_tens = self.torch.LongTensor(self.max_len).fill_(0)
        cap_tens[:len(cap_toks)] = self.torch.LongTensor([self.word2idx[word] for word in cap_toks])
        return img_tens, cap_tens, self.torch.LongTensor([len(cap_toks)])

    def __len__(self):
        return len(self.db)
