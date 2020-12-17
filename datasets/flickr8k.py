import glob
import io
import ntpath
import os

import nltk
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils_torch import split_data


class Flickr8kDataset(Dataset):
    """
    imgname: just image file name
    imgpath: full path to image file
    """

    def __init__(self, dataset_base_path='data/flickr8k/',
                 vocab_set=None, dist='val',
                 startseq="<start>", endseq="<end>", unkseq="<unk>", padseq="<pad>",
                 transformations=None,
                 return_raw=False,
                 load_img_to_memory=False,
                 return_type='tensor',
                 device=torch.device('cpu')):
        self.token = dataset_base_path + 'Flickr8k_text/Flickr8k.token.txt'
        self.images_path = dataset_base_path + 'Flicker8k_Dataset/'

        self.dist_list = {
            'train': dataset_base_path + 'Flickr8k_text/Flickr_8k.trainImages.txt',
            'val': dataset_base_path + 'Flickr8k_text/Flickr_8k.devImages.txt',
            'test': dataset_base_path + 'Flickr8k_text/Flickr_8k.testImages.txt'
        }

        self.load_img_to_memory = load_img_to_memory
        self.pil_d = None

        self.device = torch.device(device)
        self.torch = torch.cuda if (self.device.type == 'cuda') else torch

        self.return_raw = return_raw
        self.return_type = return_type

        self.__get_item__fn = self.__getitem__corpus if return_type == 'corpus' else self.__getitem__tensor

        self.imgpath_list = glob.glob(self.images_path + '*.jpg')
        self.all_imgname_to_caplist = self.__all_imgname_to_caplist_dict()
        self.imgname_to_caplist = self.__get_imgname_to_caplist_dict(self.__get_imgpath_list(dist=dist))

        self.transformations = transformations if transformations is not None else transforms.Compose([
            transforms.ToTensor()
        ])

        self.startseq = startseq.strip()
        self.endseq = endseq.strip()
        self.unkseq = unkseq.strip()
        self.padseq = padseq.strip()

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

    def __get_imgname_to_caplist_dict(self, img_path_list):
        d = {}
        for i in img_path_list:
            if i[len(self.images_path):] in self.all_imgname_to_caplist:
                d[ntpath.basename(i)] = self.all_imgname_to_caplist[i[len(self.images_path):]]
        return d

    def __get_imgpath_list(self, dist='val'):
        dist_images = set(open(self.dist_list[dist], 'r').read().strip().split('\n'))
        dist_imgpathlist = split_data(dist_images, img=self.imgpath_list, images=self.images_path)
        return dist_imgpathlist

    def __construct_vocab(self):
        words = [self.startseq, self.endseq, self.unkseq, self.padseq]
        max_len = 0
        for _, caplist in self.imgname_to_caplist.items():
            for cap in caplist:
                cap_words = nltk.word_tokenize(cap.lower())
                words.extend(cap_words)
                max_len = max(max_len, len(cap_words) + 2)
        vocab = sorted(list(set(words)))

        word2idx = {word: index for index, word in enumerate(vocab)}
        idx2word = {index: word for index, word in enumerate(vocab)}

        return vocab, word2idx, idx2word, max_len

    def get_vocab(self):
        return self.vocab, self.word2idx, self.idx2word, self.max_len

    def get_db(self):

        if self.load_img_to_memory:
            self.pil_d = {}
            for imgname in self.imgname_to_caplist.keys():
                self.pil_d[imgname] = Image.open(os.path.join(self.images_path, imgname)).convert('RGB')

        if self.return_type == 'corpus':
            df = []
            for imgname, caplist in self.imgname_to_caplist.items():
                cap_wordlist = []
                cap_lenlist = []
                for caption in caplist:
                    toks = nltk.word_tokenize(caption.lower())
                    cap_wordlist.append(toks)
                    cap_lenlist.append(len(toks))
                df.append([imgname, cap_wordlist, cap_lenlist])
            return df

        # ----- Forming a df to sample from ------
        l = ["image_id\tcaption\tcaption_length\n"]

        for imgname, caplist in self.imgname_to_caplist.items():
            for cap in caplist:
                l.append(
                    f"{imgname}\t"
                    f"{cap.lower()}\t"
                    f"{len(nltk.word_tokenize(cap.lower()))}\n")
        img_id_cap_str = ''.join(l)

        df = pd.read_csv(io.StringIO(img_id_cap_str), delimiter='\t')
        return df.to_numpy()

    @property
    def pad_value(self):
        return 0

    def __getitem__(self, index: int):
        return self.__get_item__fn(index)

    def __len__(self):
        return len(self.db)

    def get_image_captions(self, index: int):
        """
        :param index: [] index
        :returns: image_path, list_of_captions
        """
        imgname = self.db[index][0]
        return os.path.join(self.images_path, imgname), self.imgname_to_caplist[imgname]

    def __getitem__tensor(self, index: int):
        imgname = self.db[index][0]
        caption = self.db[index][1]
        capt_ln = self.db[index][2]
        cap_toks = [self.startseq] + nltk.word_tokenize(caption) + [self.endseq]
        img_tens = self.pil_d[imgname] if self.load_img_to_memory else Image.open(
            os.path.join(self.images_path, imgname)).convert('RGB')
        img_tens = self.transformations(img_tens).to(self.device)
        cap_tens = self.torch.LongTensor(self.max_len).fill_(self.pad_value)
        cap_tens[:len(cap_toks)] = self.torch.LongTensor([self.word2idx[word] for word in cap_toks])
        return img_tens, cap_tens, len(cap_toks)

    def __getitem__corpus(self, index: int):
        imgname = self.db[index][0]
        cap_wordlist = self.db[index][1]
        cap_lenlist = self.db[index][2]
        img_tens = self.pil_d[imgname] if self.load_img_to_memory else Image.open(
            os.path.join(self.images_path, imgname)).convert('RGB')
        img_tens = self.transformations(img_tens).to(self.device)
        return img_tens, cap_wordlist, cap_lenlist
