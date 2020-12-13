import math

import torch
from PIL import Image
from torch.utils.data import Dataset
import glob
import pandas as pd
import io
from torchvision import transforms

from utils_torch import split_data
from utils_torch import padding_tensor


class Flickr8kDataset(Dataset):
    """
    imgname: just image file name
    imgpath: full path to image file
    """

    def __init__(self, dataset_base_path='data/flickr8k/', vocab=None):
        self.token = dataset_base_path + 'Flickr8k_text/Flickr8k.token.txt'
        self.images = dataset_base_path + 'Flicker8k_Dataset/'

        self.dist_list = {
            'train': dataset_base_path + 'Flickr8k_text/Flickr_8k.trainImages.txt',
            'val': dataset_base_path + 'Flickr8k_text/Flickr_8k.devImages.txt',
            'test': dataset_base_path + 'Flickr8k_text/Flickr_8k.testImages.txt'
        }

        self.imgpath_list = glob.glob(self.images + '*.jpg')
        self.all_imgname_to_caplist = self.__all_imgname_to_caplist_dict()
        self.imgpath_to_caplist = self.get_imgpath_to_caplist_dict(self.get_imgpath_list())

        # if vocab is None:
        #     self.vocab, self.word2idx, self.idx2word, self.max_len = self.construct_vocab(self.imgpath_to_caplist)

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

    def get_imgpath_to_caplist_dict(self, img_path_list):
        d = {}
        for i in img_path_list:
            if i[len(self.images):] in self.all_imgname_to_caplist:
                d[i] = self.all_imgname_to_caplist[i[len(self.images):]]
        return d

    def get_imgpath_list(self, dist='val'):
        dist_images = set(open(self.dist_list[dist], 'r').read().strip().split('\n'))
        dist_imgpathlist = split_data(dist_images, img=self.imgpath_list, images=self.images)
        return dist_imgpathlist

    def add_start_end_seq(self, imgpath_to_caplist_dict, startseq="<start> ", endseq="<end>"):
        caps = []
        for key, val in imgpath_to_caplist_dict.items():
            for i in val:
                caps.append(f'{startseq} {i} {endseq}')
        return caps

    def construct_vocab(self, caps):
        words = [i.split() for i in caps]
        unique = []
        for i in words:
            unique.extend(i)
        vocab = sorted(list(set(unique)))

        word2idx = {val: index for index, val in enumerate(vocab)}
        idx2word = {index: val for index, val in enumerate(vocab)}

        return vocab, word2idx, idx2word, max(map(lambda w: len(w), words))

    # def __getitem__(self, index: int):

    def __len__(self):
        return len(self.imgpath_to_caplist)

    def get_generator(self, batch_size, encoding_dict, imgpath_to_caplist_dict, word2idx, max_len,
                      random_state=17, device=torch.device('cpu')):
        transform_train = transforms.Compose([
            transforms.Resize(256),  # smaller edge of image resized to 256
            transforms.RandomCrop(224),  # get 224x224 crop from random location
            # transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
            transforms.ToTensor(),  # convert the PIL Image to a tensor
            transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                                 (0.229, 0.224, 0.225))
        ])

        # ----- Forming a df to sample from ------
        l = ["image_id\tcaptions\n"]
        encoding_dict = {}
        for key, val in imgpath_to_caplist_dict.items():
            encoding_dict[key[len(self.images):]] = transform_train(Image.open(key))
            for i in val:
                l.append(''.join([key[len(self.images):], "\t", "<start> ", i, " <end>", "\n"]))
        img_id_cap_str = ''.join(l)

        df = pd.read_csv(io.StringIO(img_id_cap_str), delimiter='\t')

        # ---------- Generator -------
        df = df.sample(frac=1, random_state=random_state)
        iter = df.iterrows()
        c = []  # list of captions
        imgs = []  # list of imgname
        for i in range(df.shape[0]):
            x = next(iter)
            c.append(x[1][1])
            imgs.append(x[1][0])
        for b in range(int(math.ceil(len(c) / batch_size))):
            st = b * batch_size
            en = min(st + batch_size, len(c))
            img_tens = [encoding_dict[im] for im in imgs[st:en]]
            cap_tens = [torch.LongTensor([word2idx[word] for word in caption.split()]) for caption in c[st:en]]
            yield torch.stack(img_tens).to(device=device), padding_tensor(cap_tens, maxlen=max_len).to(device=device)
