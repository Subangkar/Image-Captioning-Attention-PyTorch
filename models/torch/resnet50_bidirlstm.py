import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from tqdm.auto import tqdm

from models.torch import layers
from utils_torch import preprocess


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet50 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50',
                                  pretrained=True)  # torchvision.models.resnet50(pretrained=True)  # pretrained ImageNet ResNet-50
        self.resnet50 = nn.Sequential(*(list(resnet50.children())[:-1]))
        # for p in self.resnet50.parameters():
        #     p.requires_grad = False

    def forward(self, x):
        out = self.resnet50(x)
        return out

    def encode(self, image_dset_path, image_dist_set, batch_size=32, device=torch.device('cpu')):
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        encoding_set = {}
        with torch.no_grad():
            for image in tqdm(image_dist_set):
                # (h,w) -> (1,3,224,224)
                temp_enc = self(preprocess(image, trans=trans).to(device))
                # (1,3,224,224) -> (2048,)
                encoding_set[image[len(image_dset_path):]] = torch.reshape(temp_enc, shape=(temp_enc.shape[1],))
        return encoding_set


class Decoder(nn.Module):
    def __init__(self, embedding_size, vocab_size, max_len):
        super().__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.max_len = max_len

        # (b, embedding_size)
        self.image_model = nn.Sequential(
            # (b, embedding_size)
            nn.Linear(in_features=2048, out_features=embedding_size),
            nn.ReLU(),
        )

        # (b, max_len, 512)
        self.caption_model = nn.Sequential(
            # (b, max_len, embedding_size)
            nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size),  # (b, max_len, embedding_size)
            # (b, max_len, 512)
            nn.LSTM(input_size=embedding_size, hidden_size=256, bidirectional=True, batch_first=True, )
            # dropout=0.5),
        )

        # (b, max_len, embedding_size)
        self.timedist = nn.Sequential(
            nn.Dropout(p=0.5),
            # (b, max_len, 512)
            nn.BatchNorm1d(num_features=max_len),
            # (b, max_len, embedding_size)
            layers.TimeDistributed(nn.Linear(in_features=512, out_features=embedding_size), batch_first=True),
        )

        # (b, 2000)
        self.decoder_model = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(num_features=max_len),
            # (b, max_len, 2*embedding_size) -> (b, max_len, 2000)
            nn.LSTM(input_size=2 * embedding_size, hidden_size=1000, bidirectional=True, batch_first=True, )
            # dropout=0.5),
        )

        self.pred_layer = nn.Linear(in_features=2000, out_features=vocab_size)

    def forward(self, x_img, x_cap):
        # (b, embedding_size) -> (b, max_len, embedding_size)
        x_img_in = self.image_model(x_img).view(-1, 1, self.embedding_size).repeat(1, self.max_len, 1)
        # (b, max_len) -> (b, max_len, embedding_size)
        x_cap_in = self.timedist(self.caption_model(x_cap)[0])

        # (b, max_len, embedding_size) -> (b, max_len, 2*embedding_size)
        X = torch.cat([x_img_in, x_cap_in], dim=2)
        # (b, max_len, embedding_size) -> (b, max_len, 2000) -> (b, 2000)
        X = self.decoder_model(X)[0][:, -1, :]
        X = self.pred_layer(X)
        return X
