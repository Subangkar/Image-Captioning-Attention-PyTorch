import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision

from models.torch.decoders.monolstm import Decoder


class Encoder(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained Inception-v3 and replace top fc layer."""
        super(Encoder, self).__init__()
        inception_v3 = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
        modules = list(inception_v3.children())[:-1]
        self.inception_v3 = nn.Sequential(*modules)
        self.embed = nn.Sequential(
            nn.Linear(inception_v3.fc.in_features, embed_size),
            nn.Dropout(p=0.5),
        )
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.inception_v3(images)
            features = F.relu(features, inplace=True).view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features


class Captioner(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, embedding_matrix=None, train_embd=True):
        super().__init__()
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers,
                               embedding_matrix=embedding_matrix, train_embd=train_embd)

    def forward(self, images, captions, lengths):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        return outputs

    def sample(self, images, max_len=40, endseq_idx=-1):
        features = self.encoder(images)
        captions = self.decoder.sample(features=features, max_len=max_len, endseq_idx=endseq_idx)
        return captions

    def sample_beam_search(self, images, max_len=40, endseq_idx=-1, beam_width=5):
        features = self.encoder(images)
        captions = self.decoder.sample_beam_search(features=features, max_len=max_len, beam_width=beam_width)
        return captions
