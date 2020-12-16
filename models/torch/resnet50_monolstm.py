import torch
import torch.nn as nn

from models.torch.decoders.monolstm import Decoder


class Encoder(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-50 and replace top fc layer."""
        super(Encoder, self).__init__()
        resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
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
