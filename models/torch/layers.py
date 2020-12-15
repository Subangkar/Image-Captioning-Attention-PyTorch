import torch
import torch.nn as nn


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


def embedding_layer(trainable=True, embedding_matrix=None, **kwargs):
    emb_layer = nn.Embedding(**kwargs)
    if embedding_matrix is not None:
        emb_layer.weight = nn.Parameter(torch.from_numpy(embedding_matrix).float())
    trainable = (embedding_matrix is None) or trainable
    if not trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer
