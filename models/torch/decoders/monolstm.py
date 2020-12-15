import torch
from torch import nn as nn
from torch.nn import functional as F

from models.torch.layers import embedding_layer


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, embedding_matrix=None, train_embd=True):
        """Set the hyper-parameters and build the layers."""
        super(Decoder, self).__init__()
        self.embed = embedding_layer(num_embeddings=vocab_size, embedding_dim=embed_size,
                                     embedding_matrix=embedding_matrix, trainable=train_embd)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        """Decode image feature vectors and generates captions.
        features = [b, 300]
        captions = [b, max_len]
        """
        # [b, max_len] -> [b, max_len-1]
        captions = captions[:, :-1]
        # [b, max_len-1, embed_dim]
        embeddings = self.embed(captions)
        # [b, max_len, embed_dim]
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        # [b, max_len, hidden_size]
        hiddens, _ = self.lstm(inputs)
        # [b, max_len, vocab_size]
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, features, states=None, max_len=40, endseq_idx=-1):
        """Accept a pre-processed image tensor (inputs) and return predicted
        sentence (list of tensor ids of length max_len). This is the greedy
        search approach.
        limited to single element batch as input
        features = [1, embed_dim]
        inputs = [1, 1, embed_dim]
        """
        inputs = features.unsqueeze(1)
        sampled_ids = []
        for i in range(max_len):
            # [1, 1, 256]
            hiddens, states = self.lstm(inputs, states)
            # [1, 1, 256] -> [1, 256] -> [1, 8254]
            outputs = self.linear(hiddens.squeeze(1))
            # Get the index (in the vocabulary) of the most likely integer that
            # represents a word
            # [1]
            predicted = outputs.argmax(1)
            if predicted.item() == endseq_idx:
                break
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids

    def sample_beam_search(self, features, states=None, max_len=40, beam_width=5):
        """Accept a pre-processed image tensor and return the top predicted
        sentences. This is the beam search approach.
        """
        inputs = features.unsqueeze(1)
        # Top word idx sequences and their corresponding inputs and states
        idx_sequences = [[[], 0.0, inputs, states]]
        for _ in range(max_len):
            # Store all the potential candidates at each step
            all_candidates = []
            # Predict the next word idx for each of the top sequences
            for idx_seq in idx_sequences:
                hiddens, states = self.lstm(idx_seq[2], idx_seq[3])
                outputs = self.linear(hiddens.squeeze(1))
                # Transform outputs to log probabilities to avoid floating-point
                # underflow caused by multiplying very small probabilities
                log_probs = F.log_softmax(outputs, -1)
                top_log_probs, top_idx = log_probs.topk(beam_width, 1)
                top_idx = top_idx.squeeze(0)
                # create a new set of top sentences for next round
                for i in range(beam_width):
                    next_idx_seq, log_prob = idx_seq[0][:], idx_seq[1]
                    next_idx_seq.append(top_idx[i].item())
                    log_prob += top_log_probs[0][i].item()
                    # Indexing 1-dimensional top_idx gives 0-dimensional tensors.
                    # We have to expand dimensions before embedding them
                    inputs = self.embed(top_idx[i].unsqueeze(0)).unsqueeze(0)
                    all_candidates.append([next_idx_seq, log_prob, inputs, states])
            # Keep only the top sequences according to their total log probability
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            idx_sequences = ordered[:beam_width]
        return [idx_seq[0] for idx_seq in idx_sequences]
