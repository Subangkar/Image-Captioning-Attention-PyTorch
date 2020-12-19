import itertools
import os

import numpy as np
import torch
import wandb
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu
from tqdm.auto import tqdm


def preprocess_input(x):
    x -= 0.5
    x *= 2.
    return x


# returns (3, h, w)
def preprocess(image_path, trans):
    img = Image.open(image_path).convert('RGB')
    x = trans(img)
    x = x.unsqueeze(0)
    # x = preprocess_input(x)
    return x


def greedy_predictions_gen(encoding_dict, model, word2idx, idx2word, images, max_len,
                           startseq="<start>", endseq="<end>", device=torch.device('cpu')):
    def greedy_search_predictions_util(image):
        start_word = [startseq]
        with torch.no_grad():
            while True:
                par_caps = torch.LongTensor([word2idx[i] for i in start_word])
                par_caps = padding_tensor([par_caps], maxlen=max_len).to(device=device)
                e = encoding_dict[image[len(images):]].unsqueeze(0)
                preds = model(e, par_caps).cpu().numpy()
                word_pred = idx2word[np.argmax(preds[0])]  # [0] is for first elm of batch
                start_word.append(word_pred)

                if word_pred == endseq or len(start_word) > max_len:
                    break
        return ' '.join(start_word[1:-1])

    return greedy_search_predictions_util


def beam_search_predictions_gen(beam_index, encoding_dict, model, word2idx, idx2word, images, max_len,
                                startseq="<start>", endseq="<end>", device=torch.device('cpu')):
    def beam_search_predictions_util(image):
        start = [word2idx[startseq]]

        start_word = [[start, 0.0]]

        while len(start_word[0][0]) < max_len:
            temp = []
            for s in start_word:
                with torch.no_grad():
                    par_caps = torch.LongTensor(s[0])
                    par_caps = padding_tensor([par_caps], maxlen=max_len).to(device=device)
                    e = encoding_dict[image[len(images):]].unsqueeze(0)
                    preds = model(e, par_caps).cpu().numpy()

                    word_preds = np.argsort(preds[0])[-beam_index:]

                    # Getting the top <beam_index>(n) predictions and creating a
                    # new list so as to put them via the model again
                    for w in word_preds:
                        next_cap, prob = s[0][:], s[1]
                        next_cap.append(w)
                        prob += preds[0][w]
                        temp.append([next_cap, prob])

            start_word = temp
            # Sorting according to the probabilities
            start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
            # Getting the top words
            start_word = start_word[-beam_index:]

        start_word = start_word[-1][0]
        intermediate_caption = [idx2word[i] for i in start_word]

        final_caption = []

        for i in intermediate_caption:
            if i != endseq:
                final_caption.append(i)
            else:
                break

        final_caption = ' '.join(final_caption[1:])
        return final_caption

    return beam_search_predictions_util


def split_data(l, img, images):
    temp = []
    for i in img:
        if i[len(images):] in l:
            temp.append(i)
    return temp


def get_bleu_score(img_to_caplist_dict, caption_gen_func, device=torch.device('cpu')):
    bleu_score = 0.0
    for k, v in tqdm(img_to_caplist_dict.items()):
        candidate = caption_gen_func(k).split()
        references = [s.split() for s in v]
        bleu_score += sentence_bleu(references, candidate)
    return bleu_score / len(img_to_caplist_dict)


def print_eval_metrics(img_cap_dict, encoding_dict, model, word2idx, idx2word, images, max_len,
                       device=torch.device('cpu')):
    print('\t\tGreedy:            ',
          get_bleu_score(img_cap_dict, greedy_predictions_gen(encoding_dict=encoding_dict, model=model,
                                                              word2idx=word2idx, idx2word=idx2word,
                                                              images=images, max_len=max_len)))
    for k in [3, 5, 7]:
        print(f'\t\tBeam Search k={k}:', get_bleu_score(img_cap_dict,
                                                        beam_search_predictions_gen(beam_index=k,
                                                                                    encoding_dict=encoding_dict,
                                                                                    model=model,
                                                                                    word2idx=word2idx,
                                                                                    idx2word=idx2word,
                                                                                    images=images, max_len=max_len)))


def padding_tensor(sequences, maxlen):
    """
    :param sequences: list of tensors
    :param maxlen: fixed length of output tensors
    :return:
    """
    num = len(sequences)
    # max_len = max([s.size(0) for s in sequences])
    out_dims = (num, maxlen)
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
    return out_tensor


def words_from_tensors_fn(idx2word, max_len=40, startseq='<start>', endseq='<end>'):
    def words_from_tensors(captions: np.array) -> list:
        """
        :param captions: [b, max_len]
        :return:
        """
        captoks = []
        for capidx in captions:
            # capidx = [1, max_len]
            captoks.append(list(itertools.takewhile(lambda word: word != endseq,
                                                    map(lambda idx: idx2word[idx], iter(capidx))))[1:])
        return captoks

    return words_from_tensors


def sync_files_wandb(file_path_list):
    for path in file_path_list:
        if os.path.isfile(path) and os.access(path, os.R_OK):
            wandb.save(path)
            print(f'synced {path}')
        else:
            print("Either the file is missing or not readable")
