import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing import image
from tqdm.auto import tqdm
from nltk.translate.bleu_score import sentence_bleu


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def preprocess(image_path, target_shape=(299, 299)):
    img = image.load_img(image_path, target_size=target_shape)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    return x


def greedy_predictions_gen(encoding_dict, model, word2idx, idx2word, images, max_len,
                           startseq="<start>", endseq="<end>"):
    def greedy_search_predictions_util(image):
        start_word = [startseq]
        while True:
            par_caps = [word2idx[i] for i in start_word]
            par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
            e = encoding_dict[image[len(images):]]
            preds = model.predict([np.array([e]), np.array(par_caps)])
            word_pred = idx2word[np.argmax(preds[0])]
            start_word.append(word_pred)

            if word_pred == endseq or len(start_word) > max_len:
                break

        return ' '.join(start_word[1:-1])

    return greedy_search_predictions_util


def beam_search_predictions_gen(beam_index, encoding_dict, model, word2idx, idx2word, images, max_len,
                                startseq="<start>", endseq="<end>"):
    def beam_search_predictions_util(image):
        start = [word2idx[startseq]]

        start_word = [[start, 0.0]]

        while len(start_word[0][0]) < max_len:
            temp = []
            for s in start_word:
                par_caps = sequence.pad_sequences([s[0]], maxlen=max_len, padding='post')
                e = encoding_dict[image[len(images):]]
                preds = model.predict([np.array([e]), np.array(par_caps)])

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


def get_bleu_score(img_to_caplist_dict, caption_gen_func):
    bleu_score = 0.0
    for k, v in tqdm(img_to_caplist_dict.items()):
        candidate = caption_gen_func(k).split()
        references = [s.split() for s in v]
        bleu_score += sentence_bleu(references, candidate)
    return bleu_score / len(img_to_caplist_dict)


def print_eval_metrics(img_cap_dict, encoding_dict, model, word2idx, idx2word, images, max_len):
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
