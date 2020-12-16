from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def bleu_score_fn(method_no: int = 4, ref_type='corpus'):
    """
    :param method_no:
    :param ref_type: 'corpus' or 'sentence'
    :return: bleu score
    """
    smoothing_method = getattr(SmoothingFunction(), f'method{method_no}')

    def bleu_score_corpus(reference_corpus: list, candidate_corpus: list):
        """
        :param reference_corpus: [b, 5, var_len]
        :param candidate_corpus: [b, var_len]
        """
        return corpus_bleu(reference_corpus, candidate_corpus, smoothing_function=smoothing_method)

    def bleu_score_sentence(reference_sentences: list, candidate_sentence: list):
        """
        :param reference_sentences: [5, var_len]
        :param candidate_sentence: [var_len]
        """
        return sentence_bleu(reference_sentences, candidate_sentence, smoothing_function=smoothing_method)

    if ref_type == 'corpus':
        return bleu_score_corpus
    elif ref_type == 'sentence':
        return bleu_score_sentence
