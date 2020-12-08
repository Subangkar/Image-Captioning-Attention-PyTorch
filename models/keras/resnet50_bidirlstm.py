import numpy as np
from tqdm import tqdm
import keras
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation, Flatten, Concatenate, Input, \
    Dropout, BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.applications.resnet import ResNet50
from keras.models import Model

from utils import preprocess


class Encoder:
    def __init__(self):
        model = ResNet50(weights='imagenet')
        new_input = model.input
        hidden_layer = model.layers[-2].output

        model_new = Model(new_input, hidden_layer)
        self.model = model_new

    def encode(self, image_dset_path, image_dist_set):
        encoding_set = {}
        for image in tqdm(image_dist_set):
            temp_enc = self.model.predict(preprocess(image, target_shape=(224, 224)))
            encoding_set[image[len(image_dset_path):]] = np.reshape(temp_enc, temp_enc.shape[1])
        return encoding_set


class Decoder:
    def __init__(self, embedding_size, vocab_size, max_len):
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.image_model = Sequential([
            Dense(embedding_size, input_shape=(2048,), activation='relu'),
            RepeatVector(max_len)
        ])
        self.caption_model = Sequential([
            Embedding(vocab_size, embedding_size, input_length=max_len),
            Bidirectional(LSTM(256, return_sequences=True)),
            Dropout(0.5),
            BatchNormalization(),
            TimeDistributed(Dense(300))
        ])

    def get_model(self):
        image_model = self.image_model
        caption_model = self.caption_model

        image_in = Input(shape=(2048,))
        caption_in = Input(shape=(self.max_len,))

        merged = Concatenate()([image_model(image_in), caption_model(caption_in)])
        merged = Dropout(0.5)(merged)
        merged = BatchNormalization()(merged)
        latent = Bidirectional(LSTM(1000, return_sequences=False))(merged)
        out = Dense(self.vocab_size, activation='softmax')(latent)

        return Model([image_in, caption_in], out)
