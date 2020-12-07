import math

from IPython.core.display import display
from PIL import Image
import numpy as np
import pandas as pd
import keras

from utils import *
from datasets.flickr8kCap import Flickr8k

dset = Flickr8k()

train_img = dset.get_imgpathlist(dist='train')
len(train_img)

val_img = dset.get_imgpathlist(dist='val')
len(val_img)

test_img = dset.get_imgpathlist(dist='test')
len(test_img)

# ---------------------------
# %%

train_d = dset.imgfilename_to_caplist_dict(img_path_list=train_img)
len(train_d)
val_d = dset.imgfilename_to_caplist_dict(img_path_list=val_img)
len(val_d)
test_d = dset.imgfilename_to_caplist_dict(img_path_list=test_img)
len(test_d)

# %%
caps = dset.add_start_end_seq(train_d)
vocab, word2idx, idx2word, max_len = dset.construct_vocab(caps=caps)
vocab_size = len(vocab)

# %%
samples_per_epoch = sum(map(lambda cap: len(cap.split()) - 1, caps))
samples_per_epoch

# %%
from models.incepv3 import Encoder

encoder = Encoder()
encoding_train = encoder.encode(dset.images, train_img)
encoding_test = encoder.encode(dset.images, test_img)

# %%
from models.incepv3 import Decoder

final_model = Decoder(embedding_size=300, vocab_size=vocab_size, max_len=max_len).get_model()
opt = keras.optimizers.Adam(learning_rate=1e-3)
final_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
final_model.summary()
# %%
BATCH_SIZE = 16
MODEL_NAME = f'saved_models/IncepV3_bidir_b{BATCH_SIZE}'
steps_per_epoch = int(math.ceil(samples_per_epoch / BATCH_SIZE))

# %%
final_model.fit(
    x=dset.get_generator(batch_size=BATCH_SIZE, random_state=None,
                         encoding_train=encoding_train, imgfilename_to_caplist_dict=train_d,
                         word2idx=word2idx, vocab_size=vocab_size, max_len=max_len),
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            f'{MODEL_NAME}''_ep{epoch:02d}_weights.h5',
            save_weights_only=True, period=20),
        keras.callbacks.EarlyStopping(patience=3, monitor='loss'),
        keras.callbacks.ModelCheckpoint(f'{MODEL_NAME}''_best_train.h5', monitor='loss',
                                        save_best_only=True, mode='min'),
    ])
final_model.save(f"{MODEL_NAME}_ep{3}.h5")
# %%
try_image = train_img[100]
# imshow(np.asarray(Image.open(try_image)))
display(Image.open(try_image))
print('Normal Max search:', predict_captions(try_image, encoding_test=encoding_train, final_model=final_model,
                                             word2idx=word2idx, idx2word=idx2word,
                                             images=dset.images, max_len=max_len))
print('Beam Search, k=3:',
      beam_search_predictions(try_image, beam_index=3, encoding_test=encoding_train, final_model=final_model,
                              word2idx=word2idx, idx2word=idx2word,
                              images=dset.images, max_len=max_len))
print('Beam Search, k=5:',
      beam_search_predictions(try_image, beam_index=5, encoding_test=encoding_train, final_model=final_model,
                              word2idx=word2idx, idx2word=idx2word,
                              images=dset.images, max_len=max_len))
print('Beam Search, k=7:',
      beam_search_predictions(try_image, beam_index=7, encoding_test=encoding_train, final_model=final_model,
                              word2idx=word2idx, idx2word=idx2word,
                              images=dset.images, max_len=max_len))
