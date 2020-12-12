# %%
import math

from IPython.core.display import display
from PIL import Image
import keras

from utils import *
from datasets.flickr8kCap import Flickr8k

# %%

DATASET_BASE_PATH = 'data/flickr8k/'

# %%
dset = Flickr8k(dataset_base_path=DATASET_BASE_PATH)

train_img = dset.get_imgpathlist(dist='train')
val_img = dset.get_imgpathlist(dist='val')
test_img = dset.get_imgpathlist(dist='test')
len(train_img), len(val_img), len(test_img)

# %%

train_d = dset.imgfilename_to_caplist_dict(img_path_list=train_img)
val_d = dset.imgfilename_to_caplist_dict(img_path_list=val_img)
test_d = dset.imgfilename_to_caplist_dict(img_path_list=test_img)
len(train_d), len(val_d), len(test_d)

# %%

caps = dset.add_start_end_seq(train_d)
vocab, word2idx, idx2word, max_len = dset.construct_vocab(caps=caps)
vocab_size = len(vocab)
vocab_size, max_len

# %%

samples_per_epoch = sum(map(lambda cap: len(cap.split()) - 1, caps))
samples_per_epoch_val = sum(map(lambda cap: len(cap.split()) - 1, dset.add_start_end_seq(val_d)))
samples_per_epoch_test = sum(map(lambda cap: len(cap.split()) - 1, dset.add_start_end_seq(test_d)))
samples_per_epoch, samples_per_epoch_val, samples_per_epoch_test

# %%

from models.keras.incepv3_bidirlstm import Encoder

encoder = Encoder()
encoding_train = encoder.encode(dset.images, train_img)
encoding_valid = encoder.encode(dset.images, val_img)
encoding_test = encoder.encode(dset.images, test_img)

# %%

from models.keras.incepv3_bidirlstm import Decoder

final_model = Decoder(embedding_size=300, vocab_size=vocab_size, max_len=max_len).get_model()
opt = keras.optimizers.Adam(learning_rate=1e-3)
final_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
final_model.summary()

# %%

BATCH_SIZE = 256
MODEL_NAME = f'saved_models/IncepV3_bidirlstm_b{BATCH_SIZE}'
NUM_EPOCHS = 50
steps_per_epoch = int(math.ceil(samples_per_epoch / BATCH_SIZE))
steps_per_epoch_val = int(math.ceil(samples_per_epoch_val / BATCH_SIZE))
steps_per_epoch_test = int(math.ceil(samples_per_epoch_test / BATCH_SIZE))
steps_per_epoch, steps_per_epoch_val, steps_per_epoch_test

# %%

final_model.fit(
    x=dset.get_generator(batch_size=BATCH_SIZE, random_state=None,
                         encoding_train=encoding_train, imgfilename_to_caplist_dict=train_d,
                         word2idx=word2idx, vocab_size=vocab_size, max_len=max_len),
    steps_per_epoch=steps_per_epoch,
    epochs=NUM_EPOCHS,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            f'{MODEL_NAME}''_ep{epoch:02d}_weights.h5',
            save_weights_only=True, period=20),
        keras.callbacks.EarlyStopping(patience=10, monitor='loss'),
        keras.callbacks.ModelCheckpoint(f'{MODEL_NAME}''_best_train.h5', monitor='loss',
                                        save_best_only=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-5),
    ],
    validation_data=dset.get_generator(batch_size=BATCH_SIZE, random_state=0,
                                       encoding_train=encoding_valid, imgfilename_to_caplist_dict=val_d,
                                       word2idx=word2idx, vocab_size=vocab_size, max_len=max_len),
    validation_steps=steps_per_epoch_val
)
final_model.save(f"{MODEL_NAME}_ep{NUM_EPOCHS}.h5")

# %%

final_model.fit(
    x=dset.get_generator(batch_size=BATCH_SIZE, random_state=None,
                         encoding_train=encoding_train, imgfilename_to_caplist_dict=train_d,
                         word2idx=word2idx, vocab_size=vocab_size, max_len=max_len),
    steps_per_epoch=steps_per_epoch,
    epochs=100, initial_epoch=50,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            f'{MODEL_NAME}''_ep{epoch:02d}_weights.h5',
            save_weights_only=True, period=10),
        keras.callbacks.EarlyStopping(patience=5, monitor='loss'),
        keras.callbacks.ModelCheckpoint(f'{MODEL_NAME}''_best_train.h5', monitor='loss',
                                        save_best_only=True, mode='min'),
    ])
final_model.save(f"{MODEL_NAME}_ep{100}.h5")

# %%

# model = keras.models.load_model(f'{MODEL_NAME}''_best_train.h5')
model = final_model

# %%

try_image = train_img[100]
display(Image.open(try_image))
print('Normal Max search:', greedy_predictions_gen(encoding_dict=encoding_train, model=model,
                                                   word2idx=word2idx, idx2word=idx2word,
                                                   images=dset.images, max_len=max_len)(try_image))
for k in [3, 5, 7]:
    print(f'Beam Search, k={k}:',
          beam_search_predictions_gen(beam_index=k, encoding_dict=encoding_train, model=model,
                                      word2idx=word2idx, idx2word=idx2word,
                                      images=dset.images, max_len=max_len)(try_image))

# %%

try_image = val_img[4]
display(Image.open(try_image))
print('Normal Max search:', greedy_predictions_gen(encoding_dict=encoding_valid, model=model,
                                                   word2idx=word2idx, idx2word=idx2word,
                                                   images=dset.images, max_len=max_len)(try_image))
for k in [3, 5, 7]:
    print(f'Beam Search, k={k}:',
          beam_search_predictions_gen(beam_index=k, encoding_dict=encoding_valid, model=model,
                                      word2idx=word2idx, idx2word=idx2word,
                                      images=dset.images, max_len=max_len)(try_image))

# %%

try_image = test_img[4]
display(Image.open(try_image))
print('Normal Max search:', greedy_predictions_gen(encoding_dict=encoding_test, model=model,
                                                   word2idx=word2idx, idx2word=idx2word,
                                                   images=dset.images, max_len=max_len)(try_image))
for k in [3, 5, 7]:
    print(f'Beam Search, k={k}:',
          beam_search_predictions_gen(beam_index=k, encoding_dict=encoding_test, model=model,
                                      word2idx=word2idx, idx2word=idx2word,
                                      images=dset.images, max_len=max_len)(try_image))

# %%

print("BLEU Scores:")
print("\tTrain")
print_eval_metrics(img_cap_dict=train_d, encoding_dict=encoding_train, model=model,
                   word2idx=word2idx, idx2word=idx2word,
                   images=dset.images, max_len=max_len)
print("\tValidation")
print_eval_metrics(img_cap_dict=val_d, encoding_dict=encoding_valid, model=model,
                   word2idx=word2idx, idx2word=idx2word,
                   images=dset.images, max_len=max_len)
print("\tTest")
print_eval_metrics(img_cap_dict=test_d, encoding_dict=encoding_test, model=model,
                   word2idx=word2idx, idx2word=idx2word,
                   images=dset.images, max_len=max_len)
