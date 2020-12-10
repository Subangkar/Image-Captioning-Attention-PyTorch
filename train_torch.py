# %%
import math

from IPython.core.display import display
from PIL import Image
import torch
from tqdm import trange

from utils_torch import *
from datasets.flickr8k import Flickr8kDataset

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%

DATASET_BASE_PATH = 'data/flickr8k/'

# %%
dset = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH)

train_img = dset.get_imgpath_list(dist='train')
val_img = dset.get_imgpath_list(dist='val')
test_img = dset.get_imgpath_list(dist='test')
len(train_img), len(val_img), len(test_img)

# %%

train_d = dset.get_imgpath_to_caplist_dict(img_path_list=train_img)
val_d = dset.get_imgpath_to_caplist_dict(img_path_list=val_img)
test_d = dset.get_imgpath_to_caplist_dict(img_path_list=test_img)
len(train_d), len(val_d), len(test_d)

# %%

caps = dset.add_start_end_seq(train_d)
vocab, word2idx, idx2word, max_len = dset.construct_vocab(caps=caps)
vocab_size = len(vocab)
vocab_size, max_len

# %%

samples_per_epoch = sum(map(lambda cap: len(cap.split()) - 1, caps))
samples_per_epoch


# %%
def train_model(model, train_generator, steps_per_epoch, optimizer, loss_fn, wandb_log=False):
    running_acc = 0
    running_loss = 0.0

    t = trange(steps_per_epoch, leave=True)
    for batch_idx in t:  # enumerate(iter(steps_per_epoch)):
        batch = next(train_generator)
        (enc, cap_in, next_word) = batch

        optimizer.zero_grad()
        output = model(enc, cap_in)
        loss = loss_fn(output, next_word)
        loss.backward()
        optimizer.step()

        running_acc += (torch.argmax(output, dim=1) == next_word).sum().item() / next_word.size(0)
        running_loss += loss.item()
        t.set_postfix({'loss': running_loss / (batch_idx + 1),
                       'acc': running_acc / (batch_idx + 1)}, refresh=True)

    return model, running_loss


# %%

from models.torch.resnet50_bidirlstm import Encoder

encoder = Encoder().to(device=device)
encoding_train = encoder.encode(dset.images, train_img, device=device)
encoding_valid = encoder.encode(dset.images, val_img, device=device)
encoding_test = encoder.encode(dset.images, test_img, device=device)

# %%

BATCH_SIZE = 256
MODEL_NAME = f'saved_models/resnet50_bidirlstm_emd200_b{BATCH_SIZE}'
steps_per_epoch = int(math.ceil(samples_per_epoch / BATCH_SIZE))

# %%

from models.torch.resnet50_bidirlstm import Decoder

final_model = Decoder(embedding_size=200, vocab_size=vocab_size, max_len=max_len).to(device=device)
optimizer = torch.optim.Adam(final_model.parameters(), lr=1E-3)
loss_fn = torch.nn.CrossEntropyLoss()

# %%
train_generator = dset.get_generator(batch_size=BATCH_SIZE, random_state=None, device=device,
                                     encoding_train=encoding_train, imgpath_to_caplist_dict=train_d,
                                     word2idx=word2idx, max_len=max_len)
train_loss_min = 100
for epoch in range(5):
    print(f'Epoch {epoch + 1}/{5}', flush=True)
    final_model.train()
    final_model, train_loss = train_model(model=final_model, optimizer=optimizer, loss_fn=loss_fn,
                                          train_generator=train_generator, steps_per_epoch=steps_per_epoch)
    state = {
        'epoch': epoch + 1,
        'state_dict': final_model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    if (epoch + 1) % 2 == 0:
        torch.save(state, f'{MODEL_NAME}_ep{epoch:02d}_weights.pt')
    if train_loss < train_loss_min:
        train_loss_min = train_loss
        torch.save(state, f'{MODEL_NAME}''_best_train.pt')
torch.save(final_model, f'{MODEL_NAME}_ep{5:02d}_weights.pt')
final_model.eval()

# %%

# model = torch.load(f'{MODEL_NAME}''_best_train.pt')
model = final_model

# %%

try_image = train_img[100]
display(Image.open(try_image))
print('Normal Max search:', greedy_predictions_gen(encoding_dict=encoding_train, model=model,
                                                   word2idx=word2idx, idx2word=idx2word,
                                                   images=dset.images, max_len=max_len, device=device)(try_image))
for k in [3, 5, 7]:
    print(f'Beam Search, k={k}:',
          beam_search_predictions_gen(beam_index=k, encoding_dict=encoding_train, model=model,
                                      word2idx=word2idx, idx2word=idx2word,
                                      images=dset.images, max_len=max_len, device=device)(try_image))

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
