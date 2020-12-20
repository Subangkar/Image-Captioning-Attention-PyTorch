# %%
import pickle
import wandb
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.flickr8k import Flickr8kDataset
from glove import embedding_matrix_creator
from metrics import *
from utils_torch import *

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device
# %%

DATASET_BASE_PATH = 'data/flickr8k/'

# %%

train_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, dist='train', device=device,
                            return_type='tensor',
                            load_img_to_memory=False)
vocab, word2idx, idx2word, max_len = vocab_set = train_set.get_vocab()
val_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, dist='val', vocab_set=vocab_set, device=device,
                          return_type='corpus',
                          load_img_to_memory=False)
test_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, dist='test', vocab_set=vocab_set, device=device,
                           return_type='corpus',
                           load_img_to_memory=False)
train_eval_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, dist='train', vocab_set=vocab_set, device=device,
                                 return_type='corpus',
                                 load_img_to_memory=False)
with open('vocab_set.pkl', 'wb') as f:
    pickle.dump(train_set.get_vocab(), f)
len(train_set), len(val_set), len(test_set)

# %%
vocab_size = len(vocab)
vocab_size, max_len

# %%

MODEL = "resnet50_monolstm"
EMBEDDING_DIM = 50
EMBEDDING = f"GLV{EMBEDDING_DIM}"
HIDDEN_SIZE = 256
BATCH_SIZE = 16
LR = 1e-2
MODEL_NAME = f'saved_models/{MODEL}_b{BATCH_SIZE}_emd{EMBEDDING}'
NUM_EPOCHS = 2
SAVE_FREQ = 2
LOG_INTERVAL = 25

run = wandb.init(project='image-captioning',
                 entity='datalab-buet',
                 name=f"{MODEL}_b{BATCH_SIZE}_emd{EMBEDDING}-{1}",
                 # tensorboard=True, sync_tensorboard=True,
                 config={"learning_rate": LR,
                         "epochs": NUM_EPOCHS,
                         "batch_size": BATCH_SIZE,
                         "model": MODEL,
                         "embedding": EMBEDDING,
                         "embedding_dim": EMBEDDING_DIM,
                         "hidden_size": HIDDEN_SIZE,
                         },
                 reinit=True)

# %%
embedding_matrix = embedding_matrix_creator(embedding_dim=EMBEDDING_DIM, word2idx=word2idx)
embedding_matrix.shape


# %%

def train_model(train_loader, model, loss_fn, optimizer, vocab_size, acc_fn, desc=''):
    running_acc = 0.0
    running_loss = 0.0
    model.train()
    t = tqdm(iter(train_loader), desc=f'{desc}')
    for batch_idx, batch in enumerate(t):
        images, captions, lengths = batch
        sort_ind = torch.argsort(lengths, descending=True)
        images = images[sort_ind]
        captions = captions[sort_ind]
        lengths = lengths[sort_ind]

        optimizer.zero_grad()
        # [sum_len, vocab_size]
        outputs = model(images, captions, lengths)
        # [b, max_len] -> [sum_len]
        targets = pack_padded_sequence(captions, lengths=lengths, batch_first=True, enforce_sorted=True)[0]

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        running_acc += (torch.argmax(outputs, dim=1) == targets).sum().float().item() / targets.size(0)
        running_loss += loss.item()
        t.set_postfix({'loss': running_loss / (batch_idx + 1),
                       'acc': running_acc / (batch_idx + 1),
                       }, refresh=True)
        if (batch_idx + 1) % LOG_INTERVAL == 0:
            print(f'{desc} {batch_idx + 1}/{len(train_loader)} '
                  f'train_loss: {running_loss / (batch_idx + 1):.4f} '
                  f'train_acc: {running_acc / (batch_idx + 1):.4f}')
            wandb.log({
                'train_loss': running_loss / (batch_idx + 1),
                'train_acc': running_acc / (batch_idx + 1),
            })

    return running_loss / len(train_loader)


def evaluate_model(data_loader, model, loss_fn, vocab_size, bleu_score_fn, tensor_to_word_fn, desc=''):
    running_bleu = [0.0] * 5
    model.eval()
    t = tqdm(iter(data_loader), desc=f'{desc}')
    for batch_idx, batch in enumerate(t):
        images, captions, lengths = batch
        outputs = tensor_to_word_fn(model.sample(images).cpu().numpy())

        for i in (1, 2, 3, 4):
            running_bleu[i] += bleu_score_fn(reference_corpus=captions, candidate_corpus=outputs, n=i)
        t.set_postfix({
            'bleu1': running_bleu[1] / (batch_idx + 1),
            'bleu4': running_bleu[4] / (batch_idx + 1),
        }, refresh=True)
    for i in (1, 2, 3, 4):
        running_bleu[i] /= len(data_loader)
    return running_bleu


# %%

from models.torch.densenet201_monolstm import Captioner

final_model = Captioner(EMBEDDING_DIM, HIDDEN_SIZE, vocab_size, num_layers=2,
                        embedding_matrix=embedding_matrix, train_embd=False).to(device)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=train_set.pad_value).to(device)
acc_fn = accuracy_fn(ignore_value=train_set.pad_value)
sentence_bleu_score_fn = bleu_score_fn(4, 'sentence')
corpus_bleu_score_fn = bleu_score_fn(4, 'corpus')
tensor_to_word_fn = words_from_tensors_fn(idx2word=idx2word)

params = list(final_model.decoder.parameters()) + list(final_model.encoder.embed.parameters()) + list(
    final_model.encoder.bn.parameters())

optimizer = torch.optim.Adam(params=params, lr=LR)

wandb.watch(final_model, log='all', log_freq=50)
wandb.watch(final_model.encoder, log='all', log_freq=50)
wandb.watch(final_model.decoder, log='all', log_freq=50)
wandb.save('vocab_set.pkl')

# %%
train_transformations = transforms.Compose([
    transforms.Resize(256),  # smaller edge of image resized to 256
    transforms.RandomCrop(224),  # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),  # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))
])
eval_transformations = transforms.Compose([
    transforms.Resize(256),  # smaller edge of image resized to 256
    transforms.CenterCrop(224),  # get 224x224 crop from random location
    transforms.ToTensor(),  # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))
])

train_set.transformations = train_transformations
val_set.transformations = eval_transformations
test_set.transformations = eval_transformations
train_eval_set.transformations = eval_transformations

# %%
eval_collate_fn = lambda batch: (torch.stack([x[0] for x in batch]), [x[1] for x in batch], [x[2] for x in batch])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, sampler=None, pin_memory=False)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, sampler=None, pin_memory=False,
                        collate_fn=eval_collate_fn)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, sampler=None, pin_memory=False,
                         collate_fn=eval_collate_fn)
train_eval_loader = DataLoader(train_eval_set, batch_size=BATCH_SIZE, shuffle=False, sampler=None, pin_memory=False,
                               collate_fn=eval_collate_fn)
# %%
train_loss_min = 100
val_bleu4_max = 0.0
for epoch in range(NUM_EPOCHS):
    train_loss = train_model(desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}', model=final_model,
                             optimizer=optimizer, loss_fn=loss_fn, acc_fn=acc_fn,
                             train_loader=train_loader, vocab_size=vocab_size)
    with torch.no_grad():
        train_bleu = evaluate_model(desc=f'\tTrain Bleu Score: ', model=final_model,
                                    loss_fn=loss_fn, bleu_score_fn=corpus_bleu_score_fn,
                                    tensor_to_word_fn=tensor_to_word_fn,
                                    data_loader=train_eval_loader, vocab_size=vocab_size)
        val_bleu = evaluate_model(desc=f'\tValidation Bleu Score: ', model=final_model,
                                  loss_fn=loss_fn, bleu_score_fn=corpus_bleu_score_fn,
                                  tensor_to_word_fn=tensor_to_word_fn,
                                  data_loader=val_loader, vocab_size=vocab_size)
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}',
              ''.join([f'train_bleu{i}: {train_bleu[i]:.4f} ' for i in (1, 4)]),
              ''.join([f'val_bleu{i}: {val_bleu[i]:.4f} ' for i in (1, 4)]),
              )
        wandb.log({f'val_bleu{i}': val_bleu[i] for i in (1, 2, 3, 4)})
        wandb.log({'train_bleu': train_bleu[4]})
        wandb.log({'val_bleu': val_bleu[4]})
        state = {
            'epoch': epoch + 1,
            'state_dict': final_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss_latest': train_loss,
            'val_bleu4_latest': val_bleu[4],
            'train_loss_min': min(train_loss, train_loss_min),
            'val_bleu4_max': max(val_bleu[4], val_bleu4_max),
            'train_bleus': train_bleu,
            'val_bleus': val_bleu,
        }
        torch.save(state, f'{MODEL_NAME}_latest.pt')
        wandb.save(f'{MODEL_NAME}_latest.pt')
        if train_loss < train_loss_min:
            train_loss_min = train_loss
            torch.save(state, f'{MODEL_NAME}''_best_train.pt')
            wandb.save(f'{MODEL_NAME}''_best_train.pt')
        if val_bleu[4] > val_bleu4_max:
            val_bleu4_max = val_bleu[4]
            torch.save(state, f'{MODEL_NAME}''_best_val.pt')
            wandb.save(f'{MODEL_NAME}''_best_val.pt')

torch.save(state, f'{MODEL_NAME}_ep{NUM_EPOCHS:02d}_weights.pt')
wandb.save(f'{MODEL_NAME}_ep{NUM_EPOCHS:02d}_weights.pt')
final_model.eval()

# %%
model = final_model

# %%
t_i = 1003
dset = train_set
im, cp, _ = dset[t_i]
print(''.join([idx2word[idx.item()] + ' ' for idx in model.sample(im.unsqueeze(0))[0]]))
print(dset.get_image_captions(t_i)[1])

plt.imshow(dset[t_i][0].detach().cpu().permute(1, 2, 0), interpolation="bicubic")

# %%
t_i = 500
dset = val_set
im, cp, _ = dset[t_i]
print(''.join([idx2word[idx.item()] + ' ' for idx in model.sample(im.unsqueeze(0))[0]]))
print(cp)

plt.imshow(dset[t_i][0].detach().cpu().permute(1, 2, 0), interpolation="bicubic")

# %%
t_i = 500
dset = test_set
im, cp, _ = dset[t_i]
print(''.join([idx2word[idx.item()] + ' ' for idx in model.sample(im.unsqueeze(0))[0]]))
print(cp)

plt.imshow(dset[t_i][0].detach().cpu().permute(1, 2, 0), interpolation="bicubic")

# %%
with torch.no_grad():
    model.eval()
    train_bleu = evaluate_model(desc=f'Train: ', model=final_model,
                                loss_fn=loss_fn, bleu_score_fn=corpus_bleu_score_fn,
                                tensor_to_word_fn=tensor_to_word_fn,
                                data_loader=train_eval_loader, vocab_size=vocab_size)
    val_bleu = evaluate_model(desc=f'Val: ', model=final_model,
                              loss_fn=loss_fn, bleu_score_fn=corpus_bleu_score_fn,
                              tensor_to_word_fn=tensor_to_word_fn,
                              data_loader=val_loader, vocab_size=vocab_size)
    test_bleu = evaluate_model(desc=f'Test: ', model=final_model,
                               loss_fn=loss_fn, bleu_score_fn=corpus_bleu_score_fn,
                               tensor_to_word_fn=tensor_to_word_fn,
                               data_loader=test_loader, vocab_size=vocab_size)
    for setname, result in zip(('train', 'val', 'test'), (train_bleu, val_bleu, test_bleu)):
        print(setname, end=' ')
        for ngram in (1, 2, 3, 4):
            print(f'Bleu-{ngram}: {result[ngram]}', end=' ')
            wandb.run.summary[f"{setname}_bleu{ngram}"] = result[ngram]
        print()
