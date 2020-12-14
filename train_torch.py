# %%
from IPython.core.display import display
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from matplotlib import pyplot as plt

from utils_torch import *
from datasets.flickr8k import Flickr8kDataset

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device
# %%

DATASET_BASE_PATH = 'data/flickr8k/'

# %%

train_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, dist='train', device=device,
                            load_img_to_memory=False)
vocab, word2idx, idx2word, max_len = vocab_set = train_set.get_vocab()
val_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, dist='val', vocab_set=vocab_set, device=device)
test_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, dist='test', vocab_set=vocab_set, device=device)
len(train_set), len(val_set), len(test_set)

# %%
vocab_size = len(vocab)
vocab_size, max_len

# %%

samples_per_epoch = len(train_set)
samples_per_epoch


# %%

def train_model(train_loader, model, loss_fn, optimizer, vocab_size, acc_fn, desc=''):
    running_acc = 0.0
    running_loss = 0.0
    model.train()
    t = tqdm(iter(train_loader), desc=f'{desc}')
    for batch_idx, batch in enumerate(t):
        images, captions, lengths = batch

        optimizer.zero_grad()
        outputs = model(images, captions)

        loss = loss_fn(outputs.view(-1, vocab_size), captions.view(-1))
        loss.backward()
        optimizer.step()

        running_acc += acc_fn(torch.argmax(outputs.view(-1, vocab_size), dim=1), captions.view(-1))
        running_loss += loss.item()
        t.set_postfix({'loss': running_loss / (batch_idx + 1),
                       'acc': running_acc / (batch_idx + 1),
                       }, refresh=True)

    return running_loss / len(train_loader)


# %%

MODEL = "resnet50_monolstm"
EMBEDDING_DIM = 50
EMBEDDING = f"GLV{EMBEDDING_DIM}"
BATCH_SIZE = 16
LR = 1e-2
MODEL_NAME = f'saved_models/{MODEL}_b{BATCH_SIZE}_emd{EMBEDDING}'
NUM_EPOCHS = 2

# %%

from models.torch.resnet50_monolstm import Captioner

final_model = Captioner(EMBEDDING_DIM, 256, vocab_size, num_layers=2).to(device)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=train_set.pad_value).to(device)
acc_fn = accuracy_fn(ignore_value=train_set.pad_value)

# Specify the learnable parameters of the model
params = list(final_model.decoder.parameters()) + list(final_model.encoder.embed.parameters()) + list(
    final_model.encoder.bn.parameters())

# Define the optimizer
optimizer = torch.optim.Adam(params=params, lr=LR)

# %%
train_set.transformations = transforms.Compose([
    transforms.Resize(256),  # smaller edge of image resized to 256
    transforms.RandomCrop(224),  # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),  # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))
])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, sampler=None, pin_memory=False)
train_loss_min = 100
for epoch in range(NUM_EPOCHS):
    train_loss = train_model(desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}', model=final_model,
                             optimizer=optimizer, loss_fn=loss_fn, acc_fn=acc_fn,
                             train_loader=train_loader, vocab_size=vocab_size)
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

encoder = final_model.encoder
decoder = final_model.decoder

# %%
t_i = 1003
feat = encoder(train_set[t_i][0].unsqueeze(0))
print(''.join([idx2word[idx] + ' ' for idx in decoder.sample(feat.unsqueeze(1))]))
print(train_set.get_image_captions(t_i)[1])

plt.imshow(train_set[t_i][0].detach().cpu().permute(1, 2, 0), interpolation="bicubic")

# %%
eval_transformations = transforms.Compose([
    transforms.Resize(256),  # smaller edge of image resized to 256
    transforms.CenterCrop(224),  # get 224x224 crop from random location
    transforms.ToTensor(),  # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))
])

# %%
val_set.transformations = eval_transformations
t_i = 2020
feat = encoder(val_set[t_i][0].unsqueeze(0))

print(''.join([idx2word[idx] + ' ' for idx in decoder.sample(feat.unsqueeze(1))]))
print(val_set.get_image_captions(t_i)[1])

plt.imshow(val_set[t_i][0].detach().cpu().permute(1, 2, 0), interpolation="bicubic")

# %%
test_set.transformations = eval_transformations
t_i = 2020
feat = encoder(test_set[t_i][0].unsqueeze(0))

print(''.join([idx2word[idx] + ' ' for idx in decoder.sample(feat.unsqueeze(1))]))
print(test_set.get_image_captions(t_i)[1])

plt.imshow(test_set[t_i][0].detach().cpu().permute(1, 2, 0), interpolation="bicubic")

# %%
