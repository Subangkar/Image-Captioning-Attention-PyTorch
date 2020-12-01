import json

# Load annotations file for the training images.
from collections import defaultdict

mscoco_train = json.load(open('data/annotations/train_captions.json'))
train_ids = [entry['id'] for entry in mscoco_train['images']][:1000]
train_id_to_file = {entry['id']: 'data/train2014/' + entry['file_name'] for entry in mscoco_train['images']}

# Extract out the captions for the training images
train_id_set = set(train_ids)
train_id_to_captions = defaultdict(list)
for entry in mscoco_train['annotations']:
    if entry['image_id'] in train_id_set:
        train_id_to_captions[entry['image_id']].append(entry['caption'])

# Load annotations file for the validation images.
mscoco_val = json.load(open('data/annotations/val_captions.json'))
val_ids = [entry['id'] for entry in mscoco_val['images']]
val_id_to_file = {entry['id']: 'data/val2014/' + entry['file_name'] for entry in mscoco_val['images']}

# Extract out the captions for the validation images
val_id_set = set(val_ids)
val_id_to_captions = defaultdict(list)
for entry in mscoco_val['annotations']:
    if entry['image_id'] in val_id_set:
        val_id_to_captions[entry['image_id']].append(entry['caption'])

# Load annotations file for the testing images
mscoco_test = json.load(open('data/annotations/test_captions.json'))
test_ids = [entry['id'] for entry in mscoco_test['images']]
test_id_to_file = {entry['id']: 'data/val2014/' + entry['file_name'] for entry in mscoco_test['images']}
