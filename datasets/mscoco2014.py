import json
from collections import defaultdict


class CocoCaptions:
    annot_path = {
        'train': 'annotations/captions_train2014.json',
        'valid': 'annotations/captions_val2014.json',
        'test': 'annotations/image_info_test2014.json',
    }

    img_path = {
        'train': 'train2014/',
        'valid': 'val2014/',
        'test': 'test2014/',
    }

    def __init__(self, base_dataset_path='data/coco2014/', dist_type='valid'):

        annotation_path = CocoCaptions.annot_path[dist_type]
        images_path = CocoCaptions.img_path[dist_type]

        # Load annotations file for the images.
        distrb = json.load(open(base_dataset_path + annotation_path))
        ids = [entry['id'] for entry in distrb['images']]
        id_to_file = {entry['id']: base_dataset_path + images_path + entry['file_name'] for entry in
                      distrb['images']}

        # Extract out the captions for the images
        id_set = set(ids)
        id_to_captions = defaultdict(list)
        for entry in distrb['annotations']:
            if entry['image_id'] in id_set:
                id_to_captions[entry['image_id']].append(entry['caption'])

        self.ids = list(ids)
        self.id_to_file = id_to_file
        self.id_to_captions = id_to_captions

    def __getitem__(self, index: int) -> (str, list):
        """
        :param index:
        :return: (img_file_path, list_of_captions)
        """
        return self.id_to_file[self.ids[index]], self.id_to_captions[self.ids[index]]

    def __len__(self):
        return len(self.ids)
