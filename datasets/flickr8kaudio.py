from collections import defaultdict


class Flickr8kAudio:
    capt_path = 'Flickr8k_text/Flickr8k.token.txt'
    img_path = 'Flicker8k_Dataset/'
    dist_detail_path = {
        'train': 'Flickr8k_text/Flickr_8k.trainImages.txt',
        'valid': 'Flickr8k_text/Flickr_8k.devImages.txt',
        'test': 'Flickr8k_text/Flickr_8k.testImages.txt',
    }
    audio_path = 'flickr_audio/wavs/'

    def __init__(self, base_dataset_path='data/flickr8k/', dist_type='valid'):
        img_id_cap_map = self.image_to_caption_dict(image_path=base_dataset_path + Flickr8kAudio.img_path,
                                                    captn_path=base_dataset_path + Flickr8kAudio.capt_path)
        ids = set(img_id_cap_map.keys())
        id_to_file = defaultdict()
        for i, (img_id, capt) in enumerate(img_id_cap_map.items()):
            # id_to_file.append(base_dataset_path + Flickr8kAudio.img_path + img_id + '.jpg')
            id_to_file[img_id] = base_dataset_path + Flickr8kAudio.img_path + img_id + '.jpg'

        self.ids = list(ids)
        self.id_to_file = id_to_file
        self.id_to_captions = img_id_cap_map

    def image_to_caption_dict(self, image_path, captn_path):
        with open(captn_path) as f:
            data = f.read()

        descriptions = dict()

        for el in data.strip().split("\n"):

            tokens = el.split()
            image_id, image_desc = tokens[0], tokens[1:]

            # dropping .jpg from image id
            image_id = image_id.split(".")[0]

            image_desc = " ".join(image_desc)

            if image_id in descriptions:
                descriptions[image_id].append(image_desc)
            else:
                descriptions[image_id] = [image_desc]

        return descriptions

    def __getitem__(self, index: int) -> (str, list):
        """
        :param index:
        :return: (img_file_path, list_of_captions)
        """
        return self.id_to_file[self.ids[index]], self.id_to_captions[self.ids[index]]

    def __len__(self):
        return len(self.ids)
