from collections import defaultdict
import re
import os.path, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from project_enums import DatasetChoice


class Flickr8kAudio:
    capt_path = 'Flickr8k_text/Flickr8k.token.txt'
    img_path = 'Flicker8k_Dataset/'
    dist_detail_path = {
        'train': 'Flickr8k_text/Flickr_8k.trainImages.txt',
        'valid': 'Flickr8k_text/Flickr_8k.devImages.txt',
        'test': 'Flickr8k_text/Flickr_8k.testImages.txt',
    }
    audio_path = 'flickr_audio/wavs/'

    def __init__(self, base_dataset_path='data/flickr8k/', dist_type=DatasetChoice.DEV.value):

        self.base_dataset_path = base_dataset_path
        ids_to_filter = None
        id_file = None
        print(DatasetChoice.TRAIN)
        if dist_type == DatasetChoice.TRAIN.value:
            id_file = base_dataset_path + Flickr8kAudio.dist_detail_path['train']
        elif dist_type == DatasetChoice.TEST.value:
            id_file = base_dataset_path + Flickr8kAudio.dist_detail_path['test']
        elif dist_type == DatasetChoice.DEV.value:
            id_file = base_dataset_path + Flickr8kAudio.dist_detail_path['valid']
        else:
            print('provide valid type')

        with open(id_file) as f:
            ids_to_filter = set([item.split('.')[0] for item in f.read().splitlines() if len(item) != 0])

        self.caption_count_per_image = 5

        self.dictionary = dict()
        self.buid_dict(token_path=self.base_dataset_path + Flickr8kAudio.capt_path)

        tmp_dict = dict()
        for id in ids_to_filter:
            tmp_dict[id] = self.dictionary[id]
        self.dictionary = tmp_dict

        self.ids = list(self.dictionary.keys())

    def buid_dict(self, token_path):
        with open(token_path, "r") as f:

            data = f.read()

        calc_img_path = self.base_dataset_path + Flickr8kAudio.img_path
        cal_wav_path = self.base_dataset_path + Flickr8kAudio.audio_path
        print(calc_img_path)
        descriptions = dict()

        try:
            for el in data.split("\n"):
                tokens = el.split()
                if len(tokens) < 2: continue
                image_id, image_desc = tokens[0], tokens[1:]

                # dropping .jpg from image id
                image_id = image_id.split(".")[0]
                image_desc.insert(0, 'startseq')
                image_desc.append('endseq')

                image_desc = " ".join(image_desc)
                image_desc = re.sub(r'[^\w\s]', '', image_desc)

                if image_id in descriptions:
                    descriptions[image_id]['descs'].append(image_desc)
                else:
                    descriptions[image_id] = dict()
                    descriptions[image_id]['descs'] = list()
                    descriptions[image_id]['descs'].append(image_desc)
                    descriptions[image_id]['im_path'] = calc_img_path + tokens[0].split("#")[0]
                    descriptions[image_id]['wavs'] = [cal_wav_path + image_id + '_' + str(cap_id) + '.wav' for cap_id in
                                                      range(self.caption_count_per_image)]


        except Exception as e:
            print("Exception got :- \n", e)

        self.dictionary = descriptions
        #         pickle.dump(self.dictionary,open("full_dictionary.pkl","wb"))
        return self.dictionary

    def __getitem__(self, index: int) -> (str, list, list):
        """
        :param index:
        :return: (img_file_path, list_of_captions, list_of_wav_files)
        """
        image_id = self.ids[index]
        return self.dictionary[image_id]['im_path'], self.dictionary[image_id]['descs'], self.dictionary[image_id][
            'wavs']

    def __len__(self):
        return len(self.ids)
