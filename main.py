# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# %%
DATASET_BASE_PATH = 'data/coco2014/'
# %%
from datasets.mscoco2014 import CocoCaptions
from datasets.flickr8kaudio import Flickr8kAudio

val_set = Flickr8kAudio(dist_type='valid')
# val_ids = val_set.ids
# val_id_to_file = val_set.id_to_file
# val_id_to_captions = val_set.id_to_captions
val_set[2]
len(val_set)
# %%
# len(val_id_set)
# x = val_ids[8091]

# display(val_id_to_file[x])
# %matplotlib inline

from IPython.display import Image, Audio
# Image(val_id_to_file[x])
from PIL import Image

image = Image.open(val_set[2][0])
image.show()
val_set[2][2]

Audio(val_set[2][2][0])
# %%
