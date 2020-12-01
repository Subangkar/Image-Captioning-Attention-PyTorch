# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# %%
DATASET_BASE_PATH = 'data/coco2014/'
# %%
from datasets.mscoco2014 import CocoCaptions

val_set = CocoCaptions(dist_type='valid')
val_ids = val_set.ids
val_id_to_file = val_set.id_to_file
val_id_to_captions = val_set.id_to_captions
val_set[2]
len(val_set)
# %%
# len(val_id_set)
x = val_ids[0]

# display(val_id_to_file[x])
# %matplotlib inline

from IPython.display import Image
# Image(val_id_to_file[x])
from PIL import Image

image = Image.open(val_id_to_file[x])
image.show()
val_id_to_captions[x]

# %%
