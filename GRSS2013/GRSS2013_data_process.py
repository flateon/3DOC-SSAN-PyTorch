from pathlib import Path

import numpy as np
from PIL import Image
from osgeo import gdal
from osgeo.gdal_array import DatasetReadAsArray

PATCH_SIZE = (144, 13, 13)
PATCH_HALF = int((PATCH_SIZE[1] - 1) / 2)

data_path = Path('2013_IEEE_GRSS_DF_Contest_CASI.tif')
train_labels_path = Path('train_roi.tif')
val_labels_path = Path('val_roi.tif')

data_img = DatasetReadAsArray(gdal.Open(str(data_path)))
train_labels_img, val_labels_img = Image.open(train_labels_path), Image.open(val_labels_path)
train_labels, val_labels = np.array(train_labels_img), np.array(val_labels_img)


def normalize(data):
    return ((data - data.min()) / (data.max() - data.min())).astype(np.float32)


def standardize(data):
    return ((data - np.mean(data)) / np.std(data)).astype(np.float32)


def patch_cutting(data_img, label_img):
    data_img = np.pad(data_img, ((0, 0), (PATCH_HALF, PATCH_HALF), (PATCH_HALF, PATCH_HALF)))
    label_img = np.pad(label_img, ((PATCH_HALF, PATCH_HALF), (PATCH_HALF, PATCH_HALF)))
    # data_img = normalize(data_img)
    data_img = standardize(data_img)
    img_patch, labels = [], []
    for x, y in np.argwhere(label_img != 0):
        patch = data_img[:, x - PATCH_HALF:x + PATCH_HALF + 1, y - PATCH_HALF:y + PATCH_HALF + 1]
        img_patch.append(patch)
        labels.append(label_img[x, y])
    return np.stack(img_patch), np.stack(labels) - 1


train_patch, train_labels = patch_cutting(data_img, train_labels)
val_patch, val_labels = patch_cutting(data_img, val_labels)

# patch_all = np.concatenate((train_patch,val_patch))
# labels_all = np.concatenate((train_labels,val_labels))
# idxs = np.arange(len(labels_all))
# np.random.shuffle(idxs)
# train_patch, train_labels = patch_all[idxs[:2832]],labels_all[idxs[:2832]]
# val_patch,val_labels =  patch_all[idxs[2832:]],labels_all[idxs[2832:]]

np.save('train_data.npy', train_patch)
np.save('test_data.npy', val_patch)
np.save('train_labels.npy', train_labels)
np.save('test_labels.npy', val_labels)
