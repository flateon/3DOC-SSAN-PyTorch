from data_proces_utils import *
from PIL import Image
from osgeo import gdal
from osgeo.gdal_array import DatasetReadAsArray

PATCH_SIZE = (13, 13)
SAMPLE_NUM_OF_EACH_CLASS = (198, 190, 192, 188, 186, 182, 196, 191, 193, 191, 181, 192, 184, 181, 187)
DATASET_PATH = Path('./Houston').absolute()

data_path = DATASET_PATH / '2013_IEEE_GRSS_DF_Contest_CASI.tif'
train_labels_path = DATASET_PATH / 'train_roi.tif'
val_labels_path = DATASET_PATH / 'val_roi.tif'
np.random.seed(123)

data = DatasetReadAsArray(gdal.Open(str(data_path)))
train_labels, val_labels = np.asarray(Image.open(train_labels_path)), np.asarray(Image.open(val_labels_path))
labels = train_labels + val_labels

preprocess(data, labels, PATCH_SIZE, SAMPLE_NUM_OF_EACH_CLASS, save_path=DATASET_PATH)
