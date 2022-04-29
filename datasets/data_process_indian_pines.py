from data_proces_utils import *
import scipy.io as sio

PATCH_SIZE = (13, 13)
SAMPLE_NUM_OF_EACH_CLASS = (30, 150, 150, 100, 150, 150, 20, 150, 15, 150, 150, 150, 150, 150, 50, 50)
DATASET_PATH = Path('./Indian_Pines').absolute()

data_path = DATASET_PATH / 'Indian_pines_corrected.mat'
labels_path = DATASET_PATH / 'Indian_pines_gt.mat'
np.random.seed(123)

data = sio.loadmat(data_path)["indian_pines_corrected"].transpose(2, 0, 1)
labels = sio.loadmat(labels_path)["indian_pines_gt"]

preprocess(data, labels, PATCH_SIZE, SAMPLE_NUM_OF_EACH_CLASS, save_path=DATASET_PATH)
