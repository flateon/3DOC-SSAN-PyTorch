from data_proces_utils import *
import scipy.io as sio

PATCH_SIZE = (13, 13)
SAMPLE_NUM_OF_EACH_CLASS = (548, 540, 392, 542, 256, 532, 375, 514, 231)
DATASET_PATH = Path('./Pavia_University').absolute()

data_path = DATASET_PATH / 'PaviaU.mat'
labels_path = DATASET_PATH / 'PaviaU_gt.mat'
np.random.seed(123)

data = sio.loadmat(data_path)["paviaU"].transpose(2, 0, 1)
labels = sio.loadmat(labels_path)["paviaU_gt"]

preprocess(data, labels, PATCH_SIZE, SAMPLE_NUM_OF_EACH_CLASS, save_path=DATASET_PATH)
