import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler

load_path1 = './Indian_pines_corrected.mat'
load_data = sio.loadmat(load_path1)
data_load = load_data["indian_pines_corrected"]
load_path2 = './Indian_pines_gt.mat'
load_labels = sio.loadmat(load_path2)
labels_load = load_labels["indian_pines_gt"]

np.save('./data_origin.npy', data_load)
np.save('./labels_origin.npy', labels_load)

data_normal = (data_load - data_load.min()) / (data_load.max() - data_load.min())
data_reshape = data_normal.reshape(-1, 200)
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data_reshape)
data_stand = data_standardized.reshape(145, 145, 200)

data_pad = np.pad(data_stand, ((6, 6), (6, 6), (0, 0)), mode="symmetric")
print("data_pad" + str(data_pad.shape))
print("labels_load" + str(labels_load.shape))
print("###########################################################")

data_list = []
labels_list = []
row_list = []
col_list = []

for i in range(145):
    for j in range(145):
        data_block = data_pad[i:i + 13, j:j + 13, ...]
        labels_block = labels_load[i, j, ...]
        if labels_block > 0:
            data_list.append(data_block)
            labels_list.append(labels_block)
            row_list.append(i)
            col_list.append(j)

data_all = np.array(data_list)

data_all = np.moveaxis(data_all, 3, 1)

labels_all = np.array(labels_list)
row = np.array(row_list)
col = np.array(col_list)
print("data_all" + str(data_all.shape))
print("labels_all" + str(labels_all.shape))
print("row" + str(row))
print("col" + str(col))
print("========================================================")

np.save('./data_all.npy', data_all.astype('float32'))
np.save('./labels_all.npy', labels_all - 1)
np.save('./row_all.npy', row)
np.save('./col_all.npy', col)

data_train_list = []
data_test_list = []
labels_train_list = []
labels_test_list = []
division_points = [0, 30, 150, 150, 100, 150, 150, 20, 150, 15, 150, 150, 150, 150, 150, 50, 50]

for c in range(1, 17):
    index = np.array(np.where(labels_all == c))
    index_block = np.squeeze(index)
    data_chose = data_all[index_block, ...]
    labels_chose = labels_all[index_block, ...]
    print("c" + str(c))
    print("data_chose" + str(data_chose.shape))
    print("labels_chose" + str(labels_chose.shape))

    aa = np.arange(0, data_chose.shape[0])
    np.random.shuffle(aa)
    data_shuffle = data_chose[aa, ...]
    labels_shuffle = labels_chose[aa, ...]
    print("data_shuffle" + str(data_shuffle.shape))
    print("labels_shuffle" + str(labels_shuffle.shape))

    mid = division_points[c]
    data_train_list.extend(data_shuffle[: mid, ...])
    data_test_list.extend(data_shuffle[mid:, ...])
    labels_train_list.extend(labels_shuffle[: mid, ...])
    labels_test_list.extend(labels_shuffle[mid:, ...])

data_train = np.array(data_train_list).astype("float32")
data_test = np.array(data_test_list).astype("float32")
labels_train_scalar = np.array(labels_train_list)
labels_test_scalar = np.array(labels_test_list)
labels_train = labels_train_scalar - 1
labels_test = labels_test_scalar - 1

print("data_train" + str(data_train.shape))
print("data_test" + str(data_test.shape))
print("labels_train" + str(labels_train.shape))
print("labels_test" + str(labels_test.shape))
print("*********************************************************")

np.save("./data_train.npy", data_train)
np.save("./data_test.npy", data_test)
np.save("./labels_train.npy", labels_train)
np.save("./labels_test.npy", labels_test)
