import numpy as np
from pathlib import Path


def standardize(data):
    data = data.transpose(1, 2, 0)
    data = ((data - np.mean(data, axis=(0, 1))) / np.std(data, axis=(0, 1))).astype(np.float32)
    return data.transpose(2, 0, 1)


def patch_cutting(data_img, label_img, patch_size=(9, 9)):
    patch_half_h = int((patch_size[0] - 1) / 2)
    patch_half_w = int((patch_size[1] - 1) / 2)
    data_img = np.pad(data_img, ((0, 0), (patch_half_h, patch_half_h), (patch_half_w, patch_half_w)), mode="symmetric")
    img_patch, labels = [], []
    for x, y in np.argwhere(label_img != 0):
        patch = data_img[:, x:x + patch_size[0], y:y + patch_size[1]]
        img_patch.append(patch)
        labels.append(label_img[x, y])
    return np.stack(img_patch), np.stack(labels) - 1, np.argwhere(label_img != 0)


def shuffle(patch_all, labels_all, position):
    """Shuffle dataset"""
    idxs = np.arange(len(labels_all))
    np.random.shuffle(idxs)
    return patch_all[idxs], labels_all[idxs], position[idxs]


def pick_samples(patch_all, labels_all, position, num_samples_each_classes, save_path):
    # Pick a given number of train samples
    train_idx_all, val_idx_all = [], []
    for class_idx, sample_num in enumerate(num_samples_each_classes):
        idx = np.argwhere(labels_all == class_idx)
        train_idx, val_idx = idx[:sample_num], idx[sample_num:]
        train_idx_all.append(train_idx)
        val_idx_all.append(val_idx)
    train_idx_all, val_idx_all = np.concatenate(train_idx_all)[:, 0], np.concatenate(val_idx_all)[:, 0]
    train_patch, val_patch = patch_all[train_idx_all], patch_all[val_idx_all]
    train_labels, val_labels = labels_all[train_idx_all], labels_all[val_idx_all]
    train_position, val_position = position[train_idx_all], position[val_idx_all]

    # NCHW
    np.save(save_path / 'data_train.npy', train_patch)
    np.save(save_path / 'data_test.npy', val_patch)
    np.save(save_path / 'labels_train.npy', train_labels)
    np.save(save_path / 'labels_test.npy', val_labels)
    np.save(save_path / 'position_train.npy', train_position)
    np.save(save_path / 'position_test.npy', val_position)


def preprocess(data, labels, patch_size, num_samples_each_classes, save_path=Path('./')):
    """data: CHW"""
    np.save(save_path / 'data_origin.npy', data)
    np.save(save_path / 'labels_origin.npy', labels)
    data = standardize(data)
    patch_all, labels_all, position = patch_cutting(data, labels, patch_size)
    patch_all, labels_all, position = shuffle(patch_all, labels_all, position)
    pick_samples(patch_all, labels_all, position, num_samples_each_classes, save_path)
