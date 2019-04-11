import configparser
import os

import h5py
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

config = configparser.RawConfigParser()
config.read_file(open(r'./configuration.txt'))


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)

def get_datasets(path, train, label, output_dir):
    # Load training images
    train_images = np.array(pickle.load(open(os.path.join(path, train), "rb")))
    train_images = np.transpose(train_images, (0,3,1,2))

    # Load image labels
    labels = np.array(pickle.load(open(os.path.join(path, label), "rb")))
    labels = labels[..., 0]
    labels[labels > 0] = 1

    # Split data:
    seed = int(config.get('settings', 'seed'))
    X, X_test, y, y_test = train_test_split(train_images, labels, test_size=0.1, random_state=seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Write data to disk:
    write_hdf5(X_train, os.path.join(path, "X_train.hdf5"))
    write_hdf5(y_train, os.path.join(path, "y_train.hdf5"))

    write_hdf5(X_valid, os.path.join(path, "X_valid.hdf5"))
    write_hdf5(y_valid, os.path.join(path, "y_valid.hdf5"))

    write_hdf5(X_test, os.path.join(path, "X_test.hdf5"))
    write_hdf5(y_test, os.path.join(path, "y_test.hdf5"))


if __name__ == '__main__':
    data_dir = config.get('data paths', 'data_dir')
    file_train = config.get('data paths', 'file_train')
    file_label = config.get('data paths', 'file_label')
    output_dir = config.get('data paths', 'output_dir')

    get_datasets(data_dir, file_train, file_label, output_dir)