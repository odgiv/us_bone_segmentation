import os
import random
import json
import h5py
import numpy as np

MIN_IMG_SIZE = (266, 369)
NUM_ROWS_CUT_BOTTOM = 33


class DataLoader():

    def __init__(self):
        try:
            with open("./params.json") as f:
                params = json.load(f)
                self.train_val_datasets_path = params["train_val_datasets_path"]
                self.test_datasets_path = params["test_datasets_path"]
        except FileNotFoundError:
            print("params.json file doesn't exist for DataLoader.")
            exit()

    def _load_XY_from(self, path):
        list_us_array = []
        list_gt_array = []

        for f in sorted(os.listdir(path)):
            # If f is directory, not a file
            files_directory = os.path.join(path, f)
            if not os.path.isdir(files_directory):
                continue
            print("entering directory: ", files_directory)
            h5f = h5py.File(os.path.join(files_directory, 'us_gt_vol.h5'), 'r')
            us_vol = h5f['us_vol'][:]
            gt_vol = h5f['gt_vol'][:]
            gt_vol = np.transpose(gt_vol, (1, 0, 2))

            cut_at_ax0 = 0
            cut_at_ax1 = 0
            # To check maximum num of consecutive all 0.0 rows from bottom.
            # for i in range(us_vol.shape[-1]):
            #     sli = us_vol[:, :, i]
            #     num_zero_bottom_rows = 0
            #     for j in range(sli.shape[0]-1, 0, -1):
            #         row = sli[j, :]
            #         if np.all(row == 0.0):
            #             num_zero_bottom_rows += 1
            #         else:
            #             break

            #     if max_num_zero_bottom_rows < num_zero_bottom_rows:
            #         max_num_zero_bottom_rows = num_zero_bottom_rows
            # print(max_num_zero_bottom_rows)

            if us_vol.shape[0] > MIN_IMG_SIZE[0]:
                cut_at_ax0 = random.randrange(
                    0, (us_vol.shape[0] - MIN_IMG_SIZE[0]), 1)

            if us_vol.shape[1] > MIN_IMG_SIZE[1]:
                cut_at_ax1 = random.randrange(
                    0, (us_vol.shape[1] - MIN_IMG_SIZE[1]), 1)

            us_vol = us_vol[cut_at_ax0:cut_at_ax0 +
                            MIN_IMG_SIZE[0] - NUM_ROWS_CUT_BOTTOM, cut_at_ax1:cut_at_ax1 + MIN_IMG_SIZE[1], :]
            gt_vol = gt_vol[cut_at_ax0:cut_at_ax0 +
                            MIN_IMG_SIZE[0] - NUM_ROWS_CUT_BOTTOM, cut_at_ax1:cut_at_ax1 + MIN_IMG_SIZE[1], :]

            list_us_array.append(us_vol)
            list_gt_array.append(gt_vol)

        X = np.dstack(list_us_array)
        Y = np.dstack(list_gt_array)

        X = np.transpose(X, (2, 0, 1))
        Y = np.transpose(Y, (2, 0, 1))

        X = np.expand_dims(X, -1)
        Y = np.expand_dims(Y, -1)

        np.random.seed(1)
        np.random.shuffle(X)
        np.random.seed(1)
        np.random.shuffle(Y)

        return X, Y

    def loadTestDatasets(self):
        X, Y = self._load_XY_from(self.test_datasets_path)
        return X, Y

    def loadTrainValDatasets(self, val_ratio=0.8):
        X, Y = self._load_XY_from(self.train_val_datasets_path)

        X_train, Y_train = np.array([]), np.array([])
        X_valid, Y_valid = np.array([]), np.array([])

        X_train = X[0:int(X.shape[0]*val_ratio), :, :]
        Y_train = Y[0:int(Y.shape[0]*val_ratio), :, :]
        X_valid = X[int(X.shape[0]*val_ratio):, :, :]
        Y_valid = Y[int(Y.shape[0]*val_ratio):, :, :]

        return X_train, Y_train, X_valid, Y_valid

if __name__ == "__main__":
    loader = DataLoader()
    X_train, Y_train, X_val, Y_val = loader.loadTrainValDatasets()
    assert(X_train.shape == Y_train.shape)
    assert(X_val.shape == Y_val.shape)

    X_test, Y_test = loader.loadTestDatasets()
    assert(X_test.shape == Y_test.shape)
