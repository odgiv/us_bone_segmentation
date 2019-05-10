import os
import random
import json
import h5py
import numpy as np
from scipy.ndimage import binary_closing, binary_opening
from skimage.draw import line

#TODO: move these arguments to params.json
MIN_IMG_SIZE = (465, 381)#(266, 369)
NUM_ROWS_CUT_BOTTOM = 0 #33


class DataLoader():

    def __init__(self, params):
        self.train_datasets_path = params["train_datasets_path"]
        self.valid_datasets_path = os.path.join(self.train_datasets_path, params["valid_datasets_folder"]) 
        self.test_datasets_path = os.path.join(self.train_datasets_path, params["test_datasets_folder"])

        print("train valid datasets path: {}, test dataset path: {}".format(self.train_val_datasets_path, self.test_datasets_path))

    def _load_XY_from(self, path):
        list_us_array = []
        list_gt_array = []

        for f in sorted(os.listdir(path)):
            # If f is directory, not a file
            f_full_path = os.path.join(path, f)
            if os.path.isdir(f_full_path):                
                if f_full_path != self.test_datasets_path and f_full_path != self.valid_datasets_path:
                    print("entering directory: ", f_full_path)

                    h5f = h5py.File(os.path.join(f_full_path, 'us_gt_vol.h5'), 'r')
                else:
                    continue
            else:
                print("using a file: ", f_full_path)
                h5f = h5py.File(f_full_path, 'r')

            us_vol = h5f['us_vol'][:]
            gt_vol = h5f['gt_vol'][:]
            print("Shape of us_vol: ", us_vol.shape)
            print("Shape of gt_vol: ", gt_vol.shape)
            
            #gt_vol = np.transpose(gt_vol, (1, 0, 2))

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

    def loadValidDatasets(self):
        X, Y = self._load_XY_from(self.valid_datasets_path)
        return X, Y

    def loadTrainDatasets(self):
        X, Y = self._load_XY_from(self.train_datasets_path)
        return X, Y