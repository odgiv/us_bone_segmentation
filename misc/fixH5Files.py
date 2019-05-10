"""
With this script, read the given h5 file and cut us_vol from start_index to end_index at the end dimension
and transpose gt_vol to match it with us_vol then store again to the same h5 file. 

Example usage:
python fixH5Files.py -o H:\\14_02_2019_Ben_in_vivo\\12-59-16 -f H:\\14_02_2019_Ben_in_vivo\\12-59-16\\us_gt_vol.h5 -s 250 -e 350
"""

import argparse
import os
import h5py
import numpy as np
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", "-o", type=str, required=True)
parser.add_argument("--file_path", "-f", type=str, required=True, help="Images are stored in h5 file.")
parser.add_argument("--start_index", "-s", type=int, required=True)
parser.add_argument("--end_index", "-e", type=int, required=True)

args = parser.parse_args()

output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read h5 file
f = h5py.File(args.file_path, 'r')
us = f["us_vol"]
us = us[:, :, args.start_index: args.end_index]
gt = f["gt_vol"]
gt = np.transpose(gt, (1, 0, 2))

h5f = h5py.File(os.path.join(output_dir, "us_gt_vol_new.h5"), 'w')
h5f.create_dataset('us_vol', data=us)
h5f.create_dataset('gt_vol', data=gt)
h5f.close()

f.close()

print("Done")
