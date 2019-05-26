"""
This script is used to generate images that show raw input image, ground truth image and overlapped image side by side.
Example usage: 
python generateTrainingDataPreviews.py -o H:\\14_02_2019_Ben_in_vivo\\preview\\13-02-41 -f H:\\14_02_2019_Ben_in_vivo\\13-02-41\\us_gt_vol.h5 -s 100 -e 350 -p -t
"""
import argparse
import os
import h5py
import numpy as np
import cv2 as cv
from preprocessGt import preprocess_gt
from utils import preprocess, batch_img_generator


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", "-o", type=str, required=True)
parser.add_argument("--file_path", "-f", type=str, required=True, help="Images are stored in h5 file.")
parser.add_argument("--start_index", "-s", type=int, default=0)
parser.add_argument("--end_index", "-e", type=int, default=-1)
parser.add_argument("--preprocess_gt", "-p", dest="preprocess_gt", default=True, action='store_false'),
parser.add_argument("--threshold_gt", "-t", dest="threshold_gt", default=True, action='store_false')

args = parser.parse_args()


def generate_original_and_overlayed_imgs(gt_volume, us_img_volume, start_index=0, end_index=-1, is_threshold=True, is_clear_below_break_points_gt=True):
    step = 1
    us_img_volume = np.transpose(us_img_volume, [2, 0, 1])
    # us_img_volume = np.expand_dims(us_img_volume, -1)
    gt_volume = np.transpose(gt_volume, [2, 0, 1])
    # gt_volume = np.expand_dims(gt_volume, -1)
    
    gen = batch_img_generator(us_img_volume, gt_volume)
    
    for bone_img, gt_img, _ in gen:
    
    # while step < 100: 
        # bone_img, gt_img = next(gen)
        
        # bone_img = us_img_volume[index, :, :]

        # gt_img = preprocess_gt(bone_img, gt_img, is_threshold, is_clear_below_break_points_gt)

        # bone_img, gt_img = preprocess(bone_img, gt_img)
        bone_img = np.uint8(bone_img)
        gt_img = np.uint8(gt_img)

        bone_img = np.squeeze(bone_img)
        gt_img = np.squeeze(gt_img)

        gt_img = gt_img * 255        

        overlapped_img = cv.addWeighted(bone_img, 1, gt_img, 0.2, 0)

        step+=1
        yield bone_img, gt_img, overlapped_img


output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read h5 file
f = h5py.File(args.file_path, 'r')
us = f["us_vol"]
gt = f["gt_vol"]


i = 0
for bone_img, gt_img, overlapped_img in generate_original_and_overlayed_imgs(gt, us, start_index=args.start_index, end_index=args.end_index, is_threshold=args.threshold_gt, is_clear_below_break_points_gt=args.preprocess_gt):

    blank_img = np.zeros((bone_img.shape[0], bone_img.shape[1]*3), np.uint8)
    blank_img[:, :bone_img.shape[1]] = bone_img        
    blank_img[:, bone_img.shape[1]:bone_img.shape[1]*2] = gt_img        
    blank_img[:, bone_img.shape[1]*2:] = overlapped_img
    
    cv.imwrite(os.path.join(output_dir, str(i) + ".png"), blank_img)
    i += 1

print("Done")