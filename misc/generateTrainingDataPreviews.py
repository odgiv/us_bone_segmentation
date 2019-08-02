"""
This script is used to generate input and ground truth images to different folders.
It is also possible to see overlapped images. 
Example usage: 
python misc/generateTrainingDataPreviews.py -f H:\\09_05_2019_Ben_in_vivo\\11-22-27\\us_gt_vol_new.h5 -p
"""
import argparse
import os
import h5py
import numpy as np
import cv2 as cv
from preprocessGt import preprocess_gt
from utils import img_and_mask_generator


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir_img", "-odi", type=str, default="H:\\14_02_2019_Ben_in_vivo_imgs\\overlapped_imgs_1")
parser.add_argument("--output_dir_gt", "-odg", type=str, default="H:\\14_02_2019_Ben_in_vivo_imgs\\gts")
parser.add_argument("--file_path", "-f", type=str, required=True, help="Images are stored in h5 file.")
parser.add_argument("--prefix", "-pre", type=str, default="14_02_2019_Ben")
parser.add_argument("--start_index", "-s", type=int, default=0)
parser.add_argument("--end_index", "-e", type=int, default=None)
parser.add_argument("--preprocess_gt", "-p", dest="preprocess_gt", default=False, action='store_true'),
parser.add_argument("--threshold_gt", "-t", dest="threshold_gt", default=False, action='store_true')

args = parser.parse_args()

def generate_original_and_overlayed_imgs(gt_volume, us_img_volume, start_index=0, end_index=None, is_threshold=True, is_clear_below_break_points_gt=True):
    step = 1

    print(us_img_volume.shape)
    print(gt_volume.shape)
    
    us_img_volume = np.transpose(us_img_volume, [2, 0, 1])
    us_img_volume = np.expand_dims(us_img_volume, -1)
    gt_volume = np.transpose(gt_volume, [2, 1, 0])
    gt_volume = np.expand_dims(gt_volume, -1)
    
    gen = img_and_mask_generator(us_img_volume, gt_volume, batch_size=1, shuffle=False)
    
    if start_index == 0 and end_index == None:
        count = gt_volume.shape[0]
    else:
        count = end_index - start_index

    # count = 100
    
    while step <= count: 
        bone_img, gt_img = next(gen)

        bone_img = bone_img * 255  # generator scales bone images.

        gt_img[gt_img>0] = 1
        gt_img[gt_img==0] = 0

        bone_img = np.uint8(bone_img)
        gt_img = np.uint8(gt_img)

        bone_img = np.squeeze(bone_img)
        gt_img = np.squeeze(gt_img)

        gt_img = preprocess_gt(bone_img, gt_img, is_threshold, is_clear_below_break_points_gt)        
        
        gt_img = gt_img * 255        

        overlapped_img = cv.addWeighted(bone_img, 1, gt_img, 0.2, 0)

        step+=1
        yield bone_img, gt_img, overlapped_img


output_dir_img = args.output_dir_img
output_dir_gt = args.output_dir_gt

if not os.path.exists(output_dir_img):
    os.makedirs(output_dir_img)

if not os.path.exists(output_dir_gt):
    os.makedirs(output_dir_gt)

# Read h5 file
f = h5py.File(args.file_path, 'r')
us = f["us_vol"]
gt = f["gt_vol"]


i = 0
for bone_img, gt_img, overlapped_img in generate_original_and_overlayed_imgs(gt, us, start_index=args.start_index, end_index=args.end_index, is_threshold=args.threshold_gt, is_clear_below_break_points_gt=args.preprocess_gt):
    dir_as_prefix = args.file_path.split("\\")[-2]
    
    # blank_img = np.zeros((bone_img.shape[0], bone_img.shape[1]*3), np.uint8)
    # blank_img[:, :bone_img.shape[1]] = bone_img        
    # blank_img[:, bone_img.shape[1]:bone_img.shape[1]*2] = gt_img        
    # blank_img[:, bone_img.shape[1]*2:] = overlapped_img
    cv.imwrite(os.path.join(output_dir_img, args.prefix + "_" + dir_as_prefix + "_" + str(i) + ".jpg"), overlapped_img)

    # cv.imwrite(os.path.join(output_dir_gt, args.prefix + "_" + dir_as_prefix + "_" + str(i) + ".jpg"), gt_img)
    # cv.imwrite(os.path.join(output_dir_img, args.prefix + "_" + dir_as_prefix + "_" + str(i) + ".jpg"), bone_img)
    i += 1

print("Done")