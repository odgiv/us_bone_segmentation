from scipy.ndimage import binary_closing, binary_opening
from skimage.draw import line
import numpy as np
import cv2 as cv
import argparse
import h5py
import os

def remove_non_consequetive_white_pixels(gt):
    """
    Remove all surfaces that appear below the first surface in gt. First, we find connected components using cv.
    If there are more than one component, we find center for each of component after finding contour.
    We keep the component on the top by comparing X coordinate of center coordinates. 
    """
    ret, labels = cv.connectedComponentsWithStats(gt)
    if ret <= 2:
        return gt
    top_r = 0
    top_r_y = 0
    for r in range(1, ret): # 0 for background 
        new_label = np.array(labels)
        
        # order of the next 2 lines is important
        new_label[labels != r] = 0
        new_label[labels == r] = 255
        # print((new_label == 255).sum())
        new_label = np.expand_dims(new_label, 2)
        new_label = np.uint8(new_label)
                
        contours, hierarchy = cv.findContours(new_label , cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        # for j in range(len(contours)):
        if len(contours) == 1:
            c = contours[0]        
            M = cv.moments(c)
            
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            
            if (top_r_y > cY or top_r_y == 0) and cX != 0 and cY != 0:
                top_r_y = cY
                top_r = r 
    
    if top_r != 0:
        gt[labels != top_r] = 0
        gt[labels == top_r] = 1

    return gt

def threshold_gt(img, gt, threshold_val=0):
    """
    First, we have to threshold input image 
    """

    assert(img.shape == gt.shape)

    # img_thresholded = img > threshold_val
    
    all_zero_rows = np.where(~img.any(axis=1))[0]
    all_zero_cols = np.where(~img.any(axis=0))[0]

    for r in all_zero_rows:
        gt[r,:] = 0

    for c in all_zero_cols:
        gt[:,c] = 0

    # gt[~img_thresholded] = 0
    # gt = binary_closing(gt, structure=np.ones((10, 10))).astype(np.uint8)
    # gt = binary_opening(gt, structure=np.ones((4, 4))).astype(np.uint8)
    
    return gt

def find_breaking_points_from_top_axis2(img, gt):
    center_in_x = round(gt.shape[1] / 2)

    left_most_point_from_top = None
    right_most_point_from_top = None

    for i in range(gt.shape[0]):
        l_rr_top, l_cc_top = line(0, center_in_x, i, 0)
        r_rr_top, r_cc_top = line(0, center_in_x, i, gt.shape[1] - 1)
        
        if left_most_point_from_top == None:
            for (lr, lc) in zip(l_rr_top, l_cc_top):
                if left_most_point_from_top == None and gt[lr, lc] == 1:
                    left_most_point_from_top = (lr, lc)
                    break

        if right_most_point_from_top == None:
            for (rr, rc) in zip(r_rr_top, r_cc_top):
                if right_most_point_from_top == None and gt[rr, rc] == 1:
                    right_most_point_from_top = (rr, rc)
                    break
        
        if left_most_point_from_top and right_most_point_from_top:
            break
    
    
    if left_most_point_from_top == None:
        for i in range(round(gt.shape[1] / 2)):
            l_rr_top, l_cc_top = line(0, center_in_x, gt.shape[0] - 1, i)
            for (lr, lc) in zip(l_rr_top, l_cc_top):
                if left_most_point_from_top == None and gt[lr, lc] == 1:
                    left_most_point_from_top = (lr, lc)
                    break

            if left_most_point_from_top:
                break
    
    if right_most_point_from_top == None:
        for i in range(gt.shape[1] - 1, round(gt.shape[1] / 2), -1):
            r_rr_top, r_cc_top = line(0, center_in_x, gt.shape[0] - 1, i)
            for (rr, rc) in zip(r_rr_top, r_cc_top):
                if right_most_point_from_top == None and gt[rr, rc] == 1:
                    right_most_point_from_top = (rr, rc)
                    break
            
            if right_most_point_from_top:
                break

    return left_most_point_from_top, right_most_point_from_top

def find_breaking_points_from_top_axis1(img, gt):
    center_in_x = round(gt.shape[1] / 2)

    left_most_point_from_top = None
    right_most_point_from_top = None

    left_boundary_rr, left_boundary_cc = line(0, 105, gt.shape[0] - 1, 0)
    right_boundary_rr, right_boundary_cc = line(0, 447, gt.shape[0] - 1, gt.shape[1] - 1)

    for (lb_r, lb_c) in zip(left_boundary_rr, left_boundary_cc):
            gt[:lb_r, :lb_c] = 0
    
    for (rb_r, rb_c) in zip(right_boundary_rr, right_boundary_cc):
            gt[:rb_r, rb_c:] = 0
            

    for left_boundary_point in zip(left_boundary_rr, left_boundary_cc):
        l_rr_top, l_cc_top = line(0, center_in_x, *left_boundary_point)
        # r_rr_top, r_cc_top = line(0, center_in_x, i, gt.shape[1] - 1)
        
        if left_most_point_from_top == None:
            for (lr, lc) in zip(l_rr_top, l_cc_top):
                if left_most_point_from_top == None and gt[lr, lc] == 1:
                    left_most_point_from_top = (lr, lc)
                    break
                    
        if left_most_point_from_top:
            break
    
    for righ_boundary_point in zip(right_boundary_rr, right_boundary_cc):
        r_rr_top, r_cc_top = line(0, center_in_x, *righ_boundary_point)

        if right_most_point_from_top == None:
            for (rr, rc) in zip(r_rr_top, r_cc_top):
                if right_most_point_from_top == None and gt[rr, rc] == 1:
                    right_most_point_from_top = (rr, rc)
                    break

        if right_most_point_from_top:
            break
    
    if left_most_point_from_top == None:
        for i in range(round(gt.shape[1] / 2)):
            l_rr_top, l_cc_top = line(0, center_in_x, gt.shape[0] - 1, i)
            for (lr, lc) in zip(l_rr_top, l_cc_top):
                if left_most_point_from_top == None and gt[lr, lc] == 1:
                    left_most_point_from_top = (lr, lc)
                    break

            if left_most_point_from_top:
                break
    
    if right_most_point_from_top == None:
        for i in range(gt.shape[1] - 1, round(gt.shape[1] / 2), -1):
            r_rr_top, r_cc_top = line(0, center_in_x, gt.shape[0] - 1, i)
            for (rr, rc) in zip(r_rr_top, r_cc_top):
                if right_most_point_from_top == None and gt[rr, rc] == 1:
                    right_most_point_from_top = (rr, rc)
                    break
            
            if right_most_point_from_top:
                break

    return left_most_point_from_top, right_most_point_from_top

def clear_gt_below_breaking_point_from_top(img, gt, axis):    

    if axis == 1:        
        left_most_point_from_top, right_most_point_from_top = find_breaking_points_from_top_axis1(img, gt)    
    elif axis == 2:
        left_most_point_from_top, right_most_point_from_top = find_breaking_points_from_top_axis2(img, gt)    

    if left_most_point_from_top and right_most_point_from_top:
        lb_rr, lb_cc = line(left_most_point_from_top[0], left_most_point_from_top[1], gt.shape[0] - 1, round(gt.shape[1] / 2))
        rb_rr, rb_cc = line(right_most_point_from_top[0], right_most_point_from_top[1], gt.shape[0] - 1, round(gt.shape[1] / 2))
        
        for (lb_r, lb_c) in zip(lb_rr, lb_cc):
            gt[lb_r:, :lb_c] = 0

        for (rb_r, rb_c) in zip(rb_rr, rb_cc):
            gt[rb_r:, rb_c:] = 0
    
       
    return gt

def preprocess_gt(img, gt_img, is_threshold, is_clear_below_bps, axis):
    if is_threshold:
        gt_img = threshold_gt(img, gt_img)

    if is_clear_below_bps:
        gt_img = clear_gt_below_breaking_point_from_top(img, gt_img, axis)

    gt_img = remove_non_consequetive_white_pixels(gt_img)
    
    return gt_img
