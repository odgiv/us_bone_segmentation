import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from processUsData import read_us_data
from createGroundTruth import generate_gt_volume

parent_directory = 'H:\\2019_07_17_Ben\\' #'D:\\Data\\IPASM\\bone_data\\phantom_data\\2018_11_23_CAOS_record\\' 
ground_truth_mesh_filepath = parent_directory + 'ground_truth.stl'
model2bone_transformation_filepath = parent_directory + 'model2bone.txt'
us_img_data_filename = "vol_postProcessedImage_cropped.b8"
img2thigh_transformation_filename = "image2thighTransformationRefined.txt"
manual_transformation_filename = "volImg2GtMeshTransformation.txt"
slice_axis = 2 # There are from 0 to 2 axes in the 3d volume. 
skip_step = 50
start_at = 230

# Iterate over each sub directory which contains us data in parent_directory.
for f in sorted(os.listdir(parent_directory)):
    # If f is directory, not a file
    files_directory = os.path.join(parent_directory, f)
    if os.path.isdir(files_directory):
        print("Entering directory: {}".format(files_directory))

        if input("Skip this directory? (y/n): ") == "y":
            continue
        
        # if not os.path.exists(os.path.join(files_directory, img2thigh_transformation_filename)):
        #     continue

        ground_truth_mesh_filepath = os.path.join(parent_directory, files_directory, 'ground_truth_imSpace.stl')
        manual_transformation_filepath = os.path.join(parent_directory, files_directory, manual_transformation_filename)
        ultrasound_data = read_us_data(os.path.join(files_directory, us_img_data_filename))
        us_image_data = ultrasound_data['image_data']
        print("Shape of us_img_volume: {}".format(us_image_data.shape))
        for start_index in range(start_at, us_image_data.shape[slice_axis], skip_step):
            end_index = start_index + 1
            print("start_index_in_z: {} and end_index_in_z: {}".format(start_index, end_index))
            gt_volume = generate_gt_volume(files_directory=files_directory,
                                            gt_mesh_filepath=ground_truth_mesh_filepath,
                                            model2bone_transformation_filepath=model2bone_transformation_filepath,
                                            us_image_data=us_image_data,                                                
                                            img2thigh_transformation_filename=img2thigh_transformation_filename,
                                            manual_transformation_filepath=manual_transformation_filepath,
                                            slice_axis=slice_axis,
                                            start_end_indices=(start_index, end_index))
            
            step = 1
            if slice_axis == 1:
                size = gt_volume.shape[0]
            elif slice_axis == 0:
                size = gt_volume.shape[1]
            else:
                size = gt_volume.shape[2]
            for z_index in range(0, size, step):
                # bone_img = us_img_volume[:, :, z_index]
                if slice_axis == 0:
                    bone_img = us_image_data[start_index:end_index, :, :][z_index, :, :]                
                    gt_img = np.uint8(np.transpose(gt_volume, (0, 2, 1))[:, :, z_index] * 255)
                elif slice_axis == 1:
                    bone_img = us_image_data[:,start_index:end_index, :][:, z_index, :]
                    gt_img = np.uint8(np.transpose(gt_volume, (1, 2, 0))[:, :, z_index] * 255)
                else:
                    bone_img = us_image_data[:, :, start_index:end_index][:, :, z_index]
                    gt_img = np.uint8(np.transpose(gt_volume, (1, 0, 2))[:, :, z_index] * 255)
                

                # Read image from numpy array as greyscale
                # gt_img = np.uint8(gt_volume[:, :, z_index].T * 255)

                overlapped_img = cv.addWeighted(bone_img, slice_axis, gt_img, 0.2, 0)
                cv.imshow("Overlayed", overlapped_img)
                cv.imshow("Original", bone_img)

                print("Showing image {} / {} with step: {}".format(z_index + 1, end_index - start_index, step))
                cv.waitKey(0)
                cv.destroyAllWindows()
