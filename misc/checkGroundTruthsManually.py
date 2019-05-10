from createGroundTruth import generate_gt_volume
import os
import numpy as np
import cv2 as cv

parent_directory = 'D:\\Data\\IPASM\\bone_data\\in_vivo_data\\14_02_2019_Ben\\' #'D:\\Data\\IPASM\\bone_data\\phantom_data\\2018_11_23_CAOS_record\\' 
ground_truth_mesh_filepath = parent_directory + 'ground_truth.stl'
model2bone_transformation_filepath = parent_directory + 'model2bone.txt'
us_img_data_filename = "vol_postProcessedImage_cropped.b8"
img2thigh_transformation_filename = "image2thighTransformationRefined.txt"

# Iterate over each sub directory which contains us data in parent_directory.
for f in sorted(os.listdir(parent_directory)):
    # If f is directory, not a file
    files_directory = os.path.join(parent_directory, f)
    if os.path.isdir(files_directory):
        print("Entering directory: {}".format(files_directory))

        if input("Skip this directory? (y/n): ") == "y":
            continue
        
        if not os.path.exists(os.path.join(files_directory, img2thigh_transformation_filename)):
            continue

        for start_index_in_z in range(0, 400, 50):
            end_index_in_z = start_index_in_z + 1
            print("start_index_in_z: {} and end_index_in_z: {}".format(
                start_index_in_z, end_index_in_z))
            gt_volume, us_img_volume = generate_gt_volume(files_directory=files_directory,
                                                          gt_mesh_filepath=ground_truth_mesh_filepath,
                                                          model2bone_transformation_filepath=model2bone_transformation_filepath,
                                                          us_img_data_filename=us_img_data_filename,
                                                          img2thigh_transformation_filename=img2thigh_transformation_filename,
                                                          start_end_indices_z_axis=(start_index_in_z, end_index_in_z))
            print("Shape of us_img_volume: {}".format(us_img_volume.shape))
            step = 1
            for z_index in range(0, gt_volume.shape[0], step):
                # bone_img = us_img_volume[:, :, z_index]
                bone_img = us_img_volume[:,start_index_in_z:end_index_in_z, :][:, z_index, :]

                # Read image from numpy array as greyscale
                # gt_img = np.uint8(gt_volume[:, :, z_index].T * 255)
                gt_img = np.uint8(np.transpose(gt_volume, (1, 2, 0))[:, :, z_index] * 255)

                overlapped_img = cv.addWeighted(bone_img, 1, gt_img, 0.2, 0)
                cv.imshow("Overlayed", overlapped_img)
                cv.imshow("Original", bone_img)

                print("Showing image {} / {} with step: {}".format(z_index +
                                                                   1, end_index_in_z - start_index_in_z, step))
                cv.waitKey(0)
                cv.destroyAllWindows()
