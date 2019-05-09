"""
This script is used to create ground truths of ultrasound bone image volumes by calculating 
distance between points in the ground truth and an image volume. 

- Load gt (ground truth) mesh file. Get vertices of the mesh.
- Transform it to bone space by using model_to_bone_transformation matrix. (in model2bone txt file)
- Then transform the result again to image space by using inverse of img_to_thigh_transformation matrix. 
  As result, we will have the ground truth mesh points transformed to image space.
- Read ultrasound volume image data.
- Build 3D space from every location of the image volume. 
  For example: [[0 0 0],[0 0 1],[0 0 2],..., [X Y Z]] where X,Y,Z are dimensions of the image volume.
- Since our ground truth mesh is now in the image space, we can determine for each points of image volume,
  whether it is ground truth (bone surface) or not by simply calculating distance to the nearest point of 
  ground truth mesh. We are using KDTree algorithm for the search and max distance bound must be 
  specified.
"""

import numpy as np
from matplotlib import pyplot
from mpl_toolkits import mplot3d
from scipy import spatial
from PIL import Image
from stl import mesh
import time
import os
import argparse
import h5py
from processUsData import read_us_data

DISTANCE_UPPPER_BOUND = 7.0


def transform_ground_truth_model_to_image_space(ground_truth_mesh, model_to_bone_transformation, img_to_thigh_transformation):
    """
    Summary    
    -------

    Parameters
    ----------        

    Returns
    -------

    """
    # Reshape to Mx3
    mesh_vertices = ground_truth_mesh.vectors.reshape(-1, 3)

    # Append column of 1s at the end to make it Mx4
    mesh_vertices = np.append(mesh_vertices, np.ones(
        (mesh_vertices.shape[0], 1), dtype=np.float), axis=1)

    # Transform model to bone space
    mesh_vertices_in_bone_space = np.matmul(mesh_vertices, model_to_bone_transformation)

    # Multiply by inverse of img to thigh transformation
    mesh_vertices_in_img_space = np.matmul(mesh_vertices_in_bone_space, np.linalg.inv(img_to_thigh_transformation))

    # Remove the last column
    mesh_vertices_in_img_space = np.delete(mesh_vertices_in_img_space, -1, axis=1)
    mesh_vertices_in_img_space = mesh_vertices_in_img_space.astype(np.float64)

    return mesh_vertices_in_img_space


def generate_gt_volume(**kwargs):
    """
    Summary    
    -------

    Parameters
    ----------        

    Returns
    -------

    """

    files_directory = kwargs["files_directory"]
    ground_truth_mesh_file = kwargs["gt_mesh_filepath"]
    model2bone_transformation_file = kwargs["model2bone_transformation_filepath"]
    img2thigh_transformation_filename = kwargs["img2thigh_transformation_filename"]
    us_img_data_filename = kwargs["us_img_data_filename"]
    (start_at, end_at) = kwargs["start_end_indices_z_axis"] if "start_end_indices_z_axis" in kwargs else (0, -1)

    model_to_bone_transformation = np.loadtxt(model2bone_transformation_file)
    ground_truth_mesh = mesh.Mesh.from_file(ground_truth_mesh_file)
    img_to_thigh_transformation = np.loadtxt(os.path.join(files_directory, img2thigh_transformation_filename))    
    img_to_thigh_transformation = img_to_thigh_transformation.transpose()

    gt_mesh_vertices_in_img_space = transform_ground_truth_model_to_image_space(
        ground_truth_mesh, model_to_bone_transformation, img_to_thigh_transformation)

    # Initialize search tree for nearest neibors search
    search_tree = spatial.KDTree(gt_mesh_vertices_in_img_space)
    ultrasound_data = read_us_data(os.path.join(files_directory, us_img_data_filename))
    us_image_data = ultrasound_data['image_data']

    # fix indices
    # if end_at == -1 or end_at > us_image_data.shape[-1]:
    #     end_at = us_image_data.shape[-1]
    # if start_at < 0:
    #     start_at = 0

    # us_img_query_points = np.mgrid[0:us_image_data.shape[1], 0:us_image_data.shape[0], start_at:end_at].reshape(
    #     3, -1).T  # 0:us_image_data.shape[2]
    # us_img_query_points = us_img_query_points.astype(np.float64)

    if end_at == -1 or end_at > us_image_data.shape[1]:
        end_at = us_image_data.shape[1]
    if start_at < 0:
        start_at = 0

    # Swap x, y axes
    us_img_query_points = np.mgrid[start_at:end_at, 0:us_image_data.shape[0], 0:us_image_data.shape[2]].reshape(
        3, -1).T  # 0:us_image_data.shape[2]
    us_img_query_points = us_img_query_points.astype(np.float64)

    # Change position of x, y axis of us_image_data.shape. Because ??
    #gt_vol = create_ground_truth_2Dimage(xyz, search_tree, (us_image_data.shape[1], us_image_data.shape[0]))
    startime = time.time()
    print("a query started.")
    nearest_neibors = search_tree.query(
        us_img_query_points, distance_upper_bound=DISTANCE_UPPPER_BOUND)
    duration = (time.time() - startime) / 60
    print("a query took {} minutes".format(duration))

    distances = np.array(nearest_neibors[0])

    distances[distances != np.inf] = 1
    distances[distances == np.inf] = 0

    # ground_truth_vol = distances.reshape(
    #     (us_image_data.shape[1], us_image_data.shape[0], len(range(start_at, end_at))))

    ground_truth_vol = distances.reshape(
        (len(range(start_at, end_at)), us_image_data.shape[0], us_image_data.shape[-1]))

    # return ground_truth_vol, us_image_data[:, :, start_at:end_at]
    return ground_truth_vol, us_image_data


if __name__ == "__main__":
    """
    Show overlapped 2d images of ground truth and original images 
    at given slices in Z axis of given 3d US volume image.
    """

    # 13-37-47 13-51-47 14-00-18
    parent_directory = "D:\\Data\\IPASM\\bone_data\\in_vivo_data\\14_02_2019_Ben\\" #"D:\\Data\\IPASM\\bone_data\\phantom_data\\2018_11_23_CAOS_record\\"
    child_directories = [""]
    ground_truth_mesh_file = parent_directory + 'ground_truth.stl'    
    model2bone_transformation_file = parent_directory + 'model2bone.txt'
    output_directory = "D:\\Users\\odgiiv.khuurkhunkhuu\\Datasets\\14_02_2019_Ben_in_vivo\\"

    slice_indices_filename = "slice_indices.txt"
    us_img_data_filename = "vol_postProcessedImage_cropped.b8"
    img2thigh_transformation_filename = "image2thighTransformationRefined.txt"

    # Read start and end indices from file for each sub directory.
    # slice_indices = {}
    # with open(os.path.join(output_directory, slice_indices_filename)) as f:
    #     lines = f.readlines()

    #     for line in lines:
    #         [dirname, start, end] = line.strip().split(" ")
    #         slice_indices[dirname] = (int(start), int(end))

    # Iterate over each sub directory which contains us data in parent_directory.
    fs = sorted(os.listdir(parent_directory))
    for f in fs:
        # If f is directory, not a file
        files_directory = os.path.join(parent_directory, f)
        if os.path.isdir(files_directory):
            print("Entering directory: {}".format(files_directory))
        else:
            continue

        start_index_in_z, end_index_in_z = 0, -1  # slice_indices[f]
        if start_index_in_z == 0 and end_index_in_z == 0:
            continue

        gt_volume, us_img_volume = generate_gt_volume(files_directory=files_directory,
                                                      gt_mesh_filepath=ground_truth_mesh_file,
                                                      model2bone_transformation_filepath=model2bone_transformation_file,
                                                      img2thigh_transformation_filename=img2thigh_transformation_filename,
                                                      us_img_data_filename=us_img_data_filename,
                                                      start_end_indices_z_axis=(start_index_in_z, end_index_in_z))

        if not os.path.exists(os.path.join(output_directory, f)):
            os.makedirs(os.path.join(output_directory, f))

        h5f = h5py.File(os.path.join(output_directory, f, "us_gt_vol.h5"), 'w')
        h5f.create_dataset('us_vol', data=us_img_volume)
        h5f.create_dataset('gt_vol', data=gt_volume)
        h5f.close()
        print("written a h5 file.")

        # step = 1

        # for z_index in range(0, gt_volume.shape[-1], step):
        #     bone_img = Image.fromarray(us_img_volume[:, :, z_index])
        #     # Read image from numpy array as greyscale
        #     gt_img = Image.fromarray(
        #         np.uint8(gt_volume[:, :, z_index].T * 255), 'L')

        #     overlapped_img = Image.blend(bone_img, gt_img, 0.3)
        #     overlapped_img.show()

        #     print("Showing image at index {} / {} with step: {}".format(z_index,
        #                                                                 gt_volume.shape[-1] - 1, step))
        #     input("Press Enter to continue...")

    print("Finished.")
