"""
This script is used for evaluating model on test dataset.

Usage:
python evaluate.py -m unet -w unet_epoch_4_val_meanIoU_0.333_meanLoss_0.053_meanHd_66.633_meanDice_0.492_mCombi_133.338.h5  -d /media/dataraid/tensorflow/segm/experiments_data/ -s -i 7 -t

resulting output images are in ../model_dir/test_results/
"""
import tensorflow as tf
import argparse
import os
import sys
import json
import numpy as np
import cv2 as cv
from PIL import Image
from utils import Params, img_and_mask_generator, delete_dir_content, hausdorf_distance
from skimage.morphology import skeletonize, label as find_connected

IMG_H = 465
IMG_W = 381
TEST_IMGS_DIR = "test300_imgs"
TEST_GTS_DIR = "test300_gts"

def eval(model, model_dir, weight_file_path, store_imgs, dataset_path, ex_id, thinning):
    
    img_h = 465
    img_w = 381
    test_results_path = os.path.join(model_dir, "test_results", "new_results")
    x_test_path = os.path.join(dataset_path, TEST_IMGS_DIR)
    y_test_path = os.path.join(dataset_path, TEST_GTS_DIR)
    test_gen = img_and_mask_generator(x_test_path, y_test_path, batch_size=1, shuffle=False)

    x_test_path_data = os.path.join(x_test_path, 'data')
    num_imgs = len([name for name in os.listdir(x_test_path_data) if os.path.isfile(os.path.join(x_test_path_data, name))])

    segmentor_net = model
    IoUs = []
    hds = []
    dices = []    
    combis = []
    
    sess = tf.keras.backend.get_session()
    sess.run(tf.global_variables_initializer())

    segmentor_net.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    segmentor_net.fit(x=np.zeros((1, img_h, img_w, 1)), y=np.zeros((1, img_h, img_w, 1)), epochs=0, steps_per_epoch=0)
    segmentor_net.load_weights(weight_file_path)

    if not os.path.isdir(test_results_path):
        os.makedirs(test_results_path)
    elif store_imgs:
        delete_dir_content(test_results_path)

    i = 0
    for img, label in test_gen:
                
        pred = segmentor_net(tf.convert_to_tensor(img, tf.float32))

        pred_np = pred.eval(session=sess)
        pred_np = np.argmax(pred_np, axis=-1)
        pred_np = np.expand_dims(pred_np, -1)

        label[label>=0.5] = 1
        label[label<0.5] = 0        
        label = label.astype('uint8')     
        
        if thinning:            
            label = skeletonize(np.squeeze(label))
            pred_np = skeletonize(np.squeeze(pred_np))
            new_pred = np.array(pred_np)

            labels, num = find_connected(pred_np, return_num=True)
            
            for n in range(1, num+1):
                if np.sum(labels == n) <= 50:

                    new_pred[labels == n] = 0
            
            pred_np = new_pred


        IoU = np.sum(pred_np[label == 1]) / float(np.sum(pred_np) + np.sum(label) - np.sum(pred_np[label == 1]))
        print("Iou: ", IoU)
        IoUs.append(IoU)
        
        label = np.squeeze(label) 
        pred_np = np.squeeze(pred_np)
        # pred = pred.astype('uint8')

        pred_locations = np.argwhere(pred_np == 1)
        label_locations = np.argwhere(label == 1)        

        hd = hausdorf_distance(pred_locations, label_locations)
        hds.append(hd)

        print("Hausdorf: ", hd)

        combi = 100 * (1-IoU) + hd 

        combis.append(combi)
        print("Combi: ", combi)

        # img = np.squeeze(img) * 255
        # pred_img = pred_np * 255
        # label_img = label * 255
        if store_imgs:
            # pred_img = Image.fromarray(pred_img.astype(np.uint8), mode='P')
            # label_img = Image.fromarray(label_img.astype(np.uint8), mode='P')
            # img = Image.fromarray(img.astype(np.uint8), mode='P')

            # I = Image.new('RGB', (img.size[0]*5, img.size[1]))
            # I.paste(img, (0, 0))
            # I.paste(label_img, (img.size[0], 0))
            # I.paste(pred_img, (img.size[0]*2, 0))
            # I.paste(Image.blend(img.convert("L"), label_img.convert("L"), 0.2), (img.size[0]*3, 0))
            # I.paste(Image.blend(img.convert("L"), pred_img.convert("L"), 0.2), (img.size[0]*4, 0))
            

            #I = cv.cvtColor(np.float32(img), cv.COLOR_GRAY2RGB)
            
            img = np.squeeze(img) * 255
            new_img = np.zeros((*img.shape, 3), dtype=np.uint8)
            new_img[:,:,0] = img.astype(np.uint8)
            new_img[:,:,1] = img.astype(np.uint8)
            new_img[:,:,2] = img.astype(np.uint8)
            new_img[:,:,1][pred_np == 1] =  255
            new_img[:,:,0][pred_np == 1] =  255

            new_img[:,:,1][label == 1] =  255
             
            name = 'img_{}_iou_{:.4f}_hausdorf_{:.4f}.jpg'.format(i, IoU, hd)
            #I = Image.fromarray(new_img.astype('uint8'))
            #I.save(os.path.join(test_results_path, name))
            
            cv.imwrite(os.path.join(test_results_path, name),new_img.astype('uint8'))
        i += 1

        if num_imgs == i:
            break

        print(str(i) + '/' + str(num_imgs))

    IoUs = np.array(IoUs, dtype=np.float64)
    mIoU = np.mean(IoUs, axis=0)
    hds = np.array(hds, dtype=np.float64)
    mhd = np.mean(hds, axis=0)
    combis = np.array(combis, dtype=np.float64)
    mCombi = np.mean(combis, axis=0)
    print("mIoU: ", mIoU, "mHausdorf:", mhd, "mCombi:", mCombi)

    weight_file_name = os.path.basename(weight_file_path) 
    start_index = weight_file_name.find("epoch_")
    end_index = weight_file_name.find("_val_")
    epoch_num =  weight_file_name[start_index+len("epoch_"):end_index]
    
    os.rename(test_results_path, os.path.join(model_dir, "test_results", "exp{}_epoch{}_mIoU_{:.2f}_mHd_{:.2f}_mC_{:.2f}_nImgs_{}".format(ex_id, epoch_num, mIoU, mhd, mCombi, num_imgs)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", help="Name of the model. Use either of unet or attention-unet.")
    parser.add_argument("-d", "--dataset_path", required=True)
    parser.add_argument("-w", "--weight_file", help="The weight file name")
    parser.add_argument("-s", "--store_imgs", default=False, action='store_true')
    parser.add_argument("-i", "--exp_id", type=int, required=True)
    parser.add_argument("-t", "--thinning", action='store_true')
    args = parser.parse_args()

    assert(args.model_name in ['unet', 'attentionUnet'])

    sys.path.append('./models/unet')

    if args.model_name == 'unet':
        model_dir = './models/unet'    
        from base_model import Unet
        model = Unet()

    elif args.model_name == 'attentionUnet' :        
        model_dir = './models/attentionUnet'
        sys.path.append(model_dir)
        from model import AttentionalUnet
        model = AttentionalUnet()

    eval(model, model_dir, os.path.join(model_dir, "experiments", "experiment_id_%d" % args.exp_id, args.weight_file), args.store_imgs, args.dataset_path, args.exp_id, args.thinning)
    
