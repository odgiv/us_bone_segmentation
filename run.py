"""
Used for running model on a single US image.
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

def run(model, weight_file_path, input_imgs_dir, output_imgs_dir, store_imgs, thinning):           

    segmentor_net = model
    IoUs = []
    hds = []
    dices = []    
    combis = []
    
    sess = tf.keras.backend.get_session()
    sess.run(tf.global_variables_initializer())

    segmentor_net.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    segmentor_net.fit(x=np.zeros((1, IMG_H, IMG_W, 1)), y=np.zeros((1, IMG_H, IMG_W, 1)), epochs=0, steps_per_epoch=0)
    segmentor_net.load_weights(weight_file_path)

    
    i = 0
    for img_fname in os.listdir(input_imgs_dir):

        if img_fname[-4:] not in [".jpg", ".png", ".JPEG", ".PNG"]:
            continue
        
        img = cv.imread(os.path.join(input_imgs_dir, img_fname), cv.IMREAD_GRAYSCALE)
            
        img = np.expand_dims(img, axis=0)        
        img = np.expand_dims(img, axis=-1)        
        pred = segmentor_net(tf.convert_to_tensor(img, tf.float32))

        pred_np = pred.eval(session=sess)
        pred_np = np.argmax(pred_np, axis=-1)
        pred_np = np.expand_dims(pred_np, -1)
        
        
        if thinning:            
            
            pred_np = skeletonize(np.squeeze(pred_np))

            new_pred = np.array(pred_np)

            labels, num = find_connected(pred_np, return_num=True)
            
            for n in range(1, num+1):
                if np.sum(labels == n) <= 100:

                    new_pred[labels == n] = 0
            
            pred_np = new_pred

        if store_imgs:
                    
            img = np.squeeze(img) # 255
            new_img = np.zeros((*img.shape, 3), dtype=np.uint8)
            new_img[:,:,0] = img.astype(np.uint8)
            new_img[:,:,1] = img.astype(np.uint8)
            new_img[:,:,2] = img.astype(np.uint8)
            new_img[:,:,1][pred_np == 1] =  255
            new_img[:,:,0][pred_np == 1] =  255
             
            name = '{}_segmented{}'.format(img_fname[:-4], img_fname[-4:])
            
            
            cv.imwrite(os.path.join(output_imgs_dir, name), new_img.astype('uint8'))
        i += 1    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", help="Name of the model. Use either of unet or attention-unet.")    
    parser.add_argument("-i", "--input_imgs_dir", help="Directory path where input images are.")            
    parser.add_argument("-o", "--output_imgs_dir", help="Directory path where input images are.")            
    parser.add_argument("-w", "--weight_file", help="The weight file name")
    parser.add_argument("-s", "--store_imgs", default=False, action='store_true')    
    parser.add_argument("-t", "--thinning", default=False, action='store_true')
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

    run(model, args.weight_file, args.input_imgs_dir, args.output_imgs_dir, args.store_imgs, args.thinning)