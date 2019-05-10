import tensorflow as tf
from PIL import Image
import numpy as np
import os
import utils
from scipy.special import expit


def evaluate(test_model_specs, params):
    (X_test, Y_test) = test_model_specs["dataset"]
    unet = test_model_specs["unet"]

    IoUs = []
    hds = []
    sess = tf.keras.backend.get_session()   
    sess.run(tf.global_variables_initializer())
    
    weight_file_path = os.path.join(os.path.abspath(os.path.join(os.path.realpath(__file__), '..')), params.weight_file_subpath)
    print("FULL_WEIGHT_PATH", weight_file_path)
    unet.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    unet.fit(x=np.zeros((1,params.img_h,params.img_w,1)), y=np.zeros((1,params.img_h,params.img_w,1)), epochs=0, steps_per_epoch=0)
    unet.load_weights(weight_file_path)

    if not os.path.isdir(params.test_results_path):
        os.mkdir(params.test_results_path)
    else:
        utils.delete_dir_content(params.test_results_path)

    for i in range(X_test.shape[0]):

        img = X_test[i, :]
        label = Y_test[i, :]

        img = np.expand_dims(img, axis=0)
        img_norm = img/255 #np.max(img)
        img_norm = img_norm.astype('float32')

        img_norm = tf.convert_to_tensor(img_norm)
        pred = unet(img_norm)

        # print(pred.shape)
        # print(img.shape)

        pred = pred.eval(session=sess)
        

        pred = np.argmax(pred, axis=-1)
        pred = np.expand_dims(pred, -1)
        pred = np.squeeze(pred, axis=0)

        IoU = np.sum(pred[label == 1]) / float(np.sum(pred) + np.sum(label) - np.sum(pred[label == 1]))
        print("Iou: ", IoU)
        IoUs.append(IoU)

        pred_img = np.squeeze(pred)
        pred_img = np.uint8(pred_img)
        
        pred_coordinates = np.argwhere(pred_img == 1)

        # print(pred_coordinates)

        label_img = np.squeeze(label) 
        label_img = np.uint8(label_img)

        label_coordinates = np.argwhere(label_img == 1)
        # print(label_coordinates)
        
        img = np.squeeze(img)

        hd = utils.hausdorf_distance(pred_coordinates, label_coordinates)
        hds.append(hd)
        print("Hausdorf: ", hd)

        # pred_img = pred_img * 255
        # label_img = label_img * 255
    
        # pred_img = Image.fromarray(pred_img.astype(np.uint8), mode='P')
        # label_img = Image.fromarray(label_img.astype(np.uint8), mode='P')
        # img = Image.fromarray(img.astype(np.uint8), mode='P')

        # I = Image.new('RGB', (pred_img.size[0]*3, pred_img.size[1]))
        # I.paste(img, (0, 0))
        # I.paste(label_img, (pred_img.size[0], 0))
        # I.paste(pred_img, (pred_img.size[0]*2, 0))

        # name = 'img_{}_iou_{:.4f}_hausdorf_{:.4f}.jpg'.format(i, IoU, hd)
        # I.save(os.path.join(params.test_results_path, name))
        
        print(str(i) + '/' + str(X_test.shape[0]))
    IoUs = np.array(IoUs, dtype=np.float64)
    mIoU = np.mean(IoUs, axis=0)
    mhd = np.mean(hds, axis=0)
    print("mIoU: ", mIoU, "mHausdorf:", mhd)
