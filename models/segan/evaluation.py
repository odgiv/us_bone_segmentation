import tensorflow as tf
from PIL import Image
import numpy as np
import os
import utils
from scipy.special import expit


def evaluate(test_model_specs, params):
    (X_test, Y_test) = test_model_specs["dataset"]
    segmentor_net = test_model_specs["segmentor_net"]

    utils.delete_dir_content(params.test_results_path)

    IoUs = []
    hds = []
    sess = tf.keras.backend.get_session()   
    sess.run(tf.global_variables_initializer())

    weight_file_path = os.path.join(os.path.abspath(os.path.join(os.path.realpath(__file__), '..')), params.weight_file_subpath)

    segmentor_net.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    segmentor_net.fit(x=np.zeros((1,1,1,1)), y=np.zeros((1,1,1,1)), epochs=0, steps_per_epoch=0)
    segmentor_net.load_weights(weight_file_path)

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
        pred = segmentor_net(img_norm)
        # pred = expit(pred)  # sigmoid

        pred = pred.eval(session=sess)
        # print(type(pred))
        # round like tf.round
        pred[pred <= 0.5] = 0
        pred[pred > 0.5] = 1 
        
        # pred = tf.round(pred)  

        # print("pred tensor shape: ", pred.get_shape())
        # pred = tf.squeeze(pred, axis=(0))

        # print("pred shape: ", pred.shape)
        # print("img shape: ", img.shape)
        # print("label shape: ", label.shape)

        # intersection = tf.boolean_mask(pred, label == 1)
        # tensor_label = tf.cast(label, tf.float32)
        # gt_sum = tf.reduce_sum(tensor_label)
        pred = np.squeeze(pred, axis=(0))
        # IoU = tf.reduce_sum(intersection) / (tf.to_float(tf.reduce_sum(pred) +  gt_sum - tf.reduce_sum(intersection)))
                    
        # # iou = sess.run(IoU)
        # print(iou)
        IoU = np.sum(pred[label == 1]) / float(np.sum(pred) + np.sum(label) - np.sum(pred[label == 1]))
        print("Iou: ", IoU)
        IoUs.append(IoU)

        pred_img = np.squeeze(pred)
        pred_img = np.uint8(pred_img)
        
        pred_coordinates = np.argwhere(pred_img == 1)

        label_img = np.squeeze(label)
        label_img = np.uint8(label_img)

        label_coordinates = np.argwhere(label_img == 1)

        hd = utils.hausdorf_distance(pred_coordinates, label_coordinates)
        hds.append(hd)
        print("Hausdorf: ", hd)

        pred_img = pred_img * 255
        label_img = label_img * 255
        img = np.squeeze(img)
        
        # generate each image
        pred_img = Image.fromarray(pred_img.astype(np.uint8), mode='P')
        label_img = Image.fromarray(label_img.astype(np.uint8), mode='P')
        img = Image.fromarray(img.astype(np.uint8), mode='P')

        I = Image.new('RGB', (pred_img.size[0]*3, pred_img.size[1]))
        I.paste(img, (0, 0))
        I.paste(label_img, (pred_img.size[0], 0))
        I.paste(pred_img, (pred_img.size[0]*2, 0))

        name = 'img_{}_iou_{:.4f}_hausdorf_{:.4f}.jpg'.format(i, IoU, hd)
        I.save(os.path.join(params.test_results_path, name))

        print(str(i) + '/' + str(X_test.shape[0]))

    IoUs = np.array(IoUs, dtype=np.float64)
    mIoU = np.mean(IoUs, axis=0)
    mhd = np.mean(hds, axis=0)
    print("mIoU: ", mIoU, "mHausdorf: ", mhd)
