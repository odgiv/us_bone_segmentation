import tensorflow as tf
from PIL import Image
import numpy as np
import os
import utils
from scipy.special import expit


def evaluate(test_model_specs, params):
    (X_test, Y_test) = test_model_specs["dataset"]
    unet = test_model_specs["unet"]

    utils.delete_dir_contents(params.test_results_path)

    IoUs = []
    sess = tf.keras.backend.get_session()   
    sess.run(tf.global_variables_initializer())

    unet.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    unet.fit(x=np.zeros((1,params.img_h,params.img_w,1)), y=np.zeros((1,params.img_h,params.img_w,1)), epochs=0, steps_per_epoch=0)
    unet.load_weights(params.save_weights_path + 'unet_weights_val_maxIoU_0.948.h5')

    if not os.path.isdir(params.test_results_path):
        os.mkdir(params.test_results_path)

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
        print(IoU)
        IoUs.append(IoU)

        pred_img = np.squeeze(pred) * 255
        label_img = np.squeeze(label) * 255
        img = np.squeeze(img)
        


        # pred_img = Image.fromarray(np.asarray(pred_img.eval(session=sess), dtype=np.uint8), mode='P')
        pred_img = Image.fromarray(pred_img.astype(np.uint8), mode='P')
        label_img = Image.fromarray(label_img.astype(np.uint8), mode='P')
        img = Image.fromarray(img.astype(np.uint8), mode='P')

        I = Image.new('RGB', (pred_img.size[0]*3, pred_img.size[1]))
        I.paste(img, (0, 0))
        I.paste(label_img, (pred_img.size[0], 0))
        I.paste(pred_img, (pred_img.size[0]*2, 0))

        name = str(i) + '.jpg'
        I.save(os.path.join(params.test_results_path, name))

    IoUs = np.array(IoUs, dtype=np.float64)
    mIoU = np.mean(IoUs, axis=0)

    print("mIoU: ", mIoU)
