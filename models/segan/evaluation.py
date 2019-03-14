import tensorflow as tf
from PIL import Image
import numpy as np
import os
import shutil
from scipy.special import expit


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def evaluate(test_model_specs, params):
    (X_test, Y_test) = test_model_specs["dataset"]
    segmentor_net = test_model_specs["segmentor_net"]

    for root, dirs, files in os.walk(params.test_results_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

    IoUs = []
    for i in range(X_test.shape[0]):

        img = X_test[i, :]
        label = Y_test[i, :]

        img = np.expand_dims(img, axis=0)
        img_norm = img/255 #np.max(img)
        img_norm = img_norm.astype('float32')

        pred = segmentor_net.predict_on_batch(img_norm)
        # pred = expit(pred)  # sigmoid

        # round like tf.round
        pred[pred <= 0.5] = 0
        pred[pred > 0.5] = 1

        pred = np.squeeze(pred, axis=(0))

        print("pred shape: ", pred.shape)
        print("img shape: ", img.shape)
        print("label shape: ", label.shape)

        IoU = np.sum(pred[label == 1]) / (float(np.sum(pred) + np.sum(label) - np.sum(pred[label == 1])))
        IoUs.append(IoU)

        pred_img = np.squeeze(pred) * 255
        label_img = np.squeeze(label) * 255
        img = np.squeeze(img)

        # print(img.shape, pred.shape, label.shape)

        pred_img = Image.fromarray(pred_img.astype(np.uint8), mode='P')
        label_img = Image.fromarray(label_img.astype(np.uint8), mode='P')
        img = Image.fromarray(img.astype(np.uint8), mode='P')

        I = Image.new('RGB', (pred.shape[1]*3, pred.shape[0]))
        I.paste(img, (0, 0))
        I.paste(label_img, (pred.shape[1], 0))
        I.paste(pred_img, (pred.shape[1]*2, 0))

        name = str(i) + '.jpg'
        I.save(os.path.join(params.test_results_path, name))

    IoUs = np.array(IoUs, dtype=np.float64)
    mIoU = np.mean(IoUs, axis=0)
    print("mIoU: ", mIoU)
