import tensorflow as tf
from PIL import Image
import numpy as np
import os
from input_fn import _parse_function
import shutil
import scipy

def sigmoid(x):
    return 1./ (1. + np.exp(-x))


def evaluate(test_model_specs, params):
    (X_test, Y_test) = test_model_specs["dataset"]
    segmentor_net = test_model_specs["segmentor_net"]

    for root, dirs, files in os.walk(params.test_results_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

    for i in range(X_test.shape[0]):

        img = X_test[i,:]
        label = Y_test[i,:]
        #img, label = _parse_function(img, label)
        
        img = np.expand_dims(img, axis=0)   
        img = img.astype('float32')

        # print(img.shape, label.shape)
        pred = segmentor_net.predict_on_batch(img)
        pred = scipy.special.expit(pred)
        pred = np.squeeze(pred)    

    
        pred[pred < 1.0] = 0
        pred[pred >= 1.0] = 1
        
        pred = pred * 255

        label = np.squeeze(label) * 255    
        img = np.squeeze(img)

        # print(img.shape, pred.shape, label.shape)

        pred_img = Image.fromarray(pred.astype(np.uint8), mode='P')        
        label_img = Image.fromarray(label.astype(np.uint8), mode='P')
        img = Image.fromarray(img.astype(np.uint8), mode='P')

        I = Image.new('RGB', (pred.shape[1]*3, pred.shape[0]))
        I.paste(img, (0, 0))
        I.paste(label_img, (pred.shape[1], 0))
        I.paste(pred_img, (pred.shape[1]*2, 0))

        name = str(i) + '.jpg'
        I.save(os.path.join(params.test_results_path, name))
        