import tensorflow as tf
from PIL import Image
import numpy as np
import os
import shutil
from scipy.special import expit


def evaluate(test_model_specs, params):
    (X_test, Y_test) = test_model_specs["dataset"]
    segmentor_net = test_model_specs["segmentor_net"]

    for root, dirs, files in os.walk(params.test_results_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

    IoUs = []
    sess = tf.keras.backend.get_session()   
    sess.run(tf.global_variables_initializer())

    segmentor_net.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    segmentor_net.fit(x=np.zeros((1,1,1,1)), y=np.zeros((1,1,1,1)), epochs=0, steps_per_epoch=0)
    segmentor_net.load_weights(params.save_weights_path + 'segan_best_weights.h5')

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