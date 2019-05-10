import tensorflow as tf
from data_loader import DataLoader
from misc.preprocessGt import preprocess_gt


def _parse_function(image, label):
    #label = preprocess_gt(image, label, is_threshold=False, is_clear_below_bps=True)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, tf.cast(label, tf.float32)


def input_fn(is_training, is_eager, X, Y, params):
    
    num_samples = X.shape[0]

    def parse_fn(f, l): return _parse_function(f, l)

    if is_training:
        if is_eager:
            dataset = (tf.data.Dataset.from_tensor_slices((X, Y))
                    .shuffle(num_samples)
                    .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
                    .batch(params.batch_size)
                    .prefetch(1)
                    )
        else:
            features_placeholder = tf.placeholder(X.dtype, X.shape)
            labels_placeholder = tf.placeholder(Y.dtype, Y.shape)
            dataset = (tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
                    .shuffle(num_samples)
                    .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
                    .batch(params.batch_size)
                    .prefetch(1)
            )
            iterator = dataset.make_initializable_iterator()
            images, labels = iterator.get_next()
            iterator_init_op = iterator.initializer

            return {"images": images,
                    "labels": labels, 
                    "X_placeholder": features_placeholder, 
                    "Y_placeholder": labels_placeholder,
                    "X": X, 
                    "Y": Y,
                    "iterator_init_op": iterator_init_op
                    }
    else:
        dataset = (X, Y)
        

    return {"dataset": dataset}
