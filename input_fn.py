import tensorflow as tf
from data_loader import DataLoader

def _parse_function(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

def input_fn(is_training, X, Y, params):
    features_placeholder = tf.placeholder(X.dtype, X.shape)
    labels_placeholder = tf.placeholder(Y.dtype, Y.shape)
    num_samples = X.shape[0]

    parse_fn = lambda f, l: _parse_function(f, l)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
            .shuffle(num_samples)
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1)
        )
    else: 
        dataset = (tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1)
        )

    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer
    
    inputs = {"images": images, "labels": labels, "iterator_init_op": iterator_init_op, "X": X, "Y": Y, "X_placeholder": features_placeholder, "Y_placeholder": labels_placeholder}
    return inputs
