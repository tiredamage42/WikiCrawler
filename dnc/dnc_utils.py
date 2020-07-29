import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def reduce_prod(x, axis, name=None):
    """
    Efficient reduce product over axis.
    Uses tf.cumprod and tf.gather_nd as a workaround to the poor performance of calculating tf.reduce_prod's gradient on CPU.
    """
    with tf.name_scope(name, 'util_reduce_prod', values=[x]):
        cp = tf.cumprod(x, axis, reverse=True)
        size = tf.shape(cp)[0]
        idx1 = tf.range(tf.cast(size, tf.float32), dtype=tf.float32)
        idx2 = tf.zeros([size], tf.float32)
        indices = tf.stack([idx1, idx2], 1)
        return tf.gather_nd(cp, tf.cast(indices, tf.int32))


