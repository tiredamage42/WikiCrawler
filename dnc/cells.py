import tensorflow as tf
import collections
import numpy as np

tf.disable_v2_behavior()

'''
a non-recurrent 'cell' to use as the controller

using this makes sure that any recurrent functionality
is learned using the DNC's read/write heads.
'''

class LinearDummy(tf.nn.rnn_cell.RNNCell):
    def __init__(self, layer_name, features):
        self.layer_name = layer_name
        self.features = features
        self.is_built = False
    
    @property
    def state_size(self):
        return 1
    @property
    def output_size(self):
        return self.features
    
    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):  
            with tf.variable_scope(self.layer_name) as scope:

                if not self.is_built:
                    in_shape_l = inputs.get_shape().as_list()
                    self.w = tf.get_variable('weights', shape=[in_shape_l[-1], self.features], dtype=tf.float32, initializer=None, trainable=True)
                    self.b = tf.get_variable('bias', shape=[self.features], dtype=tf.float32, initializer=None, trainable=True)        
                    self.is_built = True
        
                mx = tf.matmul(inputs, self.w, name="mx")
                # add bias
                add_b = tf.add(mx, self.b, name="add_bias")
                # activate
                out = tf.nn.leaky_relu(add_b, name="activated")
        
        return out, state

    def zero_state(self, batch_size, dtype):
        return tf.zeros([batch_size, 1])
    def update_state(self, final_state):
        return tf.no_op()
    def reset_state(self):
        return tf.no_op()
