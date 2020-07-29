import tensorflow.compat.v1 as tf
from . import graph_utils

tf.disable_v2_behavior()
    
'''
class embeddings for Natural Language Processing
'''
class Embeddings():
    def __init__(self, layer_name, vocab_size, embedding_size):
        self.layer_name = layer_name
        self.is_built = False
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        
                
    def __call__(self, in_tensor):    
        with tf.variable_scope(self.layer_name) as scope:

            if not self.is_built:
                self.embeddings = graph_utils.get_variable(
                    'embeddings', 
                    [self.vocab_size, self.embedding_size], 
                    dtype=tf.float32, 
                    initializer=None, 
                    trainable=True, 
                    save_var=True
                )
                self.is_built = True

            return tf.nn.embedding_lookup(self.embeddings, in_tensor)
    
'''
fully connected dense layer
'''
class Dense():
    def __init__(self, layer_name, features, activation=tf.nn.leaky_relu, use_bias=True):
        self.is_built = False
        self.layer_name = layer_name
        self.use_bias = use_bias
        self.features = features
        self.activation = activation

    def __call__(self, in_tensor):
        with tf.variable_scope(self.layer_name) as scope:

            in_shape_g = tf.shape(in_tensor)
            in_shape_l = in_tensor.get_shape().as_list()
            if not self.is_built:
                self.b = None
                self.w = graph_utils.get_variable(
                    'weights', 
                    [in_shape_l[-1], self.features], 
                    dtype=tf.float32, 
                    initializer=None, 
                    trainable=True, 
                    save_var=True
                )
                if self.use_bias:
                    self.b = graph_utils.get_variable(
                        'biases', 
                        [self.features], 
                        dtype=tf.float32, 
                        #getting better results with default glorot initializer
                        initializer=None,#tf.zeros_initializer, 
                        trainable=True, 
                        save_var=True
                    )
                
                self.is_built = True

            # if we get a tensor of shape [ batch, sequence, features ]
            # reshape so it's shape [ batch * sequence, features ]
            needs_reshape = len(in_shape_l) != 2
            if needs_reshape:
                in_tensor = tf.reshape(in_tensor, [ -1, in_shape_l[-1] ], name="flatten")
                
            # matrix multiplication
            out_t = tf.matmul(in_tensor, self.w, name="mx")

            # maybe add bias
            if self.b is not None:
                out_t = tf.add(out_t, self.b, name="add_bias")
            
            # maybe activate
            if self.activation is not None:
                out_t = self.activation(out_t, name="activated")
            
            # reshape back to [ batch, sequence, self.features ]
            # after our matrix multiplication
            if needs_reshape:
                out_t = tf.reshape(out_t, tf.concat([in_shape_g[:-1], tf.constant([self.features])], -1))
            
            return out_t

        