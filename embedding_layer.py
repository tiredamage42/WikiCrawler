import tensorflow.compat.v1 as tf
import graph_utils

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
                    'embeddings', [self.vocab_size, self.embedding_size], dtype=tf.float32, 
                    initializer=None, trainable=True, save_var=True
                )
                self.is_built = True

            return tf.nn.embedding_lookup(self.embeddings, in_tensor)
    