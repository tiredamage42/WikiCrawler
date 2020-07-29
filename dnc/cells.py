import tensorflow.compat.v1 as tf
import collections
from . import layers, graph_utils

tf.disable_v2_behavior()

'''
a non-recurrent 'cell' to use as the controller

using this makes sure that any recurrent functionality
is learned using the DNC's read/write heads.
'''

class StatelessCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, layer_name, features):
        self.linear = layers.Dense(layer_name, features, activation=tf.nn.leaky_relu)
    
    @property
    def state_size(self):
        return 1
    
    @property
    def output_size(self):
        return self.linear.features
    
    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            out = self.linear(inputs)
        return out, state

    def zero_state(self, batch_size, dtype):
        return tf.zeros([batch_size, 1])

    def update_state(self, final_state):
        return tf.no_op()
    
    def reset_state(self):
        return tf.no_op()


"""
constructs Differentiable Neural Computer architecture

use as an rnn cell
"""
from dnc.memory_access import MemoryAccess

DNCStateTuple = collections.namedtuple("DNCStateTuple", (
    "controller_state", 
    "access_state", 
    "read_vectors", 
))

class DNCCell(tf.nn.rnn_cell.RNNCell):

    @property
    def state_size(self):
        return DNCStateTuple(
            controller_state=self.controller.state_size, 
            access_state=self.memory.state_size, 
            read_vectors=self.memory.output_size
        )
        
    @property
    def output_size(self):
        return self.controller.output_size + self.memory.output_size
        
    def zero_state(self, batch_size, dtype):
        return DNCStateTuple(
            controller_state=self.controller.zero_state(batch_size, dtype), 
            access_state=self.memory.zero_state(batch_size, dtype),
            read_vectors=self.read_vectors_state if self.save_state else tf.zeros([batch_size,] + [self.memory.output_size,], tf.float32)
        )

    def update_state(self, final_state):
        assert isinstance(final_state, DNCStateTuple)
        dependencies = [
            self.controller.update_state(final_state.controller_state), 
            self.memory.update_state(final_state.access_state),
            tf.assign(self.read_vectors_state, final_state.read_vectors)
        ] if self.save_state else []
        with tf.control_dependencies(dependencies):
            return tf.no_op()
    
    def reset_state(self):
        dependencies = [
            self.controller.reset_state(), 
            self.memory.reset_state(),
            tf.assign(self.read_vectors_state, tf.zeros_like(self.read_vectors_state))
        ] if self.save_state else []
        with tf.control_dependencies(dependencies):
            return tf.no_op()
   
    def __init__(self, controller_cell, save_state=False, batch_size=None, memory_size = 256, word_size = 64, num_reads = 4, num_writes = 1, clip_value=None):
        """
        controller_cell: 
            Tensorflow RNN Cell
        """     
        self.memory = MemoryAccess(memory_size, word_size, num_reads, num_writes, save_state, batch_size)
        
        self.controller = controller_cell
        self._clip_value = clip_value or 0
        self.save_state = self.memory.save_state
        if self.save_state:
            assert batch_size is not None
            with tf.variable_scope('dnc'):
                self.read_vectors_state = graph_utils.get_variable(
                    'rv_state', 
                    shape=[batch_size, self.memory.output_size], 
                    dtype=tf.float32, 
                    initializer=tf.zeros_initializer, 
                    trainable=False, save_var=False
                )

        
    def _clip_if_enabled(self, x):
        if self._clip_value <= 0:
            return x
        return tf.clip_by_value(x, -self._clip_value, self._clip_value)
        
    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__): 
            
            controller_state, access_state, read_vectors = state

            #concatenate last read vectors
            complete_input = tf.concat([inputs, read_vectors], -1)
            #processes input data through the controller network
            controller_output, controller_state = self.controller(complete_input, controller_state)
            
            controller_output = self._clip_if_enabled(controller_output)
            
            #processes input data through the memory module
            read_vectors, access_state = self.memory(controller_output, access_state)
            read_vectors = self._clip_if_enabled(read_vectors)

            #the final output by taking rececnt memory changes into account
            step_out = tf.concat([controller_output, read_vectors], -1)
            
            #return output and teh new DNC state
            return step_out, DNCStateTuple(controller_state=controller_state, access_state=access_state, read_vectors=read_vectors)
            
    