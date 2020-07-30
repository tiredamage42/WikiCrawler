'''
some custom cells with added save state functionality
cells inheret from tensorflow rnn cell, so can be used with tensorflow rnns 
'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.ops.rnn_cell import LSTMStateTuple, BasicLSTMCell

class MultiRNNCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, cells):
        '''
        cells:
            list of tensorflow rnn cells
        '''
        self.cells = cells
    
    @property
    def output_size(self):
        return self.cells[-1].output_size
        
    @property
    def state_size(self):
        return tuple(c.state_size for c in self.cells)
    
    def update_state(self, new_state):
        '''
        when saving states, add this as a dependency before the rnn output 
        to update state variables in the cell
        '''
        with tf.control_dependencies([ self.cells[i].update_state(s) for i, s in enumerate(new_state) ]):
            return tf.no_op()

    def reset_state(self):
        '''
        reset states to zero states
        '''
        with tf.control_dependencies([ c.reset_state() for c in self.cells ]):
            return tf.no_op()
    
    def zero_state(self, batch_size, dtype):
        return tuple(c.zero_state(batch_size, dtype) for c in self.cells)
        
    def __call__(self, inputs, state, scope=None):
        out = inputs
        new_state = []
        for i, cell in enumerate(self.cells):
            out, state_c = cell(out, state[i])
            new_state.append(state_c)
        return out, tuple(new_state)

'''
wrapper for tensorflow basic lstm cell, saves state in between session runs
'''
class LSTMSavedState(tf.nn.rnn_cell.RNNCell):
    def __init__(self, features, batch_size):
        self.cell = BasicLSTMCell(features)
        self.h = tf.Variable(tf.zeros([batch_size, features]), trainable=False)
        self.c = tf.Variable(tf.zeros([batch_size, features]), trainable=False)
        self.batch_size = batch_size
        self.features = features

    def update_state(self, new_state):
        return tf.group(self.h.assign(new_state[0]), self.c.assign(new_state[1]))
    
    def reset_state(self):
        return tf.group(self.h.assign(tf.zeros_like(self.h)), self.c.assign(tf.zeros_like(self.c)))

    @property
    def state_size(self):
        return self.cell.state_size

    @property
    def output_size(self):
        return self.cell.output_size

    def zero_state(self, batch_size, dtype):
        return LSTMStateTuple(self.h, self.c)
        
    def __call__(self, inputs, state, scope=None):
        out, state = self.cell(inputs, state)
        h, c = state
        return out, LSTMStateTuple(h, c)
   
