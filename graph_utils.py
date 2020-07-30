'''
tensorflow graph utils

global step, training bool, any saved graph variables that arent trainable, etc...
'''

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

GLOBAL_STEP = None
SAVED_STEP_INFO = None

_INCREMENT_STEP = None
_STEP_INFO_PH = None
_ASSIGN_STEP_INFO = None


'''
save custom information in checkpoint variables
loss, accuracy, etc...
'''
def save_step_info(session, step_info):
    fd = {_STEP_INFO_PH: step_info}
    session.run(_ASSIGN_STEP_INFO, fd)

'''
initialize variables and ops
'''
def initialize_graph_constants(training_mode, saved_info_shape=None):
    global GLOBAL_STEP
    global SAVED_STEP_INFO
    global _INCREMENT_STEP
    global _STEP_INFO_PH
    global _ASSIGN_STEP_INFO
    _INCREMENT_STEP = None
    _STEP_INFO_PH = None
    _ASSIGN_STEP_INFO = None
    
    if training_mode:
        GLOBAL_STEP = tf.Variable(0, dtype=tf.int64, name='global_step', trainable=False)
        tf.add_to_collection(SAVE_VARS, GLOBAL_STEP)
        # op to add 1 to gloal step
        _INCREMENT_STEP = tf.assign(GLOBAL_STEP, GLOBAL_STEP + 1)
                
        #any custom info variable
        SAVED_STEP_INFO = tf.Variable(tf.fill([], 0.0), dtype=tf.float32, name='saved_step_info', trainable=False)
        tf.add_to_collection(SAVE_VARS, SAVED_STEP_INFO)
        #ops to save step info
        _STEP_INFO_PH = tf.placeholder(dtype=tf.float32, shape=[], name="saved_step_info_placeholder")
        _ASSIGN_STEP_INFO = SAVED_STEP_INFO.assign(_STEP_INFO_PH)
    


_AUX_OPS = "__aux_ops__" 
SAVE_VARS = "__save_vars__"


'''
return the average of the original tensor scalar value
adds the increment op to _AUX_OPS collection, run these ops
to add the current value to the running average
every time the average tensor is evaluated it resets the count
'''
def average_tracker(orig_val):
    total = tf.Variable(0.0, trainable=False)
    batches = tf.Variable(0.0, trainable=False)
    avg_dummy = tf.Variable(0.0, trainable=False)
    
    inc_op = tf.group(tf.assign(total, total+orig_val), tf.assign(batches, batches + 1))
    tf.add_to_collection(_AUX_OPS, inc_op)

    with tf.control_dependencies([inc_op]):
        assign_dmmy = tf.assign(avg_dummy, total / batches)
        with tf.control_dependencies([inc_op, assign_dmmy]):
            rs = tf.group(tf.assign(total, 0.0), tf.assign(batches, 0.0))
        
            with tf.control_dependencies([inc_op, assign_dmmy, rs]):
                get_dmmy = tf.identity(avg_dummy)
                average = tf.identity(get_dmmy) #total / batches
            
    return average
   



'''
convenience function to build tf variables
adds them to save vars collection, and adds histogram summaries if specified
(unless we're reusing variables)

TODO: need a more robust solution for checking reuse
'''
def get_variable(name, shape, dtype, initializer, trainable, save_var):
    v = tf.get_variable(name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable)
    if not tf.get_variable_scope().reuse:
        if save_var:
            tf.add_to_collection(SAVE_VARS, v)
    return v

'''print trainable variables and add any to savve vars that arent there already'''
def _check_save_and_train_vars():
    save_vars = tf.get_collection(SAVE_VARS)
    print("Trainable Variables:")
    for v in tf.trainable_variables():
        print('\t{0} {1}'.format(v.name, v.shape))
        if not (v in save_vars):
            tf.add_to_collection(SAVE_VARS, v)
            print("\t^^^^^^NOT IN SAVE VARS!!!!^^^^^")

    print("Save Variables Not In Train Vars:")
    for v in save_vars:
        if not (v in tf.trainable_variables()):
            print('\t{0} {1}'.format(v.name, v.shape))