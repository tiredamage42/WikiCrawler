'''
tensorflow graph utils

global step, training bool, any saved graph variables that arent trainable, etc...
'''

import tensorflow as tf
tf.disable_v2_behavior()

IS_TRAINING = False
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
    #if SAVED_STEP_INFO is None:
    #    return
    fd = {_STEP_INFO_PH: step_info}
    session.run(_ASSIGN_STEP_INFO, fd)

'''
initialize variables and ops
'''
def initialize_graph_constants(training_mode, saved_info_shape=None):
    global GLOBAL_STEP
    global SAVED_STEP_INFO
    global IS_TRAINING
    global _INCREMENT_STEP
    global _STEP_INFO_PH
    global _ASSIGN_STEP_INFO
    IS_TRAINING = False
    _INCREMENT_STEP = None
    _STEP_INFO_PH = None
    _ASSIGN_STEP_INFO = None
    
    GLOBAL_STEP = tf.Variable(0, dtype=tf.int64, name='global_step', trainable=False)
    tf.add_to_collection(SAVE_VARS, GLOBAL_STEP)
    
    if training_mode:
        # op to add 1 to gloal step
        _INCREMENT_STEP = tf.assign(GLOBAL_STEP, GLOBAL_STEP + 1)
        
        IS_TRAINING = tf.placeholder(tf.bool, name="is_training")
        
        if saved_info_shape is not None:    
            #any custom info variable
            SAVED_STEP_INFO = tf.Variable(tf.fill(saved_info_shape, 0.0), dtype=tf.float32, name='saved_step_info', trainable=False)
            tf.add_to_collection(SAVE_VARS, SAVED_STEP_INFO)
            #ops to save step info
            _STEP_INFO_PH = tf.placeholder(dtype=tf.float32, shape=saved_info_shape, name="saved_step_info_placeholder")
            _ASSIGN_STEP_INFO = SAVED_STEP_INFO.assign(_STEP_INFO_PH)
        else:
            SAVED_STEP_INFO = None
    


        
SAVE_VARS = "__save_vars__"
TRAINING_SUMMARIES = "__train_summaries__"
VALIDATION_SUMMARIES = '__validation_summaries__'
DEBUGGING_SUMMARIES = '__debug_summaries__'
TEST_SUMMARIES = '__test_summaries__'


def _check_filters_vars(f, var_name):
    if f is None:
        return True
    for fil in f:
        if fil in var_name:
            return True
    return False
'''
get a list of trainable variables that contain the filters
if filters is None returns all trainable variables
'''
def get_trainable_vars(filters=None):
    return [var for var in tf.trainable_variables() if _check_filters_vars(filters, var.name)]

'''add to multiple collections'''
def add_to_collections(collections, tensor):
    for collection in collections:
        tf.add_to_collection(collection, tensor)

'''
convenience function to build tf variables
adds them to save vars collection, and adds histogram summaries if specified
(unless we're reusing variables)

TODO: need a more robust solution for checking reuse
'''
def get_variable(name, shape, dtype, initializer, trainable, save_var, add_histograms):
    v = tf.get_variable(name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable)
    if not tf.get_variable_scope().reuse:
        if save_var:
            tf.add_to_collection(SAVE_VARS, v)
        if add_histograms:
            tf.add_to_collection(TRAINING_SUMMARIES, tf.summary.histogram(v.name, v))
    return v

'''print trainable variables and add any to savve vars that arent there already'''
def _check_save_and_train_vars():
    save_vars = tf.get_collection(SAVE_VARS)
    print("Trainable Variables:")
    for v in tf.trainable_variables():
        print('\t{0} {1}'.format(v.name, v.shape))
        if not (v in save_vars):
            tf.add_to_collection(SAVE_VARS, v)
            #print("\t^^^^^^NOT IN SAVE VARS!!!!^^^^^")

    print("Save Variables Not In Train Vars:")
    for v in save_vars:
        if not (v in tf.trainable_variables()):
            print('\t{0} {1}'.format(v.name, v.shape))