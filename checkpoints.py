import tensorflow.compat.v1 as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
tf.disable_v2_behavior()

'''print a checkpoint file'''
def checkpoint_file_as_string(file_name):
    return pywrap_tensorflow.NewCheckpointReader(file_name).debug_string().decode("utf-8")

'''
gets the names and shapes of variables in a checkpoint file

returns list of tuples:
    (var name, var shape, var value)
'''
def get_variables_in_checkpoint_file(file_name):
    cp_vars = []
    print("Building Checkpoint Reader...")
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    print("Getting Var2Shape Map...")
    var_to_shape_map = reader.get_variable_to_shape_map()
    print("Assembling Variables List...")
    for tensor_name in sorted(var_to_shape_map):
        var_value = reader.get_tensor(tensor_name)
        cp_vars.append( ( tensor_name, np.shape(var_value), var_value ) )
    return cp_vars
    

'''
get references to the variables int eh graph corresponding to
the variables in the checkpoint file

filter_vars_not_in_graph:
    allow discrepancies between the checkpoint and the graphs
    only tensors with matching names/shapes will be loaded correctly
'''
def get_graph_vars(tensors_in_checkpoint, filter_vars_not_in_graph):
    
    full_var_list = []
    
    for checkpoint_var in tensors_in_checkpoint:
        tensor_name, tensor_shape, tensor_val = checkpoint_var
        try:
            # get reference to the var in the graph
            graph_tensor = tf.get_default_graph().get_tensor_by_name(tensor_name+":0")
            # shape of the variable in graph already
            graph_tensor_shape = graph_tensor.get_shape().as_list()
            
            #if the shape is different, raise an error unless we're filtering those variables out
            if graph_tensor_shape != list(tensor_shape):
                print(tensor_name)
                print("\tShape Mismatch (Graph/Checkpoint): {0}/{1}".format(graph_tensor_shape, tensor_shape) )
                if not filter_vars_not_in_graph:
                    raise ValueError('_')
            else:
                full_var_list.append(graph_tensor)
        except:
            print('Not found: '+ tensor_name)
            if not filter_vars_not_in_graph:
                raise ValueError('_')
            
    return full_var_list

'''
Load the latest checkpoint file in a directory
'''
def load_checkpoint(session, checkpoint_directory, filter_vars_not_in_graph=True):

    print("\n\nTrying to restore last checkpoint ...")
    try:
        latest_ckpt_path = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_directory)
        print("\nFound checkpoint:\n" + latest_ckpt_path)
        print("\n\nCollecting variables in checkpoint...")
        tensors_in_cp  = get_variables_in_checkpoint_file(latest_ckpt_path)
        print("\n\nGetting graph vars...")
        if filter_vars_not_in_graph:
            print("\n\nFiltering variables that aren't in the current graph...")
        tensors_to_load = get_graph_vars(tensors_in_cp, filter_vars_not_in_graph)
        print("Loading checkpoint...")
        tf.train.Saver(tensors_to_load).restore(session, latest_ckpt_path)    
        print("Loaded\n{0}".format(latest_ckpt_path))
        return True
    except:
        print("Failed to restore checkpoint. Initializing variables instead.")
        return False
