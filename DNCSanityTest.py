'''
script to make sure my DNC implementation 
can at least repeat back a random series in order

visualizes the read / write weighting and usage vectors during debugging
'''

import os
# suppress info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow.compat.v1 as tf
from rnn.copy_sequence import copy_sequence_toy_data_fn     
from dnc import cells, visualization, layers, graph_utils
from rnn import rnn
import matplotlib.pyplot as plt
import log_utils

tf.disable_v2_behavior()

vocab_size = 6

'''DATASET'''
# copy repeat task (binary sequences)
inputs_fn = copy_sequence_toy_data_fn(
    vocab_size=vocab_size, 
    seq_length_range_train=[4,4], series_length_range_train=[2,4], 
    seq_length_range_val=[4,4], series_length_range_val=[4,4], 
    num_validation_samples=100, error_pad_steps = 2
)

'''MODEL'''
def model_fn (data_in):
    seq_in = data_in['sequence']
    targets = data_in['target']

    #build the dnc cell
    cell = cells.DNCCell(cells.StatelessCell("linear", features=64), memory_size=16, word_size=16, num_reads=1, num_writes=1)

    output, _, final_loop_state = rnn.dynamic_rnn(
        seq_in, cell, 
        #for visualizing memory views
        loop_state_fn=visualization.loop_state_fn, 
        initial_loop_state=visualization.initial_loop_state()
    )
    
    logits_fn = layers.Dense("logits", vocab_size, activation=None)
    logits = logits_fn(output)
    preds = tf.sigmoid(logits)

    with tf.variable_scope('Loss'):
        loss =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets))

    #return dictionary of tensors to keep track of
    args_dict = {}

    with tf.variable_scope('train'):
        o = tf.train.RMSPropOptimizer(1e-4, momentum=0.9)
        
        gvs = o.compute_gradients(loss, var_list=tf.trainable_variables())

        '''clip gradients'''
        gradients, variables = zip(*gvs)
        gradients, _ = tf.clip_by_global_norm(gradients, 2.0)
        capped_gvs = zip(gradients, variables)
        
        args_dict['optimizer'] = o.apply_gradients(capped_gvs)

    #track loss average every 100 steps
    args_dict['avg_loss'] = graph_utils.average_tracker(loss)

    #track loop state in tensorboard
    args_dict['mem_view'] = visualization.assemble_mem_view(final_loop_state, 0, 0, [seq_in, preds, targets], None)
    
    return args_dict

'''TRAINING'''

def _plot_image (image, name):
    plt.imshow(image, interpolation='nearest', cmap='binary')
    plt.ylabel("Mem(R:Read G:Write)---Usage---Targets---Outputs---Inputs")
    # Remove ticks from the plot.
    plt.axes().set_xticks([])
    plt.axes().set_yticks([])
    
    directory = 'images'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    plt.savefig(os.path.join(directory, name) + '.png')
    plt.close()
    
def _train_iteration_dummy(session, model_args, data_args, train_ops):
    
    run_return = session.run(
        {
            #increment counters for averages ops
            '__aux__': tf.get_collection(graph_utils._AUX_OPS),
            'avg_loss': model_args['avg_loss']
        }, 
        feed_dict={
            #use handle in feed dict when using that data set
            data_args['handle_ph']: train_ops['handle']
        }
    )

    model_args['avg_train_loss'] = run_return['avg_loss']
    
    return model_args

# def _train_iteration(session, model_args, do_debugs, data_args, train_ops, iteration, iterations):
    
#     run_tensors = {
#         #increment counters for averages ops
#         '__aux__': tf.get_collection(graph_utils._AUX_OPS),
#         'optimizer': model_args['optimizer'],
#     }
            
#     if do_debugs:
#         run_tensors['avg_loss'] = model_args['avg_loss']

#     run_return = session.run(run_tensors, feed_dict={
#         #use handle in feed dict when using that data set
#         data_args['handle_ph']: train_ops['handle']
#     })

#     if do_debugs:
#         model_args['avg_train_loss'] = run_return['avg_loss']
    
#     log_utils.log("\rTraining Iteration: {0}/{1}".format(iteration, iterations))
#     return model_args

def _one_shot_loop(session, msg, data_args, model_args, data_args_g, mode, iteration):
    print('\n')#msg + '\n')
    
    session.run(data_args['Iterator'].initializer)
    sample_count = data_args['SampleCount']
    batch_size = data_args['BatchSize']

    max_batches = int(sample_count / batch_size) + (1 if sample_count % batch_size > 0 else 0)
    batch = 0
    while True:
        try:

            at_end = (batch == (max_batches - 1))

            run_tensors = {
                #increment counters for averages ops
                '__aux__': tf.get_collection(graph_utils._AUX_OPS)
            }

            if at_end:
                run_tensors['avg_validation_loss'] = model_args['avg_loss']
            
            if at_end and msg == 'Debugging':
                run_tensors['mem_view'] = model_args['mem_view']

            run_return = session.run(run_tensors, feed_dict={ data_args_g['handle_ph']: data_args['handle'] })

            if at_end:
                if msg == 'Validating':
                    model_args['avg_validation_loss'] = run_return['avg_validation_loss']
                
                elif msg == 'Debugging':
                    _plot_image (run_return['mem_view'][0], 'mem_view_{}'.format(iteration))
                    
            
            log_utils.log("\r" + msg + " Batch: {0}/{1}\t".format(batch+1, max_batches))
            batch += 1

        except (tf.errors.OutOfRangeError, StopIteration):
            print('\rDone ' + msg + '                                \n')
            break

    return model_args


def run_training(batch_size, iterations, inputs_fn, model_fn):
    
    with tf.Graph().as_default():      
        
        print ('Initializing Inputs...')
        samples_in, data_args = inputs_fn(batch_size=batch_size)
        
        train_ops = data_args.get("Training", None)
        val_ops = data_args.get("Validation", None)
        debug_ops = data_args.get("Debug", None)

        print ('Initializing Model...')
        model_args = model_fn(samples_in)
        
        print ('Populating Save Vars Collection...')
        graph_utils._check_save_and_train_vars()
        
        print ("Building Session...")
        with tf.Session() as sess:

            print ("Initializing Variables...")
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            
            print ("Getting Dataset Handles...")
            train_ops['handle'] = sess.run(train_ops['Iterator'].string_handle())
            val_ops['handle'] = sess.run(val_ops['Iterator'].string_handle())
            debug_ops['handle'] = sess.run(debug_ops['Iterator'].string_handle())
            
            print ('Running Dummy Train Iteration To Populate Train Loss...')
            model_args = _train_iteration_dummy(sess, model_args, data_args, train_ops)

            def validate_and_debug(iteration, model_args):
                model_args = _one_shot_loop(sess, "Validating", val_ops, model_args, data_args, 'validation', iteration)

                #print validation results
                msg = "===========================================================\n"
                msg += "\n|LOSS: T: {0:.3} V: {1:.3}\t| ".format(model_args['avg_train_loss'], model_args['avg_validation_loss']) + "Iteration {0}".format(iteration)       
                msg += "\n\n==========================================================="
                log_utils.log(msg, bold=True)

                model_args = _one_shot_loop(sess, "Debugging", debug_ops, model_args, data_args, 'debug', iteration)
                return model_args


            #validate at 0 iteration
            model_args = validate_and_debug(0, model_args)
                    
            print ('Starting Training Session...\n')
            for i in range(iterations):
                
                do_debugs = (i % 100 == 0 or i == iterations - 1) and i != 0

                # model_args = _train_iteration(sess, model_args, do_debugs, data_args, train_ops, i, iterations)

                run_tensors = {
                    #increment counters for averages ops
                    '__aux__': tf.get_collection(graph_utils._AUX_OPS),
                    'optimizer': model_args['optimizer'],
                }
                        
                if do_debugs:
                    run_tensors['avg_loss'] = model_args['avg_loss']

                run_return = sess.run(run_tensors, feed_dict={
                    #use handle in feed dict when using that data set
                    data_args['handle_ph']: train_ops['handle']
                })

                if do_debugs:
                    model_args['avg_train_loss'] = run_return['avg_loss']
                
                log_utils.log("\rTraining Iteration: {0}/{1}".format(i, iterations))


                if do_debugs:
                    model_args = validate_and_debug(i, model_args)
            
        
run_training(
    batch_size=1, iterations=20000, 
    inputs_fn=inputs_fn, 
    model_fn=model_fn
)
