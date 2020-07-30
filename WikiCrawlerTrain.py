import os
# suppress info logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import wikipedia_utils
import numpy as np 
import time
import char_dictionary
from embedding_layer import Embeddings
from rnn import dynamic_rnn
import cells
from dense_layer import DenseLayer
import graph_utils
import checkpoints
import sys
    
def log(msg, bold=False):
    sys.stdout.write(msg)
    sys.stdout.flush()

#batch size msut be constant when saving rnn states between runs
batch_size = 1

#to not request too many pages at once
min_request_time = 1

train_seq_length = 16

#print model output from this starting text during training
debug_string_starter = "What's going to "
debug_length = 256

learn_rate=1e-4
momentum=0.9
gradient_clip=5.0

checkpoint_load_dir = 'Checkpoints/'
checkpoint_save_path = checkpoint_load_dir + 'WikiCrawler'

'''DATA'''
max_length = 0
last_time_loaded = 0
text_data = None

def load_pages_batch():
    global max_length
    global last_time_loaded
    global text_data

    #if we jsut loaded, restart the current batch again
    if time.time() - last_time_loaded < min_request_time:
        print("\nRESTARTING CURRENT WIKI BATCH\n")
        return

    #if enough time has passed, update teh dataset with new articles
    last_time_loaded = time.time()
    print("\nLOADING NEW WIKI BATCH\n")
    
    #get text from the articles
    page_texts = wikipedia_utils.get_random_pages(page_count=batch_size)
    
    #encode strings
    encoded_texts = [char_dictionary.encode_string(t['text']) for t in page_texts]

    #pad
    ls = [len(t) for t in encoded_texts]
    max_length = np.max(ls)
    text_data = np.zeros([batch_size, max_length], np.int32)
    for i in range(batch_size):
        text_data[i, :ls[i]] = encoded_texts[i]

with tf.Graph().as_default():      
    #initialize global step, etc...
    graph_utils.initialize_graph_constants(training_mode=True)

    '''MODEL'''
    seq_in_ph = tf.placeholder(shape=[batch_size, None], dtype=tf.int32, name='sequence_in')
    temperature = tf.placeholder_with_default(0.0, (), name='temperature')
    
    '''
    sample sequence---

    x = s a m p l e s e q u e n c e - - 
    y = a m p l e s e q u e n c e - - -
    lw= 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0   <- loss weights
    '''
    x_data_t = seq_in_ph[:, :-1]
    targets_t = seq_in_ph[:, 1:] #inputs shifted over in time dimension
   
    #embed inputs
    embeddings = Embeddings('embeddings', char_dictionary.vocab_size, embedding_size=256)
    seq_in = embeddings(x_data_t)

    cell = cells.MultiRNNCell( [ cells.LSTMSavedState(256, batch_size) for i in range(2) ] )

    cell_reset_op = cell.reset_state()
    
    output = dynamic_rnn(seq_in, cell, batch_size)
    
    #logits
    logits = DenseLayer("logits", char_dictionary.vocab_size)(output)

    #output
    softmax = tf.nn.softmax(logits)
    
    '''
    add some noise to the output so it's not the same ALL teh time
    '''
    # noise scaled by temperature (-1,+1) * temp
    noise = tf.random_uniform(tf.shape(softmax), minval=-temperature, maxval=temperature, dtype=tf.float32)
    # if the model is sure it's not a certain class, noise affects it less (might not work when model is well trained...)
    noise = noise * softmax

    # offset softmax values by the noise
    generated = tf.argmax(softmax + noise, -1) 

    '''LOSS'''
    cross_entorpy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets_t)
    
    #no loss at end of sequences in batch (Padding)
    loss_weights_t = tf.clip_by_value(tf.cast(targets_t, tf.float32), 0.0, 1.0)

    cross_entorpy = cross_entorpy * loss_weights_t
    loss = tf.reduce_mean(cross_entorpy)
    
    #track loss average every 100 steps
    avg_loss_op = graph_utils.average_tracker(loss)

    '''MINIMIZE LOSS OP'''
    with tf.variable_scope('training'):
        o = tf.train.RMSPropOptimizer(learn_rate, momentum=momentum)
        gvs = o.compute_gradients(loss, var_list=tf.trainable_variables())

        '''clip gradients'''
        gradients, variables = zip(*gvs)
        gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip)
        capped_gvs = zip(gradients, variables)
        
        #optimize
        apply_gradients = o.apply_gradients(capped_gvs)
        

    graph_utils._check_save_and_train_vars()
    
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        
        orig_step, saved_step, best_loss = 0, 0, 9999999999999
        
        #maybe load checkpoint
        if checkpoints.load_checkpoint(sess, checkpoint_load_dir, filter_vars_not_in_graph=False):
            orig_step, best_loss = sess.run([graph_utils.GLOBAL_STEP, graph_utils.SAVED_STEP_INFO])
            saved_step = orig_step

        #make saver for saving variables
        saver = tf.train.Saver(tf.get_collection(graph_utils.SAVE_VARS))
        
        def debug_model(temp):
            sess.run(cell_reset_op)
            
            #start list with starter tokens
            allgen = char_dictionary.encode_string(debug_string_starter)
            
            #generate one token at a time
            for _ in range(debug_length):
                allgen.append(sess.run(generated, feed_dict={ 
                    seq_in_ph: [[allgen[-1], allgen[-1]],] * batch_size, 
                    temperature:temp 
                })[0, 0])
            
            #decode and print
            print ('\nTEMPERATURE: {}'.format(temp) + '\n' + char_dictionary.decode_tokens(allgen))

            sess.run(cell_reset_op)
        
        l = 0

        # as training goes on, keep the state around for longer
        state_reset_iterations = 1
        
        def get_next_batch():
            global l
            global state_reset_iterations

            e = min(l+train_seq_length, max_length)

            # if we're dne with the batch or theres only 1 spot left (need at least two)
            # reload pages or load new pages and start form 0 again
            if (l >= e and e == max_length) or ( e - l == 1):
                l = 0
                
                #print at various sampling temperatures
                for temp in [0.0,0.5,1.0]:
                    debug_model(temp)
                
                load_pages_batch()

                state_reset_iterations += 1
                
                return get_next_batch()
            
            #get the next batch
            x = text_data[:,l:e]
            l += max((e-l) - 1, 1)
            return x
            
        i = 0
        while True:
            if i % state_reset_iterations == 0:
                sess.run(cell_reset_op)
            
            do_debugs = i % 100 == 0

            run_tensors = {
                '_': tf.get_collection(graph_utils._AUX_OPS),
                '__': graph_utils._INCREMENT_STEP,
                'optimizer': apply_gradients,
                'loss': loss
            }

            if do_debugs:    
                run_tensors['avg_loss'] = avg_loss_op
                #get input and output values to print to console
                run_tensors['rnn_in_out'] = (x_data_t, generated)

            #run the session (optimize gradients...)
            run_return = sess.run(run_tensors, feed_dict={
                #prepare feed dict for training iteration
                seq_in_ph: get_next_batch()
            })

            log("\rSaved Step {0} :: Global Step: {1} :: Read Index {2}/{3} :: State Reset Iterations: {4} :: Loss {5:.3}".format(
                saved_step, orig_step + i, l, max_length, state_reset_iterations, run_return['loss'])
            )
            
            if do_debugs:
                avg_loss = run_return['avg_loss']
                print("\nAVG Loss: {}".format(avg_loss))

                x, o = run_return['rnn_in_out']
                if o[0][0] != 0:
                    print ('O: ' + char_dictionary.decode_tokens(o[0]) + '\nI: ' + char_dictionary.decode_tokens(x[0]))
                
                #save if there was improvement
                # if avg_loss < best_loss:
                if True:
                    saved_step = orig_step + i
                    best_loss = avg_loss
                    graph_utils.save_step_info(sess, step_info=avg_loss)
                    print("\nSaving checkpoint....")
                    saver.save(sess, checkpoint_save_path)#, global_step=graph_step)
                    print("Saved!")
            i += 1
