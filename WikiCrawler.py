import os
# suppress info logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import char_dictionary
from embedding_layer import Embeddings
from rnn import dynamic_rnn
import cells
from dense_layer import DenseLayer
import checkpoints
    
batch_size = 1

checkpoint_load_dir = 'Checkpoints/'

with tf.Graph().as_default():      
    
    '''MODEL'''
    seq_in_ph = tf.placeholder(shape=[batch_size, None], dtype=tf.int32, name='sequence_in')
    temperature = tf.placeholder_with_default(0.0, (), name='temperature')
    
    '''
    sample sequence---
    x = s a m p l e s e q u e n c e - - 
    '''
    #embed inputs
    embeddings = Embeddings('embeddings', char_dictionary.vocab_size, embedding_size=256)
    seq_in = embeddings(seq_in_ph)

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
    
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        
        #maybe load checkpoint
        if checkpoints.load_checkpoint(sess, checkpoint_load_dir, filter_vars_not_in_graph=False):
            
            def debug_model(temp, start_string, length):
                sess.run(cell_reset_op)
                
                #start list with starter tokens
                allgen = char_dictionary.encode_string(start_string)

                # set state up with starting string
                for i in range(len(allgen) - 1):
                    sess.run(generated, feed_dict={ 
                        seq_in_ph: [[allgen[i]]], 
                        temperature: 0 
                    })
                
                #generate one token at a time
                for _ in range(length):
                    gen_char = sess.run(generated, feed_dict={ 
                        seq_in_ph: [[allgen[-1]]], 
                        temperature:temp 
                    })[0, 0]
                    allgen.append(gen_char)
                
                #decode and print
                print ('\n\n' + char_dictionary.decode_tokens(allgen))


            phase = 0
            temp = 0
            length = 0
            start_string = ''

            def wants_quit(response):
                return isinstance(response, str) and (response == 'q' or response == 'Q')

            while True:
                if phase == 0:
                    temp = input('Select a temperature [ 0, 1 ] (press q to quit): ')

                    if wants_quit(temp):
                        break
                    
                    if (not isinstance(temp, float)):
                        print ("Temperature must be a number!")
                    else:
                        temp = min(1.0, max(0.0, temp))
                        phase += 1

                elif phase == 1:
                    length = input('Select a length to generate (press q to quit): ')
                    if wants_quit(length):
                        break
                    
                    if (not isinstance(length, int)):
                        print ("Length must be a number!")
                    else:
                        length = max(0, length)
                        phase += 1
                
                elif phase == 2:
                    start_string = input('Select a starting string of text (press q to quit): ')
                    if wants_quit(length):
                        break
                    phase += 1

                elif phase == 3:
                    debug_model(temp, start_string, length)
                    phase = 0
                    
                    

                
