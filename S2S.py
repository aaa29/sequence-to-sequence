

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from mynn2_2 import Model
from data_utils import x_y






#define the params:

params = {'batch_size' : 128,
          'max_len_s' : 5,
          'max_len_t' : 5,
          'vocab_size_s': 10,
          'vocab_size_t': 10,
          'num_units' : 8,
          'epochs' : 500
        }







x , y = x_y(params['max_len_s'], 
            params['vocab_size_s'])



output_size = y.shape[1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



def gen_batch(x, y, batch_size):
    
    num_batches = int(x.shape[0]/batch_size)
    
    for idx in range(num_batches):
        
        start_idx = idx * batch_size
        end_idx = start_idx + batch_size
        
        batch_x = x[start_idx:end_idx]
        batch_y = y[start_idx:end_idx]
        
        if batch_x.shape[0] == batch_size:
            yield batch_x, batch_y
        else:
            yield None
        
        





model = Model(params)

loss, train_step, x_placeholder, o_placeholder, y_placeholder = model.train()

loss_v, logits_v, x_placeholder_v, y_placeholder_v = model.greedy_decode()



#prediction = tf.arg_max(logits, 1)
##
##true_pred = tf.equal(tf.arg_max(logits,1), tf.arg_max(y_placeholder,1))
##accuracy = tf.reduce_mean(tf.cast(true_pred, tf.float32))






    




with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    num_batches = int(x_train.shape[0]/params['batch_size'])
    
    
    for epoch in range(params['epochs']):
        batch_generator = gen_batch(x_train, y_train , params['batch_size'])
        total_loss = []
        for _ in range(num_batches):
            batch = next(batch_generator)
            if batch!= None:
                feed_dict = {x_placeholder : batch[0],
                             o_placeholder : batch[1][:,:-1,:],
                             y_placeholder : batch[1][:,1:,:]}
                
             
                
                _loss,_ = sess.run([loss, train_step], feed_dict= feed_dict)
                total_loss += [_loss]
            

        print("accuracy for epoch {} : {}".format(epoch,np.mean(total_loss)))
    
    
    #validation:
    
    num_batches = int(x_test.shape[0]/params['batch_size'])
    
    batch_generator = gen_batch(x_test, y_test , params['batch_size'])
    total_loss = []
    results = []
    
    
    for _ in range(num_batches):
        batch = next(batch_generator)
        if batch!= None:
            feed_dict = {x_placeholder_v : batch[0],
                         y_placeholder_v : batch[1][:,1:,:]}
            
            
            
            _loss, _logits_v= sess.run([loss_v, logits_v], feed_dict= feed_dict)
            total_loss += [_loss]
            
            results+= [(np.argmax(_logits_v,2), np.argmax(batch[1][:,1:,:],2))]
    print("test accuracy : {}".format(np.mean(total_loss)))
#        

    

    

    


        



    
    
    





