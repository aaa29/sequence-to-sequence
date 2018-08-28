#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 10:20:39 2018

@author: amine
"""

import tensorflow as tf




def fully_conncected(inputs , W, B, activation=None):
    

    

    
    temp = tf.placeholder(tf.float32, shape=[None, W.shape[1]])
    x_t0 = tf.zeros_like(temp)
    
    def fn_loop(x_t0, x_t):
        
        if activation == 'softmax':
            return tf.nn.softmax(tf.matmul(x_t, W) + B)
        else:
            return tf.matmul(x_t, W) + B
            
    
    return tf.scan(fn = fn_loop, elems = inputs, initializer= x_t0)









class SimpleRnn:
    
    def __init__(self, num_units):
        self.num_units = num_units
        
        
    def rnn(self, x, initial_state= None, train= False):
        
        
        batch_size = int(x.shape[0])
        input_size = int(x.shape[2])
        state_size = int(input_size+ self.num_units)
        
        
        if train:
            self.rnn_W = tf.Variable(tf.random_uniform((state_size, self.num_units)), name= 'rnn_W')
            self.rnn_B = tf.Variable(tf.zeros((1,self.num_units)), name= 'rnn_B')
        
            
        rnn_batch = tf.transpose(x, (1, 0, 2))
        
        
               
        if initial_state == None:
            x_t0 = tf.zeros((batch_size, self.num_units))
        else:
            x_t0 = initial_state

        
        
        
        def rnn_loop(x_t0, x_t):
            
            x_t = tf.reshape(x_t, (batch_size,input_size))
            state = tf.concat([x_t0, x_t], axis=1)
            
            
            return tf.tanh(tf.matmul(state, self.rnn_W) + self.rnn_B)
       
        
        outputs = tf.scan(fn = rnn_loop,
                          elems = rnn_batch,
                          initializer = x_t0)
        
      

        
        

        return outputs, tf.gather(outputs, 4)
    









class Encoder:
    
    def __init__(self, num_units):
        #params init
        self.encoder_rnn = SimpleRnn(num_units)     
        
    def encode(self,x, train=False):
        #encoding
        
        _, encoder_last_state= self.encoder_rnn.rnn(x, train= train)
            
        return encoder_last_state
    
    





    
    
class Decoder:
    
    def __init__(self, num_units, vocab_size):
        #params init
        self.decoder_rnn = SimpleRnn(num_units)             
        self.W = tf.Variable(tf.random_uniform((num_units, vocab_size)))
        self.B = tf.Variable(tf.random_uniform((1, vocab_size)))
            
            
             
            
    
    def decode(self, batch_size, vocab_size, sequence_size, init_state, train= False, x=None):
        #decoder
        
        
        if train:
            
            decoder_outputs, _= self.decoder_rnn.rnn(x, initial_state = init_state, train = train)
            
            decoder_outputs = tf.transpose(decoder_outputs, (1, 0, 2))
            
            
            
            logits = fully_conncected(decoder_outputs, self.W, self.B)
            
            
            return logits
        
        else:
            start = tf.fill((batch_size,1), vocab_size-1)
            start = tf.one_hot(start, vocab_size)
            
 
 
            
            def decoding_loop(current_input, state):
             
            
                decoder_outputs, decoder_last_state= self.decoder_rnn.rnn(current_input, initial_state = init_state, train = train)
                
                            
                decoder_outputs = tf.transpose(decoder_outputs, (1, 0, 2))
                
                
                
                logits = fully_conncected(decoder_outputs, self.W, self.B)
                
                
                
                if current_input.shape[1] == sequence_size:
                    return logits

                else: 
                    logits = tf.gather(logits, logits.shape[1]-1, axis=1)
                    
                    
                    index = tf.argmax(logits, 1)
                    index = tf.reshape(index, (batch_size, 1))
                    index = tf.one_hot(index, vocab_size)
                    
                    next_input = tf.concat([current_input,index], axis=1)
                    
                    print('nexttttttttttt', next_input.shape)
                    
            
                    return decoding_loop(next_input, decoder_last_state)
                
                
            return decoding_loop(start, init_state)
        
        


class Model:
    
    def __init__(self, params):
        
        self.batch_size = params['batch_size']
        self.max_len_s = params['max_len_s']
        self.max_len_t = params['max_len_t']
        self.vocab_size_s = params['vocab_size_s']
        self.vocab_size_t = params['vocab_size_t']
        self.num_units = params['num_units']
        self.epochs = params['epochs']
        
        self.x_placeholder = tf.placeholder(tf.float32, 
                                            (self.batch_size, self.max_len_s , self.vocab_size_s))
        
        self.o_placeholder = tf.placeholder(tf.float32, 
                                            (self.batch_size, None, self.vocab_size_t))
        
        self.y_placeholder = tf.placeholder(tf.int32, 
                                            (self.batch_size, None, self.vocab_size_t))
        
        #initilize the params of the model
        self.encoder = Encoder(self.num_units)
        self.decoder = Decoder(self.num_units, self.vocab_size_t)
    
    
    def train(self):
        
        #encoder the source
        encoder_last_state = self.encoder.encode(self.x_placeholder, train = True)
        
        
        #decode
        logits = self.decoder.decode(self.batch_size,
                                     self.vocab_size_t,
                                     self.max_len_t,
                                     encoder_last_state,
                                     train = True,
                                     x = self.o_placeholder)
        
        
        #calculate the loss
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = self.y_placeholder))
        train_step = tf.train.AdamOptimizer().minimize(loss)
        
        return loss, train_step, self.x_placeholder, self.o_placeholder, self.y_placeholder
    
    
    
    
    
    def greedy_decode(self):
        
        encoder_last_state = self.encoder.encode(self.x_placeholder)
        
        
        
        #decode
        logits = self.decoder.decode(self.batch_size,
                                     self.vocab_size_t,
                                     self.max_len_t,
                                     encoder_last_state)
        
        print('shapes', logits.shape, self.y_placeholder.shape)
        
        #calculate the loss
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = self.y_placeholder))
        
        return loss, logits, self.x_placeholder, self.y_placeholder
        
        
        
        
       