#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 15:34:21 2018

@author: amine

data utils
"""

import numpy as np
from keras.utils import to_categorical
from faker import Faker





def fake_dates(rows = 10000):
    
    fake = Faker()
    dates = []
    for _ in range(rows):
        dates += [fake.date()]
        
        
    dates_1 = [date.split('-') for date in dates]
    targets_1 = [list(date.__reversed__()) for date in dates_1]
    
    return dates_1, targets_1




def data_convert(data, source=True):
    
    data_s = []
    
    for e in data:
        data_s += e
        
    data_s = list(set(data_s))
    
    maxlen = max([len(s) for s in data])
    
   
    
    
    word2num = { word:num for num, word in enumerate(data_s)}
    if source:
        word2num['<PAD>'] = len(word2num)
        num2word = {num:word for num,word in zip(word2num.values(), word2num.keys())}
        num_data = [[word2num['<PAD>']]*(maxlen-len(s))+[word2num[word] for word in s]
                                                                        for s in data]
    else:
        word2num['<GO>'] = len(word2num)
        num2word = {num:word for num,word in zip(word2num.values(), word2num.keys())}
        num_data = [[word2num['<GO>']]+[word2num[word] for word in s]
                                                       for s in data]
    
    vocab_size = len(word2num)
        
    
    return to_categorical(np.array(num_data), vocab_size), word2num, num2word, vocab_size




def x_y(cols, vocab_size, rows=10000):
    
    x = np.random.choice(vocab_size-1, (rows, cols))
    
 
    
    x2 = np.zeros((1,cols,vocab_size))
    x2i = np.zeros((1,cols+1,vocab_size))
    start = to_categorical(vocab_size-1, vocab_size)
    start = start.reshape(1, 1, start.shape[0])
    
    for xi in x:
        xi2 = to_categorical(xi, vocab_size)
        xi2 = xi2.reshape(1,cols,vocab_size)
        xi2i = np.concatenate((start, np.flip(xi2, axis=1)), axis=1)
        x2 = np.concatenate((x2,xi2))
        x2i = np.concatenate((x2i, xi2i))
    
    
    return x2[1:], x2i[1:]
