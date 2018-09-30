#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jiankaiwang
"""

import numpy as np

## activation functions

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (1 - np.exp(-x)) / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)
    
def softmax(x):
    # we have adjusted this function to meet batch data
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    else:
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

## loss functions

def cross_entropy_error(y, t):
    # we have adjusted this function to meet batch data
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

## numerical gradient

def __numerical_gradient_body(f, x):    
    h = 1e-4
    grad = np.zeros_like(x)
    
    for i in range(0, x.size, 1):
        tmp_val = x[i]
        
        # calculate f(x+h)
        x[i] = tmp_val + h
        fxh1 = f(x)
        
        # calculate f(x-h)
        x[i] = tmp_val - h
        fxh2 = f(x)
        
        grad[i] = (fxh1 - fxh2) / (2*h)
        
        # restore the value
        x[i] = tmp_val
        
    return grad

def numerical_gradient(f, X):
    if X.ndim == 1:
        return __numerical_gradient_body(f, X)
    else:
        grad = np.zeros_like(X)
        
        dim,ttl = X.shape
        
        for i in range(ttl):
            grad[:,i] = __numerical_gradient_body(f, X[:,i])
        
        return grad