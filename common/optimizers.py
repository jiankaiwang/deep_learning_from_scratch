#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jiankaiwang
"""

import numpy as np

class SGD:
    def __init__(self, lr=1e-2):
        self.lr = lr
    
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
            
class Momentum:
    def __init__(self, lr=1e-2, momentum=9e-1):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, params, grads):
        if self.v == None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key, val in params.items():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]
            
class AdaGrad:
    def __init__(self, lr=1e-2):
        self.lr = lr
        self.h = None
    
    def update(self, params, grads):
        if self.h == None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                
        for key, val in params.items():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr / np.sqrt(self.h[key] + 1e-7) * grads[key]

class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999):
        self.m = None
        self.v = None
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
    
    def update(self, params, grads):
        if self.m == None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
                
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)
        
        for key, val in params.items():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * (self.m[key] / (np.sqrt(self.v[key] + 1e-7)) )       
