#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:58:18 2019

@author: jiankaiwang
"""

import sys
sys.path.append("..")
import pickle
import numpy as np
from common.time_layers import TimeEmbedding, TimeLSTM, TimeAffine, TimeSoftmaxWithLoss

class Rnnlm:
  def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
    V, D, H = vocab_size, wordvec_size, hidden_size
    rn = np.random.randn
    
    # initializing weights
    embed_W = (rn(V, D) / 100).astype('f')
    lstm_Wx = (rn(D, 4*H) / np.sqrt(D)).astype('f')
    lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
    lstm_b = np.zeros(4*H).astype('f')
    affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
    affine_b = np.zeros(V).astype('f')
    
    # generate each layers
    self.layers = [
        TimeEmbedding(embed_W),
        TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
        TimeAffine(affine_W, affine_b)
    ]
    self.loss_layer = TimeSoftmaxWithLoss()
    self.lstm_layer = self.layers[1]
    
    # gather all weights and gradients
    self.params, self.grads = [], []
    for layer in self.layers:
      self.params += layer.params
      self.grads += layer.grads
      
  def predict(self, xs):
    for layer in self.layers:
      xs = layer.forward(xs)
    return xs
  
  def forward(self, xs, ts):
    score = self.predict(xs)
    loss = self.loss_layer.forward(score, ts)
    return loss
  
  def backward(self, dout=1):
    dout = self.loss_layer.backward(dout)
    for layer in reversed(self.layers):
      dout = layer.backward(dout)
    return dout
  
  def reset_state(self):
    self.lstm_layer.reset_state()
    
  def save_params(self, file_name="Rnnlm.pkl"):
    with open(file_name, "wb") as fout:
      pickle.dump(self.params, fout)
      
  def load_params(self, file_name="Rnnlm.pkl"):
    with open(file_name, "rb") as fin:
      self.params = pickle.load(fin)