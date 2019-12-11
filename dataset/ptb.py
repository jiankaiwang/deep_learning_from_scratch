# -*- coding: utf-8 -*-
"""
@description: ptb dataset loader
@author: JianKai Wang
@reference:
  https://www.kaggle.com/myqrizzo/tf-tutorial-ptb-dataset
  https://raw.githubusercontent.com/tomsercu/lstm/master/data/
"""

import os
import urllib.request
import pickle
import numpy as np

# In[]

url_base = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/'
key_file = {'train':'ptb.train.txt', 'test':'ptb.test.txt', 'valid':'ptb.valid.txt'}
save_file = {'train':'ptb.train.npy', 'test':'ptb.test.npy', 'valid':'ptb.valid.npy'}
vocab_file = 'ptb.vocab.pkl'

dataset_dir = os.path.dirname(os.path.abspath(__file__))

# In[]

def __download(file_name):
  file_path = os.path.join(dataset_dir, file_name)
  if os.path.exists(file_path): return
  print("Downloading {} ...".format(file_name))
  
  try:
    urllib.request.urlretrieve(url_base + file_name, file_path)
  except urllib.error.URLError:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib.request.urlretrieve(url_base + file_name, file_path)
    
  print("Downloading dataset was done.")

# In[]

def load_vocab():
  vocab_path = os.path.join(dataset_dir, vocab_file)
  
  if os.path.exists(vocab_path):
    with open(vocab_path, 'rb') as fin:
      word_to_id, id_to_word = pickle.load(fin)
    return word_to_id, id_to_word
  
  word_to_id = {}
  id_to_word = {}
  data_type = "train"
  file_name = key_file[data_type]
  file_path = os.path.join(dataset_dir, file_name)
  
  __download(file_name)
  words = open(file_path).read().replace('\n', '<eos>').strip().split()
  
  for i, word in enumerate(words):
    if word not in word_to_id:
      tmp_id = len(word_to_id)
      word_to_id[word] = tmp_id
      id_to_word[tmp_id] = word

  with open(vocab_path, "wb") as fout:
    pickle.dump((word_to_id, id_to_word), fout)
    
  return word_to_id, id_to_word

# In[]

def load_data(data_type='train'):
  if data_type == "val": data_type = "valid"
  save_path = os.path.join(dataset_dir, save_file[data_type])
  
  word_to_id, id_to_word = load_vocab()
  
  if os.path.exists(save_path):
    corpus = np.load(save_path)
    return corpus, word_to_id, id_to_word
  
  file_name = key_file[data_type]
  file_path = os.path.join(dataset_dir, file_name)
  __download(file_name)
  
  words = open(file_path).read().replace('\n', '<eos>').strip().split()
  corpus = np.array([word_to_id[w] for w in words])
  
  np.save(save_path, corpus)
  return corpus, word_to_id, id_to_word

# In[]

if __name__ == "__main__":
  for data_type in ('train', 'val', 'test'):
    load_data(data_type)



