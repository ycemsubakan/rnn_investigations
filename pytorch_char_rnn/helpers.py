# https://github.com/spro/char-rnn.pytorch

import unidecode
import string
import random
import time
import math
import torch
import numpy as np

# Random hyperparamter configuration

def generate_random_hyperparams(lr_min, lr_max, K_min, K_max, num_layers_min, num_layers_max):
    lr_exp = np.random.uniform(lr_min, lr_max)
    lr = 10**(lr_exp)
    K = np.random.choice(np.arange(K_min, K_max+1),1)[0]
    num_layers = np.random.choice(np.arange(num_layers_min, num_layers_max + 1),1)[0]
    return lr, K, num_layers

# Reading and un-unicode-encoding data

all_characters = string.printable
n_characters = len(all_characters)

def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)

# Turning a string into a tensor

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

