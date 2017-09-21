#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
from tqdm import tqdm
from helpers import *
from model import *
from generate import *
import pdb
import numpy as np
from matplotlib import pyplot as plt

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--model', type=str, default='vanilla_tanh')
argparser.add_argument('--n_epochs', type=int, default=300)
argparser.add_argument('--print_every', type=int, default=1)
argparser.add_argument('--K_min', type=int, default=50)
argparser.add_argument('--K_max', type=int, default=300)
argparser.add_argument('--num_layers_min', type=int, default=1)
argparser.add_argument('--num_layers_max', type=int, default=2)
argparser.add_argument('--lr_min', type=float, default=-4)
argparser.add_argument('--lr_max', type=float, default=-2)
argparser.add_argument('--chunk_len', type=int, default=200)
argparser.add_argument('--batch_size', type=int, default=100)
argparser.add_argument('--num_configs', type=int, default=60)
argparser.add_argument('--cuda', action='store_true')
args = argparser.parse_args()

def partition_text_file(file, chunk_len, pct_train):
    chunks = []
    tmp = ''
    for idx, c in enumerate(file):
        tmp += c
        if idx % chunk_len == 0:
            chunks.append(tmp)
            tmp = ''
    np.random.shuffle(chunks)
    train_chunks = chunks[0 : int(pct_train * len(chunks))]
    test_chunks = chunks[int(pct_train * len(chunks)) :]
    train_file = ''.join(train_chunks)
    test_file = ''.join(test_chunks)
    return train_file, test_file

def random_batch(file, chunk_len, batch_size):
    file_len = len(file)
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        try:
            start_index = random.randint(0, file_len - chunk_len)
            end_index = start_index + chunk_len + 1
            chunk = file[start_index:end_index]
            inp[bi] = char_tensor(chunk[:-1])
            target[bi] = char_tensor(chunk[1:])
        except:
            continue
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def train(inp, target):
    hidden = decoder.init_hidden(args.batch_size)
    if args.cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0
    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:,c])
    loss.backward()
    decoder_optimizer.step()
    return loss.data[0] / args.chunk_len

def test(inp, target):
    hidden = decoder.init_hidden(args.batch_size)
    loss = 0
    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:,c])
    return loss.data[0] / args.chunk_len

def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

"""
file, file_len = read_file(args.filename)
train_file, test_file = partition_text_file(file, args.chunk_len, 0.75)
train_file, vld_file = partition_text_file(train_file, args.chunk_len, 0.75)
pickle.dump(train_file, open('train_set.pk', 'wb'))
pickle.dump(vld_file, open('vld_set.pk', 'wb'))
pickle.dump(test_file, open('test_set.pk', 'wb'))
"""

train_file = pickle.load(open('train_set.pk','rb'))
vld_file = pickle.load(open('vld_set.pk','rb'))
test_file = pickle.load(open('test_set.pk','rb'))

if args.cuda:
    print("Using CUDA")

for configuration in range(args.num_configs):
    
    learning_rate, hidden_size, num_layers = generate_random_hyperparams(args.lr_min, 
                                                                         args.lr_max, 
                                                                         args.K_min, 
                                                                         args.K_max, 
                                                                         args.num_layers_min, 
                                                                         args.num_layers_max)
    print(learning_rate, hidden_size, num_layers)

    decoder = CharRNN(
        n_characters,
        hidden_size,
        n_characters,
        model=args.model,
        n_layers=num_layers,
    )

    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    if args.cuda:
        decoder.cuda()

    start = time.time()
    all_losses = []

    try:
        print("Training for %d epochs..." % args.n_epochs)
        train_loss = []
        test_loss = []
        for epoch in tqdm(range(1, args.n_epochs + 1)):

            # train and report training errror
            train_loss.append(train(*random_batch(train_file, args.chunk_len, args.batch_size)))

            # report test error
            test_loss.append(test(*random_batch(test_file, args.chunk_len, args.batch_size)))

            if epoch % args.print_every == 0:
                #print('[%s (%d %d%%)]' % (time_since(start), epoch, epoch / args.n_epochs * 100))
                print('Test loss:', test_loss[-1])
                #print(generate(decoder, 'Wh', 100, cuda=args.cuda), '\n')

        print("Saving...")
        save()

        # save all relevant info to a dict and pickle
        result_dict = {}
        result_dict['model'] = args.model
        result_dict['num_layers'] = num_layers
        result_dict['hidden_size'] = hidden_size
        result_dict['learning_rate'] = learning_rate
        result_dict['chunk_len'] = args.chunk_len
        result_dict['n_epochs'] = args.n_epochs
        result_dict['batch_size'] = args.batch_size
        result_dict['train_loss'] = train_loss
        result_dict['test_loss'] = test_loss
        result_dict['final_test_loss'] = test_loss[-1]
        result_dict['elapsed_time'] = time_since(start)
        pickle.dump(result_dict, open(str(time.time()) + '.pk', 'wb'))

    except KeyboardInterrupt:
        print('Moving on to next configuration...')

