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
import torch.utils.data as data_utils

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
argparser.add_argument('--cuda', type=int, default=1)
arguments = argparser.parse_args()

arguments.cuda = arguments.cuda and torch.cuda.is_available()

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
    if arguments.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def get_loader(fl, chunk_len, batch_size):
    
    inputs = fl[:-1]
    targets = fl[1:]

    N = len(inputs)

    N = N - N % chunk_len
    inputs, targets = inputs[:N], targets[:N]
   
    all_inputs = [char_tensor(inputs[i:i+chunk_len]).view(1,-1) for i in range(0, N, chunk_len)]
    all_targets = [char_tensor(targets[i:i+chunk_len]).view(1,-1) for i in range(0, N, chunk_len)]

    all_inputs = torch.cat(all_inputs, 0)
    all_targets = torch.cat(all_targets, 0)

    dataset = data_utils.TensorDataset(data_tensor=all_inputs,
                                       target_tensor=all_targets)

    kwargs = {'num_workers': 1, 'pin_memory': True} if arguments.cuda else {}
    loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)
    return loader


def train(inp, target):
    batch_size = inp.size(0)

    hidden = decoder.init_hidden(batch_size)
    if arguments.cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0
    for c in range(arguments.chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(batch_size, -1), target[:,c])
    loss.backward()
    decoder_optimizer.step()
    return loss.data[0] / arguments.chunk_len

def test(inp, target):
    batch_size = inp.size(0)

    hidden = decoder.init_hidden(batch_size)
    if arguments.cuda:
        hidden = hidden.cuda()
    loss = 0
    for c in range(arguments.chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(batch_size, -1), target[:,c])
    return loss.data[0] / arguments.chunk_len

def save():
    save_filename = os.path.splitext(os.path.basename(arguments.filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

"""
file, file_len = read_file(arguments.filename)
train_file, test_file = partition_text_file(file, arguments.chunk_len, 0.75)
train_file, vld_file = partition_text_file(train_file, arguments.chunk_len, 0.75)
pickle.dump(train_file, open('train_set.pk', 'wb'))
pickle.dump(vld_file, open('vld_set.pk', 'wb'))
pickle.dump(test_file, open('test_set.pk', 'wb'))
"""

train_file = pickle.load(open('train_set.pk','rb'))
vld_file = pickle.load(open('vld_set.pk','rb'))
test_file = pickle.load(open('test_set.pk','rb'))

# get the loaders 

train_loader = get_loader(train_file, arguments.chunk_len, arguments.batch_size)
vld_loader = get_loader(vld_file, arguments.chunk_len, arguments.batch_size)
test_loader = get_loader(test_file, arguments.chunk_len, arguments.batch_size)


if arguments.cuda:
    print("Using CUDA")

for configuration in range(arguments.num_configs):
    
    # we should take this outside of the loop 
    learning_rate, hidden_size, num_layers = generate_random_hyperparams(arguments.lr_min, 
                                                                         arguments.lr_max, 
                                                                         arguments.K_min, 
                                                                         arguments.K_max, 
                                                                         arguments.num_layers_min, 
                                                                         arguments.num_layers_max)
    print(learning_rate, hidden_size, num_layers)

    decoder = CharRNN(
        n_characters,
        hidden_size,
        n_characters,
        model=arguments.model,
        n_layers=num_layers)

    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    if arguments.cuda:
        decoder.cuda()

    start = time.time()
    all_losses = []

    try:
        print("Training for %d epochs..." % arguments.n_epochs)
        train_loss = []
        test_loss = []
        for epoch in tqdm(range(1, arguments.n_epochs + 1)):

            # train and report training errror
            train_loss_batches = [] 
            for n, (inp, tar) in enumerate(train_loader):
                if arguments.cuda:
                    inp = inp.cuda()
                    tar = tar.cuda()
                train_loss_batch = train(Variable(inp), 
                                         Variable(tar))
                train_loss_batches.append(train_loss_batch)
                print('Processing batch [{}]'.format(n))
            train_loss.append(np.mean(train_loss_batches))

            # report test error
            test_loss_batches = []
            for n, (inp, tar) in enumerate(test_loader):
                if arguments.cuda:
                    inp = inp.cuda()
                    tar = tar.cuda()
                test_loss_batch = test(Variable(inp),
                                       Variable(tar))
                test_loss_batches.append(test_loss_batch)
            test_loss.append(np.mean(test_loss_batches))

            if epoch % arguments.print_every == 0:
                #print('[%s (%d %d%%)]' % (time_since(start), epoch, epoch / arguments.n_epochs * 100))
                print('Test loss:', test_loss[-1])

        print("Saving...")
        save()

        # save all relevant info to a dict and pickle
        result_dict = {}
        result_dict['model'] = arguments.model
        result_dict['num_layers'] = num_layers
        result_dict['hidden_size'] = hidden_size
        result_dict['learning_rate'] = learning_rate
        result_dict['chunk_len'] = arguments.chunk_len
        result_dict['n_epochs'] = arguments.n_epochs
        result_dict['batch_size'] = arguments.batch_size
        result_dict['train_loss'] = train_loss
        result_dict['test_loss'] = test_loss
        result_dict['final_test_loss'] = test_loss[-1]
        result_dict['example_gen_text'] = generate(decoder, 'Wh', 200, cuda=arguments.cuda)
        result_dict['elapsed_time'] = time_since(start)
        print(result_dict['example_gen_text'])
        pickle.dump(result_dict, open(str(time.time()) + '.pk', 'wb'))

    except KeyboardInterrupt:
        print('Moving on to next configuration...')

