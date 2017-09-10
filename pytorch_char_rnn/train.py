#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

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
argparser.add_argument('--model', type=str, default="gru")
argparser.add_argument('--n_epochs', type=int, default=2000)
argparser.add_argument('--print_every', type=int, default=1)
argparser.add_argument('--hidden_size', type=int, default=100)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--chunk_len', type=int, default=200)
argparser.add_argument('--batch_size', type=int, default=50)
argparser.add_argument('--cuda', action='store_true')
args = argparser.parse_args()

if args.cuda:
    print("Using CUDA")

file, file_len = read_file(args.filename)

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
    print('Training set size (chunks):', len(train_file)/ chunk_len)
    print('Test set size (chunks):', len(test_file)/ chunk_len)
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
        loss += test_criterion(output.view(args.batch_size, -1), target[:,c])
    return loss.data[0] / args.chunk_len

def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

# Initialize models and start training

decoder = CharRNN(
    n_characters,
    args.hidden_size,
    n_characters,
    model=args.model,
    n_layers=args.n_layers,
)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()
test_criterion = nn.NLLLoss()

if args.cuda:
    decoder.cuda()

start = time.time()
all_losses = []

try:
    print("Training for %d epochs..." % args.n_epochs)
    train_file, test_file = partition_text_file(file, args.chunk_len, 0.8)
    train_loss = []
    test_loss = []
    for epoch in tqdm(range(1, args.n_epochs + 1)):

        # train and report training errror
        train_loss.append(train(*random_batch(train_file, args.chunk_len, args.batch_size)))

        # report test error
        test_loss.append(test(*random_batch(test_file, args.chunk_len, args.batch_size)))

        if epoch % args.print_every == 0:
            print('[%s (%d %d%%)]' % (time_since(start), epoch, epoch / args.n_epochs * 100))
            print('Test NLL loss:', test_loss[-1])
            print(generate(decoder, 'Wh', 100, cuda=args.cuda), '\n')

    print("Saving...")
    save()
    np.save('result', np.array(test_loss))

except KeyboardInterrupt:
    print("Saving before quit...")
    save()

