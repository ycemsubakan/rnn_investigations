import math
import torch
from torch.nn.modules.rnn import *
from torch.nn._functions.rnn import *
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend
import pdb

class ModRNNTanhCell():
    def __init__(self, hidden_size):
        self.K = hidden_size

    def allocate_parameters(self, layer_inpt_size):
        w_ih = Parameter(torch.Tensor(self.K, layer_inpt_size))
        w_hh = Parameter(torch.Tensor(self.K, self.K))
        b_ih = Parameter(torch.Tensor(self.K))
        b_hh = Parameter(torch.Tensor(self.K))
        return w_ih, w_hh, b_ih, b_hh

    def forward(self, inpt, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        hy = F.tanh(F.linear(inpt, w_ih) + F.linear(hidden, w_hh) + b_hh) 
        return hy


class ModRNNTanhCellDiag():
    def __init__(self, hidden_size):
        self.K = hidden_size

    def allocate_parameters(self, layer_inpt_size):
        w_ih = Parameter(torch.Tensor(self.K, layer_inpt_size))
        w_hh = Parameter(torch.Tensor(self.K))
        b_ih = Parameter(torch.Tensor(self.K))
        b_hh = Parameter(torch.Tensor(self.K))
        return w_ih, w_hh, b_ih, b_hh

    def forward(self, inpt, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        hy = F.tanh(F.linear(inpt, w_ih) + hidden * w_hh + b_hh)
        return hy

class ModLSTMCell():
    def __init__(self, hidden_size):
        self.K = hidden_size

    def allocate_parameters(self, layer_inpt_size):
        w_ih = Parameter(torch.Tensor(4 * self.K, layer_inpt_size))
        w_hh = Parameter(torch.Tensor(4 * self.K, self.K))
        b_ih = Parameter(torch.Tensor(4 * self.K))
        b_hh = Parameter(torch.Tensor(4 * self.K))
        return w_ih, w_hh, b_ih, b_hh

    def forward(self, inpt, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        if inpt.is_cuda:
            igates = F.linear(inpt, w_ih)
            hgates = F.linear(hidden[0], w_hh)
            state = fusedBackend.LSTMFused()
            return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)

        hx, cx = hidden
        gates = F.linear(inpt, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)

        return hy, cy

class ModLSTMCellDiag():
    def __init__(self, hidden_size):
        self.K = hidden_size

    def allocate_parameters(self, layer_inpt_size):
        w_ih = Parameter(torch.Tensor(4 * self.K, layer_inpt_size))
        w_hh = Parameter(torch.Tensor(4 * self.K))
        b_ih = Parameter(torch.Tensor(4 * self.K))
        b_hh = Parameter(torch.Tensor(4 * self.K))
        return w_ih, w_hh, b_ih, b_hh

    def forward(self, inpt, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
        if inpt.is_cuda:
            igates = F.linear(inpt, w_ih)
            hgates = hidden[0] * w_hh
            state = fusedBackend.LSTMFused()
            return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)

        hx, cx = hidden
        gates = F.linear(inpt, w_ih) + hx * w_hh + b_hh

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)

        return hy, cy

class ModRNNBase(torch.nn.Module):
    """
    base RNN class
    """
    def __init__(self, mode, inpt_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=.1, bidirectional=False):
        super(ModRNNBase, self).__init__()
        
        # pick the rnn cell to be used
        if mode == 'vanilla_tanh':
            self.rnncell = ModRNNTanhCell(hidden_size) 
        elif mode == 'vanilla_tanh_diag':
            self.rnncell = ModRNNTanhCellDiag(hidden_size)
        elif mode == 'lstm':
            self.rnncell = ModLSTMCell(hidden_size)
        elif mode == 'lstm_diag':
            self.rnncell = ModLSTMCellDiag(hidden_size)
        else:
            raise ValueError('Unknown cell type: {}'.format(mode))

        self.mode = mode
        self.inpt_size = inpt_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_inpt_size = inpt_size if layer == 0 else hidden_size * num_directions
                w_ih, w_hh, b_ih, b_hh = self.rnncell.allocate_parameters(layer_inpt_size)
                layer_params = (w_ih, w_hh, b_ih, b_hh)
                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]
                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)
        self._data_ptrs = []
        self.reset_parameters()


    def _apply(self, fn):
        ret = super(ModRNNBase, self)._apply(fn)
        #self.flatten_parameters()
        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, inpt, hx=None):
        is_packed = isinstance(inpt, PackedSequence)
        if is_packed:
            inpt, batch_sizes = inpt
            max_batch_size = batch_sizes[0]
        else:
            batch_sizes = None
            max_batch_size = inpt.size(0) if self.batch_first else inpt.size(1)
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.autograd.Variable(inpt.data.new(self.num_layers *
                                                        num_directions,
                                                        max_batch_size,
                                                        self.hidden_size).zero_(), requires_grad=False)
        has_flat_weights = list(p.data.data_ptr() for p in self.parameters()) == self._data_ptrs
        if has_flat_weights:
            first_data = next(self.parameters()).data
            assert first_data.storage().size() == self._param_buf_size
            flat_weight = first_data.new().set_(first_data.storage(), 0, torch.Size([self._param_buf_size]))
        else:
            flat_weight = None
        func = self.fw(
            self.mode,
            self.inpt_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            train=self.training,
            bidirectional=self.bidirectional,
            batch_sizes=batch_sizes,
            dropout_state=self.dropout_state,
            flat_weight=flat_weight
        )
        output, hidden = func(inpt, self.all_weights, hx)
        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, hidden

    def __repr__(self):
        s = '{name}({inpt_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def __setstate__(self, d):
        super(ModRNNBase, self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                if self.bias:
                    self._all_weights += [weights]
                else:
                    self._all_weights += [weights[:2]]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]

    def fw(self, mode, inpt_size, hidden_size, num_layers=1, batch_first=False,
                    dropout=0, train=True, bidirectional=False, batch_sizes=None,
                    dropout_state=None, flat_weight=None):
        """
        define forward computation over cells
        """
        cell = self.rnncell.forward
        if batch_sizes is None:
            rec_factory = Recurrent
        else:
            rec_factory = variable_recurrent_factory(batch_sizes)
        if bidirectional:
            layer = (rec_factory(cell), rec_factory(cell, reverse=True))
        else:
            layer = (rec_factory(cell),)
        func = StackedRNN(layer,
                          num_layers,
                          ('lstm' in mode),
                          dropout=dropout,
                          train=train)
        def forward(inpt, weight, hidden):
            if batch_first and batch_sizes is None:
                inpt = inpt.transpose(0, 1)
            nexth, output = func(inpt, hidden, weight)
            if batch_first and batch_sizes is None:
                output = output.transpose(0, 1)
            return output, nexth
        return forward