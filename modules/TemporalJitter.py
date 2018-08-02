import torch
import torch.nn as nn


class TemporalJitter(nn.Module):

    def __init__(self, p=None, l=None):
        super(TemporalJitter, self).__init__()
        self.p = p or 0.05
        self.compute_len = l or False

        self.train = True
    
    def train(self):
        self.train = True
    
    def eval(self):
        self.train = False
    
    def forward(self, input):
        assert input.dim() == 5 or input.dim() == 4, 'boom'
        if self.train:
            input_dim4 = input.dim() == 4
            if input_dim4:
                # expand dimension: B * C * T * H * W
                input = input.view(1, input.size(0), input.size(1), input.size(2), input.size(3))
            length = input.size(2)
            output = torch.tensor(input)
            for b in range(0, input.size(0)):
                prob_del = torch.Tensor(length).bernoulli_(self.p)
                prob_dup = prob_del.index_select(0, torch.linspace(length - 1, 0, length).long())
                output_count = 0
                for t in range(0, length):
                    if prob_del[t] == 0:
                        output[b, :, output_count, :] = input[b, :, t, :]
                        output_count += 1
                    if prob_dup[t] == 1 and output_count > 0:
                        output[b, :, output_count, :] = input[b, :, output_count - 1, :]
                        output_count += 1
            if input_dim4: return output[0]
            else: return output
        else:
            return input
