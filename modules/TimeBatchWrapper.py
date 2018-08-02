import torch
import torch.nn as nn

'''
Merges timesteps and batchsize together so the given module 
can process all timesteps in parallel, time/batch dims are given
into the ctor, and default to 0 and 1, respectively. The same
dimension indices are used on the input and output, although other
dimensions can change.

Dimensions can be negative: -1 is last, -2 second-last, etc.

Module can contain parameters, but input/output must be plain tensors, 
no tables.

NOTE: obviously for some modules, this has potential effects on their correctness,
e.g. for BatchNorm. But for almost all other layers this is fine.
'''
class TimeBatchWrapper(nn.Module):

    def __init__(self, mod=None, time_dim=0, batch_dim=1):
        super(TimeBatchWrapper, self).__init__()
        assert abs(time_dim - batch_dim) == 1, 'time and batch dims must be adjacent'
        self.time_dim = time_dim
        self.batch_dim = batch_dim
        self.mod = mod
        self._dim1 = min(time_dim, batch_dim)
        self._dim2 = max(time_dim, batch_dim)
    
    def parameters(self):
        return self.mod.parameters
    
    def type(self, dst_type):
        self.mod.type(dst_type)
        self.output = self.mod.output
        self.gradInput = self.mod.gradInput
        return self
    
    def forward(self, input):
        # reshape to giant time batch
        dim1, dim2 = self._dim1, self._dim2
        if dim1 < 0: dim1 += input.dim() + 1
        if dim2 < 0: dim2 += input.dim() + 1
        assert dim1 + 1 == dim2, 'internal error: batch/time dims arent adjacent'

        # merge: multiply 1st dim size by 2nd dim size, delete 2nd dim
        newsize = list(input.size())
        newsize[dim1] = newsize[dim1] * newsize[dim2]
        del(newsize[dim2])
        newsize = tuple(newsize)
        newinput = input.contiguous().view(newsize)
        # forward pass on shaped
        out = self.mod(newinput)

        # split the dim again
        # NOTE: fails loudly if time/batch are different on output than input
        outsize = list(out.size())
        outsize[dim1] = input.size(dim2)        # dim2, since it'll get shifted
        outsize.insert(dim1, input.size(dim1))  # dim1
        outsize = tuple(outsize)

        return out.view(outsize)
