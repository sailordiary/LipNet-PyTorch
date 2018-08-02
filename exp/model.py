import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from modules.TimeBatchWrapper import TimeBatchWrapper
import math
from dataloader import GRIDDataset, warpctc_collate
from torch.utils.data import DataLoader


# NOTE: the following are legacy torch grammar, but we
# implement them anyway, in order to wrap everything
# into a single LipNet().pred module for simplicity.
class Transpose(nn.Module):

    def __init__(self, *args):
        super(Transpose, self).__init__()
        self.permutations = args

    def forward(self, input):
        for perm in self.permutations:
            input = input.transpose(*perm)
        return input


class View(nn.Module):

    def resetSize(self, *args):
        if len(args) == 1 and isinstance(args[0], torch.Size):
            self.size = args[0]
        else:
            self.size = torch.Size(args)

        self.numElements = 1
        inferdim = False
        for i in range(len(self.size)):
            szi = self.size[i]
            if szi >= 0:
                self.numElements = self.numElements * self.size[i]
            else:
                assert szi == -1
                assert not inferdim
                inferdim = True

        return self

    def __init__(self, *args):
        super(View, self).__init__()
        self.resetSize(*args)

    def forward(self, input):
        return input.view(self.size)

    def __repr__(self):
        return super(View, self).__repr__() + '({})'.format(', '.join(map(str, self.size)))


class LipNet(nn.Module):

    def __init__(self, opt, vocab_size):
        super(LipNet, self).__init__()
        self.opt = opt
        self.conv = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            # nn.BatchNorm3d(32, momentum=0.9),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(opt['dropout']),
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            # nn.BatchNorm3d(64, momentum=0.9),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(opt['dropout']),
            nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            # nn.BatchNorm3d(96, momentum=0.9),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(opt['dropout']),
            # B C T H W
            Transpose({0, 1}, {0, 2}),
            # T B C H W
            TimeBatchWrapper(mod=View(-1, 96 * 3 * 6))
        )
        # T B C*H*W
        self.gru1 = nn.GRU(96 * 3 * 6, opt['rnn_size'], 1, bidirectional=True)
        self.drp1 = nn.Dropout(opt['dropout'])
        # T B F
        self.gru2 = nn.GRU(opt['rnn_size'] * 2, opt['rnn_size'], 1, bidirectional=True)
        self.pred = nn.Sequential(
            # T B F
            nn.Dropout(opt['dropout']),
            TimeBatchWrapper(mod=nn.Linear(opt['rnn_size'] * 2, vocab_size + 1))
            # T B V'
        )
        # initializations
        for m in self.conv.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                init.constant_(m.bias, 0)
        for m in self.pred.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity='sigmoid')
                init.constant_(m.bias, 0)
        for m in (self.gru1, self.gru2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + opt['rnn_size']))
            for i in range(0, opt['rnn_size'] * 3, opt['rnn_size']):
                init.uniform_(m.weight_ih_l0[i: i + opt['rnn_size']],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0[i: i + opt['rnn_size']])
                init.constant_(m.bias_ih_l0[i: i + opt['rnn_size']], 0)
    
    def forward(self, x):
        x = self.conv(x)
        x, _ = self.gru1(x)
        x = self.drp1(x)
        x, _ = self.gru2(x)
        x = self.pred(x)
        
        return x


class Exp:
    def __init__(self, opt):
        # load data
        self.trainset = GRIDDataset(opt, dset='train')
        self.trainset.load_data()
        self.testset = GRIDDataset(opt, dset='test')
        self.testset.load_data()
        self.trainloader = DataLoader(self.trainset, batch_size=opt['dataset']['bs'],
            shuffle=True, num_workers=opt['train']['num_workers'], collate_fn=warpctc_collate)
        self.testloader = DataLoader(self.testset, batch_size=opt['dataset']['bs'],
            shuffle=True, num_workers=opt['train']['num_workers'], collate_fn=warpctc_collate)

        # define network
        self.input_img_size = self.trainset.opt['size']
        self.vocab_size = len(self.trainset.vocab)
        assert self.testset.vocab <= self.trainset.vocab, 'possible OOV characters in test set'
        self.maxT = self.trainset.opt['max_timesteps']

        self.chan, self.height, self.width = self.input_img_size
        assert self.chan == 1 or self.chan == 3, '1 or 3 channels only, in input_img_size'
        assert self.width > 0 and self.height > 0, 'width and height must be specified in input_img_size'

        self.model = LipNet(opt['network'], self.vocab_size)
        self.opt = opt

    def optim(self, epoch):
        optimfunc = torch.optim.Adam(self.model.parameters(), lr = self.opt['train']['lr'])
        
        return optimfunc
