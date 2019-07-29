import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from dataloader import GRIDDataset, ctc_collate
from torch.utils.data import DataLoader


class LipNet(nn.Module):
    def __init__(self, opt, vocab_size):
        super(LipNet, self).__init__()
        self.opt = opt
        self.conv = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(opt.dropout),
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(opt.dropout),
            nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(opt.dropout)
        )
        # T B C*H*W
        self.gru1 = nn.GRU(96 * 3 * 6, opt.rnn_size, 1, bidirectional=True)
        self.drp1 = nn.Dropout(opt.dropout)
        # T B F
        self.gru2 = nn.GRU(opt.rnn_size * 2, opt.rnn_size, 1, bidirectional=True)
        self.drp2 = nn.Dropout(opt.dropout)
        # T B V
        self.pred = nn.Linear(opt.rnn_size * 2, vocab_size + 1)
        
        # initialisations
        for m in self.conv.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                init.constant_(m.bias, 0)

        init.kaiming_normal_(self.pred.weight, nonlinearity='sigmoid')
        init.constant_(self.pred.bias, 0)

        for m in (self.gru1, self.gru2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + opt.rnn_size))
            for i in range(0, opt.rnn_size * 3, opt.rnn_size):
                init.uniform_(m.weight_ih_l0[i: i + opt.rnn_size],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0[i: i + opt.rnn_size])
                init.constant_(m.bias_ih_l0[i: i + opt.rnn_size], 0)
                init.uniform_(m.weight_ih_l0_reverse[i: i + opt.rnn_size],
                            -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                init.orthogonal_(m.weight_hh_l0_reverse[i: i + opt.rnn_size])
                init.constant_(m.bias_ih_l0_reverse[i: i + opt.rnn_size], 0)
    
    def forward(self, x):
        x = self.conv(x) # B C T H W
        x = x.permute(2, 0, 1, 3, 4).contiguous() # T B C H W
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.gru1(x)
        x = self.drp1(x)
        x, _ = self.gru2(x)
        x = self.drp2(x)
        x = self.pred(x)
        
        return x


class Exp:
    def __init__(self, opt):
        self.trainset = GRIDDataset(opt, dset='train')
        self.trainset.load_data()
        self.testset = GRIDDataset(opt, dset='test')
        self.testset.load_data()
        self.trainloader = DataLoader(self.trainset, batch_size=opt.batch_size,
            shuffle=True, num_workers=opt.num_workers, collate_fn=ctc_collate, pin_memory=True)
        self.testloader = DataLoader(self.testset, batch_size=opt.batch_size,
            shuffle=False, num_workers=opt.num_workers, collate_fn=ctc_collate, pin_memory=True)

        # define network
        self.input_img_size = [3, 50, 100]
        self.chan, self.height, self.width = self.input_img_size
        self.vocab_size = len(self.trainset.vocab)
        assert self.testset.vocab <= self.trainset.vocab, 'possible OOV characters in test set'
        self.maxT = self.trainset.opt.max_timesteps

        self.model = LipNet(opt, self.vocab_size)
        self.opt = opt

    # learning rate scheduler
    def optim(self, epoch):
        optimfunc = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr)
        
        return optimfunc

