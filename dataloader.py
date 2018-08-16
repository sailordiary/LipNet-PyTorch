from augmentation import read_data
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import os
import json
from progressbar import *
from skimage import io
import random


def round(x):
    return math.floor(x + 0.5)

def warpctc_collate(batch):
    '''
    Stack samples into warp_ctc style inputs.
    Modified based on default_collate() in PyTorch.
    By Yuan-Hang Zhang.
    '''
    # TODO: implement type assertion
    # error_msg = "batch must be list of (x, y, len); found {}"
    xs, ys, lens, indices, subtitles = zip(*batch)
    max_len = max(lens)
    x = default_collate(xs)
    x.narrow(2, 0, max_len)
    y = []
    for sub in ys: y += sub
    labels = torch.IntTensor(y)
    label_lens = torch.IntTensor([len(label) for label in ys])
    act_lens = torch.IntTensor(lens)
    ids = default_collate(indices)
    subs = default_collate(subtitles)

    return [x, labels, act_lens, label_lens, ids, subs]

class GRIDDataset(Dataset):
    def __init__(self, opt, dset='train'):
        self.opt = opt['dataset']
        self.dset = dset
        # TODO: implement overlapped speakers
        #self.overlapped_list = json.loads(open(self.opt['list_overlapped'], 'r').read())
        self.dataset = []
    
    def load_data(self):
        opt = self.opt
        vocab_unordered = {}
        vocab_unordered[' '] = True

        # iterate speakers
        print ('Loading videos for %s set...' % self.dset)
        count_s = 0
        pbar = ProgressBar().start()
        for dir_s in os.listdir(opt['datapath']):
            count_s += 1
            # get speaker videos
            for dir_v in os.listdir('%s/%s' % (opt['datapath'], dir_s)):
                if not opt['debug'] or len(self.dataset) <= opt['bs'] * 10:
                    cur_path = '%s/%s/%s' % (opt['datapath'], dir_s, dir_v)
                    # load filter
                    # check if sub was transcribed
                    sub_file = '%s/%s/align/%s.align' % (opt['alignpath'], dir_s, dir_v)
                    flag_add = os.path.exists(sub_file)
                    # check if frames exist and are of correct shape
                    if not os.path.exists(cur_path) or len(os.listdir(cur_path)) != 75:
                        flag_add = False
                    else:
                        # read image size
                        size = io.imread(cur_path + '/mouth_000.png').shape
                        if size[0] != 50 or size[1] != 100:
                            flag_add = False
                    if flag_add:
                        # load subs
                        d = {'s': dir_s, 'v': dir_v, 'words': [], 't_start': [], 't_end': []}
                        for line in open(sub_file, 'r').readlines():
                            tok = line.rstrip().split(' ')
                            if tok[2] != 'sil' and tok[2] != 'sp':
                                # store sub and append
                                d['words'].append(tok[2])
                                d['t_start'].append(int(tok[0]))
                                d['t_end'].append(int(tok[1]))
                                # build vocabulary
                                for char in tok[2]:
                                    vocab_unordered[char] = True
                        # append to subs data
                        if (not opt['test_overlapped'] and (dir_s in ['s1', 's2', 's20', 's22'])) or (opt['test_overlapped']): #and self.overlapped_list[dir_s][dir_v]):
                            if self.dset == 'test':
                                d['mode'] = 7
                                d['flip'] = False
                                d['test'] = True
                                self.dataset.append(d)
                        else:
                            if self.dset == 'train':
                                d['test'] = False
                                for flip in (False, True):
                                    # add word instances
                                    if opt['use_words']:
                                        for w_start in range(1, 7):
                                            d_i = d.copy()
                                            d_i['flip'] = flip
                                            d_i['mode'] = 1
                                            d_i['w_start'] = w_start
                                            # NOTE: it appears the authors never made use of the mode option.
                                            # All instances used were either whole sentences or individual words.
                                            d_i['w_end'] = w_start + d_i['mode'] - 1
                                            frame_v_start = max(round(1 / 1000 * d['t_start'][d_i['w_start'] - 1]), 1)
                                            frame_v_end = min(round(1 / 1000 * d['t_end'][d_i['w_end'] - 1]), 75)
                                            if frame_v_end - frame_v_start + 1 >= 3:
                                                self.dataset.append(d_i)
                                    # add whole sentences
                                    d_i = d.copy()
                                    d_i['mode'] = 7
                                    d_i['flip'] = flip
                                    self.dataset.append(d_i)
            pbar.update(int(count_s / 33 * 100))
        pbar.finish()
        # generate vocabulary
        self.vocab = []
        for char in vocab_unordered:
            self.vocab.append(char)
        self.vocab.sort()
        # invert ordered to create the char->int mapping
        # key: 1..N (reserve 0 for blank symbol and padding)
        self.vocab_mapping = {}
        for i, char in enumerate(self.vocab):
            self.vocab_mapping[char] = i + 1
        print ('videos = %d, set = %s, vocab = %d' % (len(self.dataset), self.dset, len(self.vocab)))
        print ('frame size:', tuple(opt['size']))
        print ('vocab = %s' % '|'.join(self.vocab))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        chan, H, W = self.opt['size']
        max_time = self.opt['max_timesteps']
        # images: bs x chan x T x H x W
        x = torch.zeros(chan, max_time, H, W)
        # load video using read_data() and shove into x
        d = self.dataset[index]
        # targets: bs-length tensor of targets (each one is the length of the target seq)
        frames, y, sub = read_data(d, self.opt, self.vocab_mapping)
        x[:, : frames.size(1), :, :] = frames
        # input lengths: bs-length tensor of integers, representing
        # the number of input timesteps/frames for the given batch element
        length = frames.size(1)

        return (x, y, length, index, sub)
