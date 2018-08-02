from modules.TemporalJitter import TemporalJitter
import math
import random
from PIL import Image
from torchvision import transforms
import torch


def round(x):
    return math.floor(x + 0.5)

def read_data(d, opt, vocab_mapping):
    test_mode = d['test'] or False
    mode = d['mode'] or random.randint(1, 6)
    flip = d['flip'] or False
    if mode < 7:
        w_start = d['w_start'] or random.randint(1, len(d['words']) - mode + 1)
        w_end = w_start + mode -1
    
    min_frame_v, max_frame_v = 1, 75
    sub = ''
    frame_v_start, frame_v_end = -1, -1

    # test mode
    if test_mode:
        frame_v_start, frame_v_end = min_frame_v, max_frame_v
        sub = ' '.join(d['words'])
    # train mode
    else:
        # check number of words to train on
        if mode == 7:
            frame_v_start, frame_v_end = min_frame_v, max_frame_v
            sub = ' '.join(d['words'])
        else:
            # generate target
            words = []
            for w_i in range(w_start, w_end + 1):
                words.append(d['words'][w_i - 1])
            sub = ' '.join(d['words'])

            frame_v_start = max(round(75 / 3000 * d['t_start'][w_start - 1]), 1)
            frame_v_end = min(round(75 / 3000 * d['t_end'][w_end - 1]), 75)

            # if too short, back off to whole sequence
            if frame_v_end - frame_v_start + 1 <= 2:
                frame_v_start, frame_v_end = min_frame_v, max_frame_v
                sub = ' '.join(d['words'])
    
    # construct output tensor
    y = []
    # allow whitespaces to be predicted
    if opt['mode_sub']:
        for char in sub:
            y.append(vocab_mapping[char])
    else:
        for char in sub:
            if char != ' ':
                y.append(vocab_mapping[char])
    
    # data path: BASE/s1/....../mouth_xxx.png
    cur_path = '%s/%s/%s/%s' % (opt['datapath'], d['s'], d['v'], opt['mode_img'])
    # load images
    # randomly flip video
    if test_mode: flip = False
    else: flip = flip or random.random() > 0.5

    # NOTE: since LipNet uses a fixed-size crop, we divert from the
    # original implementation, which attempts to determine H and W
    # from the first frame in the sequence.
    x = torch.Tensor(3, frame_v_end - frame_v_start + 1, 50, 100)
    transform_lst = []
    if flip:
        transform_lst.append(transforms.functional.hflip)
    transform_lst += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7136, 0.4906, 0.3283],
                std=[0.113855171, 0.107828568, 0.0917060521])
    ]
    data_transform = transforms.Compose(transform_lst)
    frame_count = 0
    for f_frame in range(frame_v_start, frame_v_end + 1):
        img = Image.open('%s_%03d.png' % (cur_path, f_frame - 1)).convert('RGB')
        img = data_transform(img)
        x[:, frame_count, :, :] = img
        frame_count += 1

    # temporal jitter
    if opt['temporal_jitter'] > 0 and not test_mode:
        temporal_jitter = TemporalJitter(opt['temporal_jitter'])
        x = temporal_jitter(x)
    
    return (x, y)
