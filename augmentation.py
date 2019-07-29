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
            sub = ' '.join(words)

            frame_v_start = max(round(1 / 1000 * d['t_start'][w_start - 1]), 1)
            frame_v_end = min(round(1 / 1000 * d['t_end'][w_end - 1]), 75)

            # if too short, back off to whole sequence
            if frame_v_end - frame_v_start + 1 <= 2:
                frame_v_start, frame_v_end = min_frame_v, max_frame_v
                sub = ' '.join(d['words'])
    
    # construct output tensor
    y = []
    # allow whitespaces to be predicted
    for char in sub: y.append(vocab_mapping[char])
    
    # load images
    # data path: $BASE/sX/.../mouth_xxx.png
    cur_path = '{}/{}/{}/{}'.format(opt.datapath, d['s'], d['v'], opt.mode_img)
    # randomly flip video
    if test_mode: flip = False
    else: flip = flip or random.random() > 0.5

    x = torch.FloatTensor(3, frame_v_end - frame_v_start + 1, 50, 100)
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
        img = Image.open('{}_{:03d}.png'.format(cur_path, f_frame - 1)).convert('RGB')
        img = data_transform(img)
        x[:, frame_count, :, :] = img
        frame_count += 1

    # temporal jitter
    if opt.temporal_aug > 0 and not test_mode:
        length = x.size(1)
        output = x.clone()
        prob_del = torch.Tensor(length).bernoulli_(opt.temporal_aug)
        prob_dup = prob_del.index_select(0, torch.linspace(length - 1, 0, length).long())
        output_count = 0
        for t in range(0, length):
            if prob_del[t] == 0:
                output[:, output_count, :] = x[:, t, :]
                output_count += 1
            if prob_dup[t] == 1 and output_count > 0:
                output[:, output_count, :] = x[:, output_count - 1, :]
                output_count += 1
        x = output
    
    return x, y, sub
