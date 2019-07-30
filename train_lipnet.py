import time
from progressbar import *
import os, sys

import random
import argparse
from collections import OrderedDict

from torchsummary import summary

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from model import Exp
from ctc_decoder import Decoder


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    parser = argparse.ArgumentParser(description='LipNet in PyTorch')
    
    parser.add_argument('--seed', default=123, type=int, help='random seed')
    parser.add_argument('--datapath', default='data/images', help='directory containing numeric video ID subdirectories')
    parser.add_argument('--alignpath', default='data/align', help='directory containing audio alignments')
    parser.add_argument('--list_overlapped', default='list_overlapped.json', help='list containing selected overlapped test samples')
    parser.add_argument('--test_overlapped', default=False, action='store_true', help='test overlapped speakers')
    
    parser.add_argument('--checkpoint', default='', help='checkpoint to be loaded')
    parser.add_argument('--lmpath', default='', help='path to KenLM language model')
                        
    parser.add_argument('--min_timesteps', default=2, type=int, help='min frames, for filtering bad data')
    parser.add_argument('--max_timesteps', default=75, type=int, help='maximum number of frames per sub, for preallocation')
    parser.add_argument('--temporal_aug', default=0.05, type=float, help='temporal jittering probability')
    parser.add_argument('--use_words', default=True, type=bool, help='whether to use word training samples')

    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=50, type=int, help='mini-batch size (default: 50)')
    parser.add_argument('--curriculum', default=0.925, type=float, help='curriculum learning weight for CTC gradients')
    parser.add_argument('--print_every', default=1, type=int, help='epochs between printing')
    parser.add_argument('--test_every', default=1, type=int, help='epochs between testing')
    parser.add_argument('--checkpoint_every', default=1, type=int, help='epochs between saving checkpoints')
    parser.add_argument('--epochs', default=10000, type=int, help='number of epochs to train')
    
    parser.add_argument('--test', default=False, action='store_true', help='only run test phase')
    
    parser.add_argument('--rnn_size', default=256, type=int, help='RNN size (default: 256)')
    parser.add_argument('--dropout', default=0.5, type=float, help='Dropout rate (default: 0.5)')

    parser.add_argument('--num_workers', default=10, type=int, help='number of data loader workers (default: 10)')
    parser.add_argument('--mode_img', default='mouth', help='image name prefix')

    opt = parser.parse_args()
    for arg in vars(opt):
        print ('opt: {}={}'.format(arg, getattr(opt, arg)))

    # deterministic training
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True

    # construct model
    exp = Exp(opt)
    model = exp.model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    summary(model, input_size=(3, opt.max_timesteps, 50, 100), device='cpu')
    print ('Type \'q\' to exit or any other key to continue: ', end='')
    if input() == 'q': sys.exit(0)

    model = model.to(device)
    decoder = Decoder(exp.trainset.vocab, lm_path=opt.lmpath)
    crit = nn.CTCLoss(reduction='none', zero_infinity=True).to(device)

    # load model
    niters, start_epoch = 0, 0
    if opt.checkpoint != '':
        print ('Loading model {}'.format(opt.checkpoint))
        checkpoint = torch.load(opt.checkpoint)
        model.load_state_dict(checkpoint['net'])
        niters, start_epoch = checkpoint['iter'], checkpoint['epoch']

    exp_name = int(time.time())
    # set up experiment results directory
    if not opt.test:
        ckpt_dir = os.path.join('checkpoints', exp_name)
        log_dir = os.path.join('logs', exp_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)

    print ('Experiment name: {}\tDevice: {}'.format('test' if opt.test else exp_name, device))

    stats = {
        'losses': [0.] * opt.epochs, 
        'losses_test': [0.] * opt.epochs,
        'loss_ewma': 0.
    }

    predictions, gt = [], []

    def predict(logits, y, lengths, y_lengths, n_show=5, mode='greedy'):
        print ('---------------------------')
        
        n = min(n_show, logits.size(1))
        
        if mode == 'greedy':
            decoded = decoder.decode_greedy(logits, lengths)
        elif mode == 'beam':
            decoded = decoder.decode_beam(logits, lengths)

        predictions.extend(decoded)

        cursor = 0
        for b in range(x.size(0)):
            y_str = ''.join([exp.trainset.vocab[ch - 1] for ch in y[cursor: cursor + y_lengths[b]]])
            gt.append(y_str)
            cursor += y_lengths[b]
            if b < n:
                print ('Test seq {}: {}; pred_{}: {}'.format(b + 1, y_str, mode, decoded[b]))

        print ('---------------------------')


    for ep in range(start_epoch, start_epoch + opt.epochs):
        optimfunc = exp.optim(ep)
        if not opt.test:
            # train loop
            model.train()
            widgets = ['Epoch {}: '.format(ep + 1), Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
            pbar = ProgressBar(widgets=widgets, maxval=len(exp.trainloader)).start()

            for i_batch, sample_batched in enumerate(exp.trainloader):
                pbar.update(i_batch + 1)
                niters += 1
                optimfunc.zero_grad()
                x, y, lengths, y_lengths, idx = sample_batched

                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss_all = crit(F.log_softmax(logits, dim=-1), y, lengths, y_lengths)
                loss = loss_all.mean()
                if torch.isnan(loss).any():
                    print ('Skipping iteration with NaN loss')
                    continue

                weight = torch.ones_like(loss_all)
                dlogits = torch.autograd.grad(loss_all, logits, grad_outputs=weight)[0]
                if opt.curriculum > 0:
                    ratio = opt.curriculum ** ep
                    for b in range(x.size(0)):
                        if exp.trainset.dataset[idx[b]]['mode'] == 1:
                            dlogits[:, b] *= ratio

                logits.backward(dlogits)
                iter_loss = loss.item()
                writer.add_scalar('Train/Loss', iter_loss, niters)
                optimfunc.step()
                stats['losses'][ep] += iter_loss * x.size(0)

            stats['losses'][ep] /= len(exp.trainset)
            pbar.finish()

            # initialise EWMA statistics
            if ep == 0 or opt.checkpoint != '':
                stats['loss_ewma'] = stats['losses'][ep]
            else:
                stats['loss_ewma'] = stats['loss_ewma'] * 0.95 + stats['losses'][ep] * 0.05

        # test loop
        predictions, gt = [], []
        print ('Running evaluation')

        model.eval()
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(exp.testloader):
                x, y, lengths, y_lengths, idx = sample_batched
                x = x.to(device)
                # XXX: invalid if y is moved to CUDA, strange
                # I try to use CuDNN implementation but it goes to native
                # y = y.to(device)
                logits = model(x)
                loss_all = crit(F.log_softmax(logits, dim=-1), y, lengths, y_lengths)
                loss = loss_all.mean()
                if torch.isnan(loss).any():
                    print ('Skipping iteration with NaN test loss')
                    continue
                stats['losses_test'][ep] += loss.item() * x.size(0)
                predict(logits, y, lengths, y_lengths, n_show=5, mode='beam' if opt.test else 'greedy')
        stats['losses_test'][ep] /= len(exp.testset)
        wer = decoder.wer_batch(predictions, gt)
        cer = decoder.cer_batch(predictions, gt)

        if not opt.test:
            writer.add_scalar('Test/Loss', stats['losses_test'][ep], niters)
            writer.add_scalar('Test/WER', wer, niters)
            writer.add_scalar('Test/CER', cer, niters)

            # print epoch statistics
            if ep % opt.print_every == 0:
                print ('Epoch {}: loss={:.5f}, avg={:.5f}, loss_test={:.5f}, loss_test_best={:.5f}'.format(ep + 1, stats['losses'][ep], stats['loss_ewma'], stats['losses_test'][ep], min(stats['losses_test'][: ep + 1])))
                print ('WER: {:.4f}, CER: {:.4f}'.format(wer, cer))
        else:
            print ('Test: loss_test={:.5f}, WER={:.4f}, CER={:.4f}'.format(stats['losses_test'][ep], wer, cer))
            break

        # save best checkpoint by loss
        if ep == 0 or stats['losses_test'][ep] < min(stats['losses_test'][: ep]):
#         if ep % opt.checkpoint_every == 0 or ep == 0 or ep == opt.epochs - 1:
            state = {
                'net': model.state_dict(),
                'optim': optimfunc.state_dict(),
                'epoch': ep + 1,
                'iter': niters,
                'opt': opt,
                'wer': wer,
                'cer': cer
            }
            torch.save(state, os.path.join(ckpt_dir, 'checkpoint_e{:06d}_loss{:.5f}.pth'.format(ep + 1, stats['losses_test'][ep])))
            print ('Saved checkpoint')

