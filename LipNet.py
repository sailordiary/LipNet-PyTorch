from exp.model import Exp
from progressbar import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import toml
import time
import random
from collections import OrderedDict
from modules.CTCDecoder import Decoder
from warpctc_pytorch import CTCLoss


if __name__ == '__main__':
    print ('Loading options...')
    opt = toml.loads(open('options.toml', 'r').read())

    # construct model
    exp = Exp(opt)
    model = exp.model
    decoder = Decoder(exp.trainset.vocab, lm_path=opt['general']['lm_path'])
    crit = CTCLoss()
    if opt['general']['cuda']:
        model = model.cuda()
        #model = nn.DataParallel(model).cuda()
        crit = crit.cuda()
    
    if opt['general']['use_keras_weights']:
        from nn_transfer import transfer
        transfer.convert_lipnet(model, 'nn_transfer/unseen-weights178.h5')
    if opt['general']['freeze_conv']:
        def freeze(m):
            m.requires_grad = False
        model.conv.apply(freeze)
    
    # load model
    try:
        niters = opt['general']['start_iter']
    except:
        niters = 0
    start_epoch = 0
    if opt['general']['checkpoint'] != '':
        print ('Loading model...')
        checkpoint = torch.load(opt['general']['checkpoint'])
        # remove "module.*" from DataParallel
        new_state_dict = OrderedDict()
        for k, v in checkpoint['net'].items():
            if k[7:] == 'module.':
                name = k[7:]
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        niters = checkpoint['iter']
        start_epoch = checkpoint['epoch'] - 1

    exp_name = int(time.time())
    random.seed(opt['general']['seed'])
    torch.manual_seed(opt['general']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt['general']['seed'])

    # set up experiment results directory
    ckpt_dir = 'ckpt/%d/' % exp_name
    log_dir = 'log/%d/' % exp_name
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    for ep in range(start_epoch, start_epoch + opt['train']['nepochs']):
        optimfunc = exp.optim(ep)
        # training
        widgets = ['Epoch %d: ' % (ep + 1), Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=len(exp.trainloader)).start()
        for i_batch, sample_batched in enumerate(exp.trainloader):
            pbar.update(i_batch + 1)
            niters += 1
            optimfunc.zero_grad()
            x, labels, act_lens, label_lens, ids, subs = sample_batched
            if opt['general']['cuda']:
                x = x.cuda()
            acts = model(x)
            loss = crit(acts, labels, act_lens, label_lens)
            if opt['train']['curriculum'] > 0:
                ratio = opt['train']['curriculum'] ** ep
                for b in range(0, x.size(0)):
                    id = ids[b]
                    if exp.trainset.dataset[id]['mode'] == 1:
                        loss[b] *= ratio
            loss = loss.mean()

            # skip this iteration if the loss is NaN
            if not torch.isnan(loss) and loss >= -1000000 and loss <= 1000000:
                writer.add_scalar('Train/Loss', loss, niters)
                loss.backward()
                optimfunc.step()
            else:
                print ('Logits:', F.softmax(acts).min(), F.softmax(acts).max(), acts.size(0), acts.size(1))
                print ('Skipping iteration with NaN loss')
            if niters % opt['general']['print_every'] == 0:
                print ('Epoch %d, iter %d: Loss = %.2f' % (ep + 1, niters, loss))
                print ('---------------------------')
                gt = subs[:5]
                decoded = decoder.decode_beam(acts, act_lens)[:5]
                print ('GROUND TRUTH:', gt)
                print ('BEST PATH:', decoder.decode_greedy(acts, act_lens)[:5])
                print ('BEAM SEARCH:', decoded)
                wer = decoder.wer_batch(decoded, gt)
                cer = decoder.cer_batch(decoded, gt)
                writer.add_scalar('Train/WER', wer, niters)
                writer.add_scalar('Train/CER', cer, niters)
                print ('---------------------------')
        pbar.finish()
        if ep % opt['general']['checkpoint_every'] == 0 or ep == 0 or ep == opt['train']['nepochs'] - 1:
            state = {
                'net': exp.model.state_dict(),
                'epoch': ep + 1,
                'iter': niters
            }
            print ('Saving checkpoint...')
            torch.save(state, ckpt_dir + 'checkpoint_e%d_iter%d.pth' % (ep + 1, niters))
        # validation
        print ('Running evaluation')
        model.eval()
        with torch.no_grad():
            dataiter = iter(exp.trainloader)
            x, _, act_lens, _, _, subs = dataiter.next()
            if opt['general']['cuda']:
                x = x.cuda()
            acts = model(x)
            print ('---------------------------')
            gt = subs[:5]
            decoded = decoder.decode_beam(acts, act_lens)[:5]
            print ('GROUND TRUTH:', gt)
            print ('BEST PATH:', decoder.decode_greedy(acts, act_lens)[:5])
            print ('BEAM SEARCH:', decoded)
            print ('---------------------------')
            wer = decoder.wer_batch(decoded, gt)
            cer = decoder.cer_batch(decoded, gt)
            print ('WER: %.2f   CER: %.2f' % (wer * 100, cer * 100))
            writer.add_scalar('Test/WER', wer, niters)
            writer.add_scalar('Test/CER', cer, niters)
            print ('---------------------------')
        model.train()
