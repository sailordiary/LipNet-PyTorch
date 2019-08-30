# "LipNet: End-to-End Sentence-level Lipreading" in PyTorch
An unofficial PyTorch implementation of the model described in ["LipNet: End-to-End Sentence-level Lipreading"](https://arxiv.org/abs/1611.01599) by Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, and Nando de Freitas. Based on the official Torch implementation.

[![LipNet video](https://img.youtube.com/vi/fa5QGremQf8/0.jpg)](https://www.youtube.com/watch?v=fa5QGremQf8)

## Usage
First, create symbolic links to where you store images and alignments in the `data` folder:

```bash
mkdir data
ln -s PATH_TO_ALIGNS data/align
ln -s PATH_TO_IMAGES data/images
```

Then run the program:

```bash
python3 train_lipnet.py
```

This trains on the "unseen speakers" split. To train on the "overlapped speakers" split:

```bash
python3 train_lipnet.py --test_overlapped
```

The overlapped speakers file list we use (`list_overlapped.json`) is exported directly from the authors' Torch implementation release [here](https://github.com/bshillingford/LipNet/blob/master/util/list_overlapped.t7).

To monitor training progress:
```bash
tensorboard --logdir logs
```

The `images` folder should be organised as:
```
├── s1
│   ├── bbaf2n
│   │   ├── mouth_000.png
│   │   ├── mouth_001.png
...
```

And the `align` folder:
```
├── s1
│   ├── bbaf2n.align
│   ├── bbaf3s.align
│   ├── bbaf4p.align
...
```

That's it! You can specify the GPU to use in the program where the environment variable `CUDA_VISIBLE_DEVICES` is set. Feel free to play around with the parameters.

## Dependencies
- Python 3.x
- PyTorch 1.1+ (for native CTCLoss and TensorBoard support; we highly recommend using nightly builds, because PyTorch CTC is quite buggy and often fixes are not reflected in due course.)
- tensorboardX (if you are not using PyTorch 1.1+, or your TensorFlow version is incompatible with native PyTorch Tensorboard support)
- [ctcdecode](https://github.com/parlance/ctcdecode) (for beam search decoding)
- torchsummary
- progressbar2
- editdistance
- scikit-image
- torchvision
- pillow

## Results
TODO

## Pending
- Add saliency visualisation
- Add preprocessing code

