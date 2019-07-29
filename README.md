# "LipNet: End-to-End Sentence-level Lipreading" in PyTorch
A PyTorch implementation of the model described in ["LipNet: End-to-End Sentence-level Lipreading"](https://arxiv.org/abs/1611.01599) by Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, and Nando de Freitas.

## Usage
First, create symbolic links to where you store images and alignments in the `data` folder:

TODO

## Dependencies
- Python 3.x
- PyTorch 1.1+ (for native CTCLoss and TensorBoard support; we highly recommend using nightly builds, because PyTorch CTC is quite buggy and often fixes are not reflected in due course.)
- tensorboardX (if you are not using PyTorch 1.1+, or your TensorFlow version is incompatible with native PyTorch Tensorboard support)
- ctcdecode (for beam search decoding)
- torchsummary
- progressbar2
- editdistance

## Results
TODO

## Pending
- Refactor code (WIP)
- Add epoch WER and CER statistics
- Add saliency visualisation
- Add preprocessing code

