[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)

# SmartPatch: Improving Handwritten Word Imitation with Patch Discriminators



## Dataset preparation

Training is done on the [IAM](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) dataset.
First download the IAM word level dataset, then execute `prepare_dataset.sh [folder of iamdb dataset]` to prepared the dataset for training.  
Afterwards, refer your folder in `load_data.py` (search `img_base`). I recommend to keep the folder at the default `./dataset`.
If you want to have the actual storage somewhere else, it is simpler to symlink your data to `./dataset` then changing it and risk something breaking.

## How to train it?

run the training with:

```bash
python main_run.py id save_weights image_folder [--additional discriminator]
```
`save_weights` is the path for storing checkpoints as `.model` and logging dictonaries as `.pickle` files. The `image_folder` prints out sample images during the training process.

`[id]` should be either:
- 0 if you want to start a fresh training session
- the id of the model in the `save_weights` directory, e.g. 1000 if you have a model named `contran-1000.model`.
- -1 if you want to train from the latest checkpoint

You can choose the additional discriminators using the flags (run --help for info). Additionally to the discriminators described in the paper, there's a discriminator that injects style information, though this one performs worse than the others according to our benchmarks.

## Benchmarking

You can generate a synthetic pseudo-IAM dataset by running
```bash
python generatePseudoIAM.py save_weights destination [--additional discriminator]
```
For convenience `genPseudoIam.sh` will generate all synthetic datasets into the HTR Benchmarking directory. (you may need to change the location of your models)
For training the HTR system, look at the `test.py` in the HTR Benchmarking directory and at [the original repository](https://github.com/omni-us/research-seq2seq-HTR)

## Acknowledgements

We thank the researchers at for releasing the code to the [HTR](https://github.com/omni-us/research-seq2seq-HTR) and [GANwriting](https://github.com/omni-us/research-GANwriting) systems on which this code is based.
