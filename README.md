[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)

# SmartPatch: Improving Handwritten Word Imitation with Patch Discriminators
![GANwritingVsSmartPatch](https://github.com/MattAlexMiracle/SmartPatch/blob/main/ComparisonGANwritingVsSmartPatch.jpg)


This is the official code for [SmartPatch: Improving Handwritten Word Imitation with Patch Discriminators](https://link.springer.com/chapter/10.1007/978-3-030-86549-8_18) published in ICDAR21.
## Abstract

As of recent generative adversarial networks have allowed for big leaps in the realism of generated images in diverse domains, not the least of which being handwritten text generation. The generation of realistic-looking hand-written text is important because it can be used for data augmentation in handwritten text recognition (HTR) systems or human-computer interaction.
We propose SmartPatch, a new technique increasing the performance of current state-of-the-art methods by augmenting the training feedback with a tailored solution to mitigate pen-level artifacts. We combine the well-known patch loss with information gathered from the parallel trained handwritten text recognition system and the separate characters of the word. This leads to a more enhanced local discriminator and results in more realistic and higher-quality generated handwritten words.

## Dataset preparation

Training is done on the [IAM](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) dataset.  
First download the IAM word level dataset, then execute `prepare_dataset.sh [folder of iamdb dataset]` to prepared the dataset for training.  
Afterwards, refer your folder in `load_data.py` (search `img_base`).  
I recommend to keep the folder at the default `./dataset`.  
If you want to move the dataset somewhere else, it is simpler to symlink your storage location to `./dataset`, rather than changing the dataset location everywhere.

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

You can choose the additional discriminators using the flags (run --help for info).
Note that the nomenclature in this repo is a little different than the one used in the paper:
- Naive patches only use rolling patches (just like in the paper)
- Smart patches center the patches on the characters, but does **not** inject character information. In the paper this is called "centered patch"
- Character patches center the patches and inject the character information. This is what the paper refers to as "smartpatch"
Additionally to the discriminators described in the paper, there's a discriminator that injects style information, though this one performs worse than the others according to our benchmarks.

## Benchmarking

You can generate a synthetic pseudo-IAM dataset by running
```bash
python generatePseudoIAM.py save_weights destination [--additional discriminator]
```
For convenience `genPseudoIam.sh` will generate all synthetic datasets into the HTR Benchmarking directory. (you may need to change the location of your models)
For training the HTR system, look at the `test.py` in the HTR Benchmarking directory and at [the original repository](https://github.com/omni-us/research-seq2seq-HTR)

## How to cite
Please cite
```
@InProceedings{10.1007/978-3-030-86549-8_18,
author="Mattick, Alexander
and Mayr, Martin
and Seuret, Mathias
and Maier, Andreas
and Christlein, Vincent",
editor="Llad{\'o}s, Josep
and Lopresti, Daniel
and Uchida, Seiichi",
title="SmartPatch: Improving Handwritten Word Imitation with Patch Discriminators",
booktitle="Document Analysis and Recognition -- ICDAR 2021",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="268--283",
abstract="As of recent generative adversarial networks have allowed for big leaps in the realism of generated images in diverse domains, not the least of which being handwritten text generation. The generation of realistic-looking handwritten text is important because it can be used for data augmentation in handwritten text recognition (HTR) systems or human-computer interaction. We propose SmartPatch, a new technique increasing the performance of current state-of-the-art methods by augmenting the training feedback with a tailored solution to mitigate pen-level artifacts. We combine the well-known patch loss with information gathered from the parallel trained handwritten text recognition system and the separate characters of the word. This leads to a more enhanced local discriminator and results in more realistic and higher-quality generated handwritten words.",
isbn="978-3-030-86549-8"
}
```

## Acknowledgements

We thank the researchers at for releasing the code to the [HTR](https://github.com/omni-us/research-seq2seq-HTR) and [GANwriting](https://github.com/omni-us/research-GANwriting) systems on which this code is based.

