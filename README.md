[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)

# GANwriting: Content-Conditioned Generation of Styled Handwritten Word Images


## Software environment:

- Ubuntu 16.04 x64
- Python 3.7
- PyTorch 1.4

## Dataset preparation

The main experiments are run on [IAM](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) since it's a multi-writer dataset. Furthermore, when you have obtained a pretrained model on IAM, you could apply it on other datasets as evaluation, such as [GW](http://www.fki.inf.unibe.ch/databases/iam-historical-document-database/washington-database),  [RIMES](http://www.a2ialab.com/doku.php?id=rimes_database:start), [Esposalles](http://dag.cvc.uab.es/the-esposalles-database/) and
[CVL](https://cvl.tuwien.ac.at/research/cvl-databases/an-off-line-database-for-writer-retrieval-writer-identification-and-word-spotting/).

## How to train it?

First download the IAM word level dataset, then execute `prepare_dataset.sh [folder of iamdb dataset]` to prepared the dataset for training.  
Afterwards, refer your folder in `load_data.py` (search `img_base`). Standard is `./dataset`

Then run the training with:

```bash
./python 0 [--additional discriminator]
```

The different options for additional discriminators can be found in the help message.

**Note**: During the training process, two folders will be created:
`imgs/` contains the intermediate results of one batch (you may like to check the details in function `write_image` from `modules_tro.py`), and `save_weights/` consists of saved weights ending with `.model` and `.pickle` files that contain dicts of logging data.

If you have already trained a model, you can use that model for further training by running:

```bash
./python [id] [--additional discriminator]
```

In this case, `[id]` should be the id of the model in the `save_weights` directory, e.g. 1000 if you have a model named `contran-1000.model`.
