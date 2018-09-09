# DeepTagger

Part-of-Speech (PoS) tagger based on Deep Learning.

[![Build Status](https://travis-ci.com/mtreviso/deeptagger.svg?token=x2rssmYXXPdD5p8iqKt2&branch=master)](https://travis-ci.com/mtreviso/deeptagger)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

DeepTagger is a simple python3 tool for extracting PoS tags 
from raw texts and training a PoS model for languages with 
labeled corpora. 
DeepTagger models are implemented using PyTorch.

Currently, we have trained models for Brazilian-Portuguese 
and English. You can download them in the following links:

- [PT-BR](http://mtreviso.github.io/deeptagger-ptbr-models)
- [EN](http://mtreviso.github.io/deeptagger-en-models)
 
See next how to install and use DeepTagger.


## Installation 

First, clone this repository using `git`:

```sh
git clone https://github.com/mtreviso/deeptagger.git
```

 Then, `cd` to the DeepTagger folder:
```sh
cd deeptagger
```

Automatically create a Python virtualenv and install all dependencies 
using `pipenv install`. And then activate the virtualenv with `pipenv shell`:
```sh
pip install pipenv
pipenv install
pipenv shell
```

Run the install command:
```sh
python setup.py install
```

Please note that since Python 3 is required, all the above commands (pip/python) 
have to be the Python 3 version.

## Getting started

#### Extracting PoS tags

Using gpu to extract tags:

```python
from deeptagger import Tagger
tagger = Tagger(gpu_id=1)
tagger.load('path/to/saved-model-dir/')
tags = tagger.predict_classes('Há livros escritos para evitar espaços vazios na estante .')
```

Where `tags` is a list of strings. Alternatively, you can predict 
probabilities for each class with `model.predict_probas()`.

#### Training a PoS tagger model
```python
from deeptagger import Tagger
args = {
  'train_path': "path/to/train.txt",
  'dev_path': "path/to/dev.txt",
  'del_word': " ",
  'del_tag': "_"
}
tagger = Tagger()
model = tagger.train(args)
model.save('path/to/model-dir/')
```

You can view all arguments and their meaning by calling `deeptagger.help()`. 
Or take a look at the section [arguments](#arguments).


#### Invoking in standalone mode (no install required)

For predicting tags:
```
python -m deeptagger predict --load path/to/saved-model-dir/ --text "Há livros escritos para evitar espaços vazios na estante ."
```

For training a model:
```
python -m deeptagger train :args:
```

You can obtain more info for each command by passing the `--help` flag.


## Examples

In the [examples folder](https://github.com/mtreviso/deeptagger/tree/master/examples) of the repository, you will find examples on real Brazilian Portuguese and English corpora.


## Arguments

#### Standalone usage
```
python3 -m deeptagger {predict,train} :args:
```

#### Arguments quick reference table
|Option                      |Default      |Description                                                                                                                                                                                              |
|----------------------------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`-h` `--help`               |`-`          |Show this help message and exit                                                                                                                                                                          |
|`-o` `--output-dir`         |`None`       |Output files for this run under this dir. If not specified, it will create a timestamp dir inside `runs` dir.                                                                                            |
|`--seed`                    |`42`         |Random seed                                                                                                                                                                                              |
|`--gpu-id`                  |`None`       |Use CUDA on the listed devices                                                                                                                                                                           |
|`--debug`                   |`-`          |Debug mode.                                                                                                                                                                                              |
|`--save`                    |`None`       |Output dir for saving the model                                                                                                                                                                          |
|`--load`                    |`None`       |Input dir for loading the model                                                                                                                                                                          |
|`--resume-epoch`            |`None`       |Resume training from a specific epoch saved in a previous execution `runs/output-dir`                                                                                                                    |
|`--train-path`              |`None`       |Path to training file                                                                                                                                                                                    |
|`--dev-path`                |`None`       |Path to validation file                                                                                                                                                                                  |
|`--test-path`               |`None`       |Path to validation file                                                                                                                                                                                  |
|`--del-word`                |` `          |Delimiter token to split sentence tokens                                                                                                                                                                 |
|`--del-tag`                 |`_`          |Delimiter token to split word tokens from  tag tokens                                                                                                                                                    |
|`--max-length`              |`inf`        |Maximum sequence length                                                                                                                                                                                  |
|`--min-length`              |`0`          |Minimum sequence length.                                                                                                                                                                                 |
|`--vocab-size`              |`None`       |Max size of the vocabulary.                                                                                                                                                                              |
|`--vocab-min-frequency`     |`1`          |Min word frequency for vocabulary.                                                                                                                                                                       |
|`--keep-rare-with-embedding`|`-`          |Keep words that occur less then min-frequency but are in embeddings vocabulary.                                                                                                                          |
|`--add-embeddings-vocab`    |`-`          |Add words from embeddings vocabulary to source/target vocabulary.                                                                                                                                        |
|`--embeddings-format`       |`None`       |Word embeddings format. See README for specific formatting instructions.                                                                                                                                 |
|`--embeddings-path`         |`None`       |Path to word embeddings file for source.                                                                                                                                                                 |
|`--model`                   |`simple_lstm`|Model architecture. Choices: `{simple_lstm, rcnn}`                                                                                                                                                       |
|`--word-embeddings-size`    |`100`        |Size of word embeddings.                                                                                                                                                                                 |
|`--conv-size`               |`100`        |Size of convolution 1D. a.k.a. number of channels.                                                                                                                                                       |
|`--kernel-size`             |`7`          |Size of the convolving kernel.                                                                                                                                                                           |
|`--pool-length`             |`3`          |Size of pooling window.                                                                                                                                                                                  |
|`--dropout`                 |`0.5`        |Dropout rate applied after RNN layers.                                                                                                                                                                   |
|`--emb-dropout`             |`0.4`        |Dropout rate applied after embedding layers.                                                                                                                                                             |
|`--bidirectional`           |`-`          |Set RNNs to be bidirectional.                                                                                                                                                                            |
|`--sum-bidir`               |`-`          |Sum outputs of bidirectional states. By default they are concatenated.                                                                                                                                   |
|`--freeze-embeddings`       |`-`          |Freeze embedding weights during training.                                                                                                                                                                |
|`--loss-weights`            |`same`       |Weights for penalize each class in loss calculation. `same` will give each class the same weights. `balanced` will give more weight for minority classes.                                                |
|`--hidden-size`             |`[100]`      |Number of neurons on the hidden layers. If you pass more sizes, then more then one hidden layer will be created. Please, take a look to your selected model documentation before setting this option.    |
|`--use-prefixes`            |`-`          |Use prefixes as feature.                                                                                                                                                                                 |
|`--prefix-embeddings-size`  |`100`        |Size of prefix embeddings.                                                                                                                                                                               |
|`--prefix-min-length`       |`1`          |Min length of prefixes.                                                                                                                                                                                  |
|`--prefix-max-length`       |`5`          |Max length of prefixes.                                                                                                                                                                                  |
|`--use-suffixes`            |`-`          |Use suffixes as feature.                                                                                                                                                                                 |
|`--suffix-embeddings-size`  |`100`        |Size of suffix embeddings.                                                                                                                                                                               |
|`--suffix-min-length`       |`1`          |Min length of suffixes.                                                                                                                                                                                  |
|`--suffix-max-length`       |`5`          |Max length of suffixes.                                                                                                                                                                                  |
|`--use-caps`                |`-`          |Use capitalization as feature.                                                                                                                                                                           |
|`--caps-embeddings-size`    |`100`        |Size of capitalization embeddings.                                                                                                                                                                       |
|`--epochs`                  |`10`         |Number of epochs for training.                                                                                                                                                                           |
|`--shuffle`                 |`-`          |Shuffle train data before each epoch.                                                                                                                                                                    |
|`--train-batch-size`        |`64`         |Maximum batch size for training.                                                                                                                                                                         |
|`--dev-batch-size`          |`64`         |Maximum batch size for evaluating.                                                                                                                                                                       |
|`--dev-checkpoint-epochs`   |`1`          |Perform an evaluation on dev set after X epochs.                                                                                                                                                         |
|`--save-checkpoint-epochs`  |`1`          |Save a checkpoint every X epochs.                                                                                                                                                                        |
|`--save-best-only`          |`-`          |Save only when validation loss is improved.                                                                                                                                                              |
|`--early-stopping-patience` |`0`          |Stop training if validation loss is not improved after passing X epochs. By defaultthe early stopping procedure is not applied.                                                                          |
|`--restore-best-model`      |`-`          |Whether to restore the model state from the epoch with the best validation loss found. If False, the model state obtained at the last step of training is used.                                          |
|`--final-report`            |`-`          |Whether to report a table with the stats history for train/dev/test set after training.                                                                                                                  |
|`--optimizer`               |`sgd`        |Optimization method. Choices: `{adam, adadelta, adagrad, adamax, sparseadam, sgd, asgd, rmsprop}`                                                                                                        |
|`--learning-rate`           |`None`       |Starting learning rate. Let unseted to use default values.                                                                                                                                               |
|`--weight-decay`            |`None`       |L2 penalty. Used for all algorithms. Let unseted to use default values.                                                                                                                                  |
|`--lr-decay`                |`None`       |Learning reate decay. Used only for: adagrad. Let unseted to use default values.                                                                                                                         |
|`--rho`                     |`None`       |Coefficient used for computing a running average of squared. Used only for: adadelta. Let unseted to use default values.                                                                                 |
|`--beta0`                   |`None`       |Coefficient used for computing a running averages of gradient and its squared. Used only for: adam, sparseadam, adamax. Let unseted to use default values.                                               |
|`--beta1`                   |`None`       |Coefficient used for computing a running averages of gradient and its squared. Used only for: adam, sparseadam, adamax. Let unseted to use default values.                                               |
|`--momentum`                |`None`       |Momentum factor. Used only for: sgd and rmsprop. Let unseted to use default values.                                                                                                                      |
|`--nesterov`                |`None`       |Enables Nesterov momentum. Used only for: sgd. Let unseted to use default values.                                                                                                                        |
|`--alpha`                   |`None`       |Smoothing constant. Used only for: rmsprop. Let unseted to use default values.                                                                                                                           |


## Contributing
Anyone can help make this project better - read [CONTRIBUTING](CONTRIBUTING.md) to get started!


## License
MIT. See the [LICENSE](LICENSE) file for more details.

