# DeepTagger

Part-of-Speech (PoS) tagger based on Deep Learning.

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/built%20with-Python3-red.svg" alt="built with Python3" /></a>
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
pip install --user pipenv
pipenv install
pipenv shell
```

Run the install command:
```sh
python setup.py install
```

Please note that since Python 3 is required, all the above commands (pip/python) 
have to bounded to the Python 3 version.

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
tagger.train(args)
tagger.save('path/to/model-dir/')
```

You can view all arguments and their meaning by calling `python3 -m deeptagger --help`. 
Or take a look at the [arguments quick reference table](#arguments).


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

In the [experiments folder](https://github.com/mtreviso/deeptagger/tree/master/experiments) 
of the repository, you will find examples on real Brazilian Portuguese and English corpora.


## Arguments

#### Standalone usage
```
python3 -m deeptagger {predict,train} :args:
```

#### Arguments quick reference table
<table class="rich-diff-level-one"> <thead> <tr>
<th width="28%">Option</th>
<th width="14%">Default</th>
<th>Description</th>
</tr> </thead> <tbody>
<tr>
<td>
<code>-h</code> <code>--help</code>
</td>
<td></td>
<td>Show this help message and exit</td>
</tr>
<tr>
<td>
<code>-o</code> <code>--output-dir</code>
</td>
<td><code>None</code></td>
<td>Output files for this run under this dir. If not specified, it will create a timestamp dir inside <code>runs</code> dir.</td>
</tr>
<tr>
<td><code>--seed</code></td>
<td><code>42</code></td>
<td>Random seed</td>
</tr>
<tr>
<td><code>--gpu-id</code></td>
<td><code>None</code></td>
<td>Use CUDA on the listed devices</td>
</tr>
<tr>
<td><code>--debug</code></td>
<td></td>
<td>Debug mode.</td>
</tr>
<tr>
<td><code>--save</code></td>
<td><code>None</code></td>
<td>Output dir for saving the model</td>
</tr>
<tr>
<td><code>--load</code></td>
<td><code>None</code></td>
<td>Input dir for loading the model</td>
</tr>
<tr>
<td><code>--resume-epoch</code></td>
<td><code>None</code></td>
<td>Resume training from a specific epoch saved in a previous execution <code>runs/output-dir</code>
</td>
</tr>
<tr>
<td><code>--train-path</code></td>
<td><code>None</code></td>
<td>Path to training file</td>
</tr>
<tr>
<td><code>--dev-path</code></td>
<td><code>None</code></td>
<td>Path to validation file</td>
</tr>
<tr>
<td><code>--test-path</code></td>
<td><code>None</code></td>
<td>Path to validation file</td>
</tr>
<tr>
<td><code>--del-word</code></td>
<td></td>
<td>Delimiter token to split sentence tokens</td>
</tr>
<tr>
<td><code>--del-tag</code></td>
<td><code>_</code></td>
<td>Delimiter token to split word tokens from tag tokens</td>
</tr>
<tr>
<td><code>--max-length</code></td>
<td><code>inf</code></td>
<td>Maximum sequence length</td>
</tr>
<tr>
<td><code>--min-length</code></td>
<td><code>0</code></td>
<td>Minimum sequence length.</td>
</tr>
<tr>
<td><code>--vocab-size</code></td>
<td><code>None</code></td>
<td>Max size of the vocabulary.</td>
</tr>
<tr>
<td><code>--vocab-min-frequency</code></td>
<td><code>1</code></td>
<td>Min word frequency for vocabulary.</td>
</tr>
<tr>
<td><code>--keep-rare-with-embedding</code></td>
<td></td>
<td>Keep words that occur less then min-frequency but are in embeddings vocabulary.</td>
</tr>
<tr>
<td><code>--add-embeddings-vocab</code></td>
<td></td>
<td>Add words from embeddings vocabulary to source/target vocabulary.</td>
</tr>
<tr>
<td><code>--embeddings-format</code></td>
<td><code>None</code></td>
<td>Word embeddings format. See README for specific formatting instructions.</td>
</tr>
<tr>
<td><code>--embeddings-path</code></td>
<td><code>None</code></td>
<td>Path to word embeddings file for source.</td>
</tr>
<tr>
<td><code>--model</code></td>
<td><code>simple_lstm</code></td>
<td>Model architecture. Choices: <code>{simple_lstm, rcnn}</code>
</td>
</tr>
<tr>
<td><code>--word-embeddings-size</code></td>
<td><code>100</code></td>
<td>Size of word embeddings.</td>
</tr>
<tr>
<td><code>--conv-size</code></td>
<td><code>100</code></td>
<td>Size of convolution 1D. a.k.a. number of channels.</td>
</tr>
<tr>
<td><code>--kernel-size</code></td>
<td><code>7</code></td>
<td>Size of the convolving kernel.</td>
</tr>
<tr>
<td><code>--pool-length</code></td>
<td><code>3</code></td>
<td>Size of pooling window.</td>
</tr>
<tr>
<td><code>--dropout</code></td>
<td><code>0.5</code></td>
<td>Dropout rate applied after RNN layers.</td>
</tr>
<tr>
<td><code>--emb-dropout</code></td>
<td><code>0.4</code></td>
<td>Dropout rate applied after embedding layers.</td>
</tr>
<tr>
<td><code>--rnn-type</code></td>
<td><code>rnn</code></td>
<td>RNN cell type. Choices are: <code>{rnn, gru, lstm}</td>
</tr>
<tr>
<td><code>--bidirectional</code></td>
<td></td>
<td>Set RNNs to be bidirectional.</td>
</tr>
<tr>
<td><code>--sum-bidir</code></td>
<td></td>
<td>Sum outputs of bidirectional states. By default they are concatenated.</td>
</tr>
<tr>
<td><code>--freeze-embeddings</code></td>
<td></td>
<td>Freeze embedding weights during training.</td>
</tr>
<tr>
<td><code>--loss-weights</code></td>
<td><code>same</code></td>
<td>Weights for penalize each class in loss calculation. <code>same</code> will give each class the same weights. <code>balanced</code> will give more weight for minority classes.</td>
</tr>
<tr>
<td><code>--hidden-size</code></td>
<td><code>[100]</code></td>
<td>Number of neurons on the hidden layers. If you pass more sizes, then more then one hidden layer will be created. Please, take a look to your selected model documentation before setting this option.</td>
</tr>
<tr>
<td><code>--use-prefixes</code></td>
<td></td>
<td>Use prefixes as feature.</td>
</tr>
<tr>
<td><code>--prefix-embeddings-size</code></td>
<td><code>100</code></td>
<td>Size of prefix embeddings.</td>
</tr>
<tr>
<td><code>--prefix-min-length</code></td>
<td><code>1</code></td>
<td>Min length of prefixes.</td>
</tr>
<tr>
<td><code>--prefix-max-length</code></td>
<td><code>5</code></td>
<td>Max length of prefixes.</td>
</tr>
<tr>
<td><code>--use-suffixes</code></td>
<td></td>
<td>Use suffixes as feature.</td>
</tr>
<tr>
<td><code>--suffix-embeddings-size</code></td>
<td><code>100</code></td>
<td>Size of suffix embeddings.</td>
</tr>
<tr>
<td><code>--suffix-min-length</code></td>
<td><code>1</code></td>
<td>Min length of suffixes.</td>
</tr>
<tr>
<td><code>--suffix-max-length</code></td>
<td><code>5</code></td>
<td>Max length of suffixes.</td>
</tr>
<tr>
<td><code>--use-caps</code></td>
<td></td>
<td>Use capitalization as feature.</td>
</tr>
<tr>
<td><code>--caps-embeddings-size</code></td>
<td><code>100</code></td>
<td>Size of capitalization embeddings.</td>
</tr>
<tr>
<td><code>--epochs</code></td>
<td><code>10</code></td>
<td>Number of epochs for training.</td>
</tr>
<tr>
<td><code>--shuffle</code></td>
<td></td>
<td>Shuffle train data before each epoch.</td>
</tr>
<tr>
<td><code>--train-batch-size</code></td>
<td><code>64</code></td>
<td>Maximum batch size for training.</td>
</tr>
<tr>
<td><code>--dev-batch-size</code></td>
<td><code>64</code></td>
<td>Maximum batch size for evaluating.</td>
</tr>
<tr>
<td><code>--dev-checkpoint-epochs</code></td>
<td><code>1</code></td>
<td>Perform an evaluation on dev set after X epochs.</td>
</tr>
<tr>
<td><code>--save-checkpoint-epochs</code></td>
<td><code>1</code></td>
<td>Save a checkpoint every X epochs.</td>
</tr>
<tr>
<td><code>--save-best-only</code></td>
<td></td>
<td>Save only when validation loss is improved.</td>
</tr>
<tr>
<td><code>--early-stopping-patience</code></td>
<td><code>0</code></td>
<td>Stop training if validation loss is not improved after passing X epochs. By defaultthe early stopping procedure is not applied.</td>
</tr>
<tr>
<td><code>--restore-best-model</code></td>
<td></td>
<td>Whether to restore the model state from the epoch with the best validation loss found. If False, the model state obtained at the last step of training is used.</td>
</tr>
<tr>
<td><code>--final-report</code></td>
<td></td>
<td>Whether to report a table with the stats history for train/dev/test set after training.</td>
</tr>
<tr>
<td><code>--optimizer</code></td>
<td><code>sgd</code></td>
<td>Optimization method. Choices: <code>{adam, adadelta, adagrad, adamax, sparseadam, sgd, asgd, rmsprop}</code>
</td>
</tr>
<tr>
<td><code>--learning-rate</code></td>
<td><code>None</code></td>
<td>Starting learning rate. Let unseted to use default values.</td>
</tr>
<tr>
<td><code>--weight-decay</code></td>
<td><code>None</code></td>
<td>L2 penalty. Used for all algorithms. Let unseted to use default values.</td>
</tr>
<tr>
<td><code>--lr-decay</code></td>
<td><code>None</code></td>
<td>Learning reate decay. Used only for: adagrad. Let unseted to use default values.</td>
</tr>
<tr>
<td><code>--rho</code></td>
<td><code>None</code></td>
<td>Coefficient used for computing a running average of squared. Used only for: adadelta. Let unseted to use default values.</td>
</tr>
<tr>
<td><code>--beta0</code></td>
<td><code>None</code></td>
<td>Coefficient used for computing a running averages of gradient and its squared. Used only for: adam, sparseadam, adamax. Let unseted to use default values.</td>
</tr>
<tr>
<td><code>--beta1</code></td>
<td><code>None</code></td>
<td>Coefficient used for computing a running averages of gradient and its squared. Used only for: adam, sparseadam, adamax. Let unseted to use default values.</td>
</tr>
<tr>
<td><code>--momentum</code></td>
<td><code>None</code></td>
<td>Momentum factor. Used only for: sgd and rmsprop. Let unseted to use default values.</td>
</tr>
<tr>
<td><code>--nesterov</code></td>
<td><code>None</code></td>
<td>Enables Nesterov momentum. Used only for: sgd. Let unseted to use default values.</td>
</tr>
<tr>
<td><code>--alpha</code></td>
<td><code>None</code></td>
<td>Smoothing constant. Used only for: rmsprop. Let unseted to use default values.</td>
</tr>
</tbody>
</table>


## Contributing
Anyone can help make this project better. Feel free to fork and open a pull request - read [CONTRIBUTING](CONTRIBUTING.md) to get started!


## Cite

If you use DeepTagger, please cite this paper:

```
Under publication...
```

## License
MIT. See the [LICENSE](LICENSE) file for more details.


