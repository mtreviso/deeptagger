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
 
See next how to install and use it.


## Installation

First, clone DeepTagger using `git`:

```sh
git clone https://github.com/mtreviso/deeptagger.git
```

 Then, `cd` to the DeepTagger folder and run the install 
 command:
```sh
cd deeptagger
python setup.py install
```

## Getting started

#### Extracting PoS tags

```python
import deeptagger

model = deeptagger.load('path/to/saved-model-dir/')
tags, probs = model.predict('The lazy fox jumps over the lazy dog.')
```

Where `tags` is a list of strings and `probs` is a list of 
floats. 

#### Training a PoS tagger model
```python
import deeptagger

args = {
  train_path: "path/to/train.txt",
  dev_path: "path/to/dev.txt",
  del_word: " ",
  del_tag: "_"    
}
model = deeptagger.train(args)
model.save('path/to/model-dir/')
```

You can view all arguments and their meaning by calling `deeptagger.help()`


#### Invoking in standalone mode (no install required)

For predicting tags:
```
python -m deeptagger predict --text "The lazy fox jumps over the lazy dog" --load path/to/saved-model-dir/
```

For training a model:
```python
python -m deeptager train :args:
```

You can obtain more info for each command by passing the `--help` flag.


## Examples

In the [examples folder](https://github.com/mtreviso/deeptagger/tree/master/examples) of the repository, you will find examples on real Brazilian Portuguese and English corpora.


## Arguments

TODO


## Contributing
Anyone can help make this project better - read [CONTRIBUTING](CONTRIBUTING.md) to get started!


## License
MIT. See the [LICENSE](LICENSE) file for more details.

