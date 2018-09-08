import logging
from collections import defaultdict
from pathlib import Path

import torch
from torchtext.data import Field

from deeptagger import constants
from deeptagger.vectors import AvailableEmbeddings
from deeptagger.vocabulary import Vocabulary


def load_vectors(options):
    vectors = None
    if options.embeddings_path is not None:
        logging.info('Loading {} word embeddings from: {}'.format(
            options.embeddings_format, options.embeddings_path))
        word_emb_cls = AvailableEmbeddings[options.embeddings_format]
        vectors = word_emb_cls(options.embeddings_path, binary=False)
    return vectors


def build_vocabs(fields_tuples, train_dataset, all_datasets, options):
    vectors = load_vectors(options)
    dict_fields = defaultdict(lambda: None)
    dict_fields.update(dict(fields_tuples))
    words_field = dict_fields['words']
    tags_field = dict_fields['tags']
    words_field.build_vocab(train_dataset,
                            vectors=vectors,
                            max_size=options.vocab_size,
                            min_freq=options.vocab_min_frequency,
                            rare_with_vectors=options.keep_rare_with_embedding,
                            add_vectors_vocab=options.add_embeddings_vocab)
    tags_field.build_vocab(*all_datasets)
    if 'prefixes' in dict_fields:
        pass
    if 'suffixes' in dict_fields:
        pass
    if 'caps' in dict_fields:
        pass
    constants.PAD_ID = dict_fields['words'].vocab.stoi[constants.PAD]
    constants.TAGS_PAD_ID = dict_fields['tags'].vocab.stoi[constants.PAD]


def load_vocabs(path, fields_tuples):
    vocab_path = Path(path, constants.VOCAB)
    vocabs = torch.load(str(vocab_path),
                        map_location=lambda storage, loc: storage)
    vocabs = dict(vocabs)
    for name, field in fields_tuples:
        field.vocab = vocabs[name]
    dict_fields = dict(fields_tuples)
    constants.PAD_ID = dict_fields['words'].vocab.stoi[constants.PAD]
    if 'tags' in dict_fields:
        constants.TAGS_PAD_ID = dict_fields['tags'].vocab.stoi[constants.PAD]


def save_vocabs(path, fields_tuples):
    # list of fields name and their vocab
    vocabs = []
    for name, field in fields_tuples:
        vocabs.append((name, field.vocab))
    # save vectors in a temporary dict
    vectors = {}
    for name, vocab in vocabs:
        vectors[name] = vocab.vectors
        vocab.vectors = None
    vocab_path = Path(path, constants.VOCAB)
    torch.save(vocabs, str(vocab_path))
    # restore vectors - useful if we want to use fields later
    for name, vocab in vocabs:
        vocab.vectors = vectors[name]


class WordsField(Field):
    """Defines a field for word tokens with default
       values from constant.py and with the vocabulary
       defined in vocabulary.py."""

    def __init__(self,
                 unk_token=constants.UNK,
                 pad_token=constants.PAD,
                 init_token=constants.START,
                 eos_token=constants.STOP,
                 batch_first=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.init_token = init_token
        self.eos_token = eos_token
        self.batch_first = batch_first
        self.vocab_cls = Vocabulary


class TagsField(Field):
    """Defines a field for tag tokens by setting unk_token to None
       and pad_token to constants.PAD as default."""

    def __init__(self,
                 unk_token=None,
                 batch_first=True,
                 pad_token=constants.PAD,
                 **kwargs):
        super().__init__(**kwargs)
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.batch_first = batch_first
