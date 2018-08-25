import argparse
import logging
import torch

from deeptagger import opts
from deeptagger import cli
from deeptagger import constants
from deeptagger.iterator import build_iterator
from deeptagger.corpus import Corpus
from deeptagger.fields import WordsField, TagsField
from deeptagger.dataset import PoSDataset
from deeptagger.vectors import AvailableEmbeddings
from deeptagger.models.rcnn import RCNN
from deeptagger.models.simple_lstm import SimpleLSTM
from deeptagger.trainer import Trainer

parser = argparse.ArgumentParser(description='DeepTagger')
opts.general_opts(parser)
opts.preprocess_opts(parser)
opts.model_opts(parser)
opts.train_opts(parser)


def main(options):
    words_field = WordsField()
    tags_field = TagsField()
    fields = [('words', words_field), ('tags', tags_field)]

    def filter_len(x):
        return options.min_length <= len(x.words) <= options.max_length

    logging.info('Building training corpus: {}'.format(options.train_path))
    train_corpus = Corpus(fields, options.del_word, options.del_tag)
    train_corpus.read(options.train_path)
    train_dataset = PoSDataset(train_corpus, filter_pred=filter_len)

    logging.info('Building dev dataset: {}'.format(options.dev_path))
    dev_corpus = Corpus(fields, options.del_word, options.del_tag)
    dev_corpus.read(options.dev_path)
    dev_dataset = PoSDataset(dev_corpus, filter_pred=filter_len)

    logging.info('Building test dataset: {}'.format(options.test_path))
    test_corpus = Corpus(fields, options.del_word, options.del_tag)
    test_corpus.read(options.test_path)
    test_dataset = PoSDataset(test_corpus, filter_pred=filter_len)

    logging.info('Loading {} word embeddings from: {}'.format(
        options.embeddings_format,
        options.embeddings_path
    ))
    word_emb_cls = AvailableEmbeddings[options.embeddings_format]
    vectors = word_emb_cls(options.embeddings_path, binary=False)

    logging.info('Building vocabulary using only train data...')
    words_field.build_vocab(train_dataset,
                            vectors=vectors,
                            min_freq=options.vocab_min_frequency,
                            rare_with_vectors=options.keep_rare_with_vectors)
    tags_field.build_vocab(train_dataset, dev_dataset, test_dataset)

    # set padding ids to their correct values
    constants.PAD_ID = words_field.vocab.stoi[constants.PAD]
    constants.TAGS_PAD_ID = tags_field.vocab.stoi[constants.PAD]

    logging.info('Words vocab size: {}'.format(len(words_field.vocab.stoi)))
    logging.info('Tags vocab size: {}'.format(len(tags_field.vocab.stoi)))

    logging.info('Building iterators...')

    train_iter = build_iterator(
        train_dataset, options.gpu_id, options.train_batch_size, is_train=True)
    dev_iter = build_iterator(
        dev_dataset, options.gpu_id, options.dev_batch_size, is_train=False)
    test_iter = build_iterator(test_dataset, options.gpu_id,
                               options.dev_batch_size, is_train=False)

    logging.info('Building model...')
    model = RCNN(
        words_field,
        tags_field,
        prefixes_field=None,
        suffixes_field=None,
        caps_field=None,
        seed=options.seed,
        device=options.gpu_id
    )

    model.build(
        word_embeddings_size=100,
        prefix_embeddings_size=20,
        suffix_embeddings_size=20,
        caps_embeddings_size=6,
        hidden_size=100,
        dropout=0.5,
        emb_dropout=0.0,
        freeze_embeddings=False,
    )

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters)  # lr=options.learning_rate

    trainer = Trainer(
        model,
        train_iter,
        optimizer,
        options,
        dev_iter=dev_iter,
        test_iter=test_iter
    )
    trainer.train()


if __name__ == '__main__':
    options = parser.parse_args()
    cli.configure_output(options)
    cli.configure_logger(options)
    cli.configure_seed(options)

    from pprint import pformat

    logging.info('Running with options:\n{}'.format(pformat(vars(options))))
    logging.info('Local output directory is: {}'.format(options.output_dir))

    main(options)
