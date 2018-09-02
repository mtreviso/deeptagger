import logging


from deeptagger import constants
from deeptagger.corpus import Corpus
from deeptagger.dataset import PoSDataset
from deeptagger.fields import WordsField, TagsField
from deeptagger.iterator import build_iterator
from deeptagger.models import build_model
from deeptagger.optimizer import build_optimizer
from deeptagger.trainer import Trainer
from deeptagger.vectors import AvailableEmbeddings


def run(options):
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
                            rare_with_vectors=options.keep_rare_with_embedding,
                            add_vectors_vocab=options.add_embeddings_vocab)
    tags_field.build_vocab(train_dataset, dev_dataset, test_dataset)

    # ensuring padding ids to their correct values
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
    model = build_model(options, fields)

    logging.info('Building optimizer...')
    optimizer = build_optimizer(options, model.parameters())

    trainer = Trainer(
        model,
        train_iter,
        optimizer,
        options,
        dev_iter=dev_iter,
        test_iter=test_iter,
        final_report=options.final_report
    )
    trainer.train()
