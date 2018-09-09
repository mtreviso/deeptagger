import logging
from pathlib import Path

from deeptagger import constants
from deeptagger import dataset
from deeptagger import fields
from deeptagger import iterator
from deeptagger import models
from deeptagger.predicter import Predicter


def run(options):
    words_field = fields.WordsField()
    tags_field = fields.TagsField()
    fields_tuples = [('words', words_field), ('tags', tags_field)]

    if options.test_path is None and options.text is None:
        raise Exception('You should inform a path to test data or a text.')

    if options.test_path is not None and options.text is not None:
        raise Exception('You cant inform both a path to test data or a text.')

    dataset_iter = None
    if options.test_path is not None:
        logging.info('Building test dataset: {}'.format(options.test_path))
        words_tuple = [('words', words_field)]  # hack since we dont have tags
        test_dataset = dataset.build(options.test_path, words_tuple, options)

        logging.info('Building test iterator...')
        dataset_iter = iterator.build(test_dataset, options.gpu_id,
                                      options.dev_batch_size, is_train=False)

    if options.text is not None:
        logging.info('Preparing text...')
        words_tuple = [('words', words_field)]  # hack since we dont have tags
        test_dataset = dataset.build_texts(options.text, words_tuple,
                                           options)

        logging.info('Building iterator...')
        dataset_iter = iterator.build(test_dataset, options.gpu_id,
                                      options.dev_batch_size, is_train=False)

    logging.info('Loading vocabularies...')
    fields.load_vocabs(options.load, fields_tuples)

    logging.info('Loading model...')
    model = models.load(options.load, fields_tuples)

    predicter = Predicter(dataset_iter, model)
    predictions = predicter.predict(options.prediction_type)

    if options.prediction_type == 'classes':
        predictions = transform_classes_to_tags(tags_field, predictions)

    example_str = save_predicted_probabilities(options.output_dir, predictions)
    if options.text is not None:
        print(options.text)
        print(example_str)


def transform_classes_to_tags(tags_field, predictions):
    tagged_predicitons = []
    for preds in predictions:
        tags_preds = [tags_field.vocab.itos[c] for c in preds]
        tagged_predicitons.append(tags_preds)
    return tagged_predicitons


def save_predicted_probabilities(directory, predictions):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output_path = Path(directory, constants.PREDICTIONS)
    logging.info('Saving predictions to {}'.format(output_path))
    ex_str = '\n'.join([' '.join(map(str, sentence))
                        for sentence in predictions])
    Path(output_path).write_text(ex_str)
    return ex_str
