from argparse import Namespace
from pathlib import Path

from deeptagger import cli
from deeptagger import constants
from deeptagger.dataset import dataset, fields
from deeptagger import features
from deeptagger import iterator
from deeptagger import models
from deeptagger import optimizer
from deeptagger import opts
from deeptagger import train
from deeptagger.predict import transform_classes_to_tags
from deeptagger.predicter import Predicter


class Tagger:

    def __init__(self, gpu_id=None):
        self.words_field = fields.WordsField()
        self.tags_field = fields.TagsField()
        self.fields_tuples = [('words', self.words_field),
                              ('tags', self.tags_field)]
        self.options = None
        self.model = None
        self.optim = None
        self.gpu_id = gpu_id
        self._loaded = False

    def load(self, dir_path):
        self.options = opts.load(dir_path)
        self.fields_tuples += features.load(dir_path)
        fields.load_vocabs(dir_path, self.fields_tuples)
        self.options.gpu_id = self.gpu_id
        self.model = models.load(dir_path, self.fields_tuples)
        if Path(dir_path, constants.OPTIMIZER).exists():
            self.optim = optimizer.load(dir_path, self.model.parameters())
        self._loaded = True

    def predict(self, texts, batch_size=32, prediction_type='classes'):
        if not self._loaded:
            raise Exception('You should load a saved model first.')

        f_tuples = list(filter(lambda x: x[0] != 'tags', self.fields_tuples))
        text_dataset = dataset.build_texts(texts, f_tuples, self.options)
        dataset_iter = iterator.build(text_dataset,
                                      self.options.gpu_id,
                                      batch_size,
                                      is_train=False)
        predicter = Predicter(dataset_iter, self.model)
        predictions = predicter.predict(prediction_type)
        if prediction_type == 'classes':
            predictions = transform_classes_to_tags(self.tags_field,
                                                    predictions)
        if isinstance(texts, str):  # special case for a str input
            return predictions[0]
        return predictions

    def predict_classes(self, texts, batch_size=32):
        return self.predict(texts,
                            batch_size=batch_size,
                            prediction_type='classes')

    def predict_probas(self, texts, batch_size=32):
        return self.predict(texts,
                            batch_size=batch_size,
                            prediction_type='probas')

    def train(self, args):
        if self._loaded:
            options = vars(self.options)
        else:
            options = opts.get_default_args()
            options.gpu_id = self.gpu_id
        options.update(args)
        options = Namespace(**options)
        options.output_dir = cli.configure_output(options.output_dir)
        cli.configure_logger(options.debug, options.output_dir)
        cli.configure_seed(options.seed)
        cli.configure_device(options.gpu_id)
        self.options = options
        result = train.run(self.options)
        self.options = result[0]
        self.fields_tuples = result[1]
        self.model = result[2]
        self.optim = result[3]
        self._loaded = True

    def save(self, dir_path):
        opts.save(dir_path, self.options)
        fields.save_vocabs(dir_path, self.fields_tuples)
        models.save(dir_path, self.model)
        optimizer.save(dir_path, self.optim)
