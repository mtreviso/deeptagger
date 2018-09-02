import logging
from abc import ABCMeta, abstractmethod

import torch


class Model(torch.nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self,
                 words_field,
                 tags_field,
                 prefixes_field=None,
                 suffixes_field=None,
                 caps_field=None):
        super().__init__()
        # Default fields and embeddings
        self.words_field = words_field
        self.tags_field = tags_field
        self.word_embeddings = words_field.vocab.vectors
        # Extra features
        self.prefixes_field = prefixes_field
        self.suffixes_field = suffixes_field
        self.caps_field = caps_field
        # Building flag
        self.is_built = False
        # Loss function has to be defined in build()
        self._loss = None

    @property
    def nb_classes(self):
        return len(self.tags_field.vocab.stoi)

    @property
    def words_padding_idx(self):
        return self.words_field.vocab.stoi[self.words_field.pad_token]

    @property
    def loss_ignore_index(self):
        return self.tags_field.vocab.stoi[self.tags_field.pad_token]

    def loss(self, pred, target):
        # (bs*ts, nb_classes)
        predicted = pred.reshape(-1, self.nb_classes)

        # Ensure y is a tensor
        y = torch.tensor(target)

        # (bs*ts, )
        y = y.reshape(-1)

        return self._loss(predicted, y)

    @abstractmethod
    def build(self, **params):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def predict_proba(self, batch):
        pred = self.forward(batch)
        return torch.exp(pred)  # assume log softmax in the output

    def extract_features(self, dataset):
        """
        :param dataset: torchtext.Dataset object
        """
        raise NotImplementedError

    def load(self, path):
        logging.debug("Loading model weights from {}".format(path))
        self.load_state_dict(
            torch.load(str(path), map_location=lambda storage, loc: storage)
        )

    def save(self, path):
        logging.debug("Saving model weights to {}".format(path))
        torch.save(self.state_dict(), str(path))
