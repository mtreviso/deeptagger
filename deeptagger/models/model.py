from abc import ABCMeta, abstractmethod

import logging
import torch
import torch.nn as nn


class Model(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(
        self,
        batch_size=1,
        seed=42,
        device=None,
        loss_weights=None,
        loss_ignore_index=-100,
        **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.is_built = False
        self.device = device
        self.seed = seed
        torch.manual_seed(self.seed)

        self._loss = None
        self._loss_weights = loss_weights
        if self._loss_weights:
            self._loss_weights = torch.FloatTensor(self._loss_weights)
        self._loss_ignore_index = loss_ignore_index

    @property
    def nb_classes(self):
        """."""
        return len(self.tags_field.vocab.stoi)

    @property
    def words_padding_idx(self):
        """."""
        return self.words_field.vocab.stoi[self.words_field.pad_token]

    def loss(self, pred, target):
        """."""
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

    @abstractmethod
    def predict(self, featured_sample):
        pass

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
