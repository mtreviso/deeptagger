import numpy as np
import torch

from deeptagger import constants
from deeptagger.models.utils import unroll, unmask


class BestValueEpoch:
    __slots__ = ['value', 'epoch']

    def __init__(self, value, epoch):
        self.value = value
        self.epoch = epoch


class Stats(object):

    def __init__(self,
                 train_vocab=None,
                 emb_vocab=None,
                 mask_id=constants.PAD_ID):
        """
        :param train_vocab: a set object with words found in training data
        :param emb_vocab: a set object with words found in embeddings data
        :param mask_id: constant used for masking tags
        """
        self.train_vocab = train_vocab
        self.emb_vocab = emb_vocab
        self.mask_id = mask_id

        # this attrs will be updated every time a new prediciton is added
        self.pred_classes = []
        self.pred_probs = []
        self.golds = []

        # this attrs will be set when get_ methods are called
        self.loss = 0
        self.acc = None
        self.acc_oov = None
        self.acc_emb = None
        self.acc_sent = None
        self.best_acc = BestValueEpoch(value=0, epoch=0)
        self.best_acc_oov = BestValueEpoch(value=0, epoch=0)
        self.best_acc_emb = BestValueEpoch(value=0, epoch=0)
        self.best_acc_sent = BestValueEpoch(value=0, epoch=0)
        self.best_loss = BestValueEpoch(value=np.Inf, epoch=0)

        # private (used for lazy calculation)
        self._flattened_preds = None
        self._flattened_golds = None

    def reset(self):
        self.pred_classes.clear()
        self.pred_probs.clear()
        self.golds.clear()
        self.loss = 0
        self.acc = None
        self.acc_oov = None
        self.acc_emb = None
        self.acc_sent = None
        self.best_acc = BestValueEpoch(value=0, epoch=0)
        self.best_acc_oov = BestValueEpoch(value=0, epoch=0)
        self.best_acc_emb = BestValueEpoch(value=0, epoch=0)
        self.best_acc_sent = BestValueEpoch(value=0, epoch=0)
        self.best_loss = BestValueEpoch(value=np.Inf, epoch=0)
        self._flattened_preds = None
        self._flattened_golds = None

    @property
    def nb_batches(self):
        return len(self.golds)

    def add(self, loss, preds, golds):
        mask = golds != self.mask_id
        pred_probs = torch.exp(preds)
        pred_classes = pred_probs.argmax(dim=-1)
        self.loss += loss
        self.pred_probs.append(unroll(unmask(pred_probs, mask)))
        self.pred_classes.append(unroll(unmask(pred_classes, mask)))
        self.golds.append(unroll(unmask(golds, mask)))

    def get_loss(self):
        return self.loss / self.nb_batches

    def _get_bins(self):
        if self._flattened_preds is None:
            self._flattened_preds = np.array(unroll(self.pred_classes))
        if self._flattened_golds is None:
            self._flattened_golds = np.array(unroll(self.golds))
        return self._flattened_preds == self._flattened_golds

    def get_acc(self):
        if self.acc is None:
            bins = self._get_bins()
            self.acc = bins.mean()
        return self.acc

    def get_acc_oov(self, words=None):
        if self.acc_oov is None:
            idx = [i for i, w in enumerate(unroll(words))
                   if w not in self.train_vocab or w == constants.UNK]
            if len(idx) == 0:
                self.acc_oov = np.Inf
                return self.acc_oov
            bins = self._get_bins()
            self.acc_oov = bins[idx].mean()
        return self.acc_oov

    def get_acc_emb(self, words=None):
        if self.acc_emb is None:
            idx = [i for i, w in enumerate(unroll(words))
                   if w not in self.emb_vocab and w != constants.UNK]
            if len(idx) == 0:
                self.acc_emb = np.Inf
                return self.acc_emb
            bins = self._get_bins()
            self.acc_emb = bins[idx].mean()
        return self.acc_emb

    def get_acc_sentence(self):
        if self.acc_sent is None:
            bins = [x == y for x, y in zip(self.pred_classes, self.golds)]
            self.acc_sent = np.mean(bins)
        return self.acc_sent

    def calc(self, current_epoch, words):
        specials = [constants.PAD, constants.START, constants.STOP]
        words = list(filter(lambda w: w not in specials, unroll(words)))
        current_loss = self.get_loss()
        current_acc = self.get_acc()
        current_acc_oov = self.get_acc_oov(words)
        current_acc_emb = self.get_acc_emb(words)
        current_acc_sent = self.get_acc_sentence()

        if current_loss < self.best_loss.value:
            self.best_loss.value = current_loss
            self.best_loss.epoch = current_epoch

        if current_acc > self.best_acc.value:
            self.best_acc.value = current_acc
            self.best_acc.epoch = current_epoch

        if current_acc_oov > self.best_acc_oov.value:
            self.best_acc_oov.value = current_acc_oov
            self.best_acc_oov.epoch = current_epoch

        if current_acc_emb > self.best_acc_emb.value:
            self.best_acc_emb.value = current_acc_emb
            self.best_acc_emb.epoch = current_epoch

        if current_acc_sent > self.best_acc_sent.value:
            self.best_acc_sent.value = current_acc_sent
            self.best_acc_sent.epoch = current_epoch

    def to_dict(self):
        return {
            'loss': self.get_loss(),
            'acc': self.get_acc(),
            'acc_oov': self.get_acc_oov(),
            'acc_emb': self.get_acc_emb(),
            'acc_sent': self.get_acc_sentence(),
            'best_loss': self.best_loss,
            'best_acc': self.best_acc,
            'best_acc_oov': self.best_acc_oov,
            'best_acc_emb': self.best_acc_emb,
            'best_acc_sent': self.best_acc_sent
        }
