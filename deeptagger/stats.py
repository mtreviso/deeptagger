from deeptagger.models.utils import unroll
import torch
import numpy as np


class Stats(object):

    def __init__(self, train_vocabulary=None, mask_id=-1):
        self.train_vocabulary = train_vocabulary
        self.mask_id = mask_id
        self.loss = 0
        self.acc = 0
        self.acc_oov = 0
        self.acc_emb = 0
        self.pred_classes = []
        self.pred_probs = []
        self.golds = []

    def reset(self):
        self.loss = 0
        self.pred_classes.clear()
        self.pred_probs.clear()
        self.golds.clear()

    @staticmethod
    def unmask(tensor, mask):
        lengths = mask.int().sum(dim=-1).tolist()
        return [x[: lengths[i]].tolist() for i, x in enumerate(tensor)]

    @property
    def nb_of_batches(self):
        return len(self.golds)

    def add(self, loss, preds, golds, words=None):
        mask = golds != self.mask_id
        pred_probs = torch.exp(preds)
        pred_classes = pred_probs.argmax(dim=-1)
        self.loss += loss
        self.pred_probs.append(unroll(self.unmask(pred_probs, mask)))
        self.pred_classes.append(unroll(self.unmask(pred_classes, mask)))
        self.golds.append(unroll(self.unmask(golds, mask)))

    def accuracy(self):
        flattened_preds = np.array(unroll(self.pred_classes))
        flattened_golds = np.array(unroll(self.golds))
        self.acc = (flattened_preds == flattened_golds).mean()
        return self.acc

    def accuracy_oov(self, train_vocabulary):
        return self.acc_oov

    def accuracy_emb(self, emb_vocabulary):
        return self.acc_emb

    def final_loss(self):
        return self.loss / self.nb_of_batches

    def get(self, metric):
        if metric == 'loss':
            return self.final_loss()
        elif metric == 'acc':
            return self.accuracy()
        else:
            raise Exception('Metric not available.')
