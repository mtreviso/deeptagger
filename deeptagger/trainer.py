import logging
import time
from pathlib import Path

import torch

from deeptagger import constants
from deeptagger import models
from deeptagger import optimizer
from deeptagger import scheduler
from deeptagger.models.utils import indexes_to_words
from deeptagger.report import report_progress, report_stats, report_stats_final
from deeptagger.stats import Stats


class Trainer:

    def __init__(
        self,
        train_iter,
        model,
        optimizer,
        scheduler_optim,
        options,
        dev_iter=None,
        test_iter=None
    ):
        self.model = model
        self.train_iter = train_iter
        self.dev_iter = dev_iter
        self.test_iter = test_iter
        self.optimizer = optimizer
        self.scheduler_optim = scheduler_optim
        self.epochs = options.epochs
        self.output_dir = options.output_dir
        self.dev_checkpoint_epochs = options.dev_checkpoint_epochs
        self.save_checkpoint_epochs = options.save_checkpoint_epochs
        self.save_best_only = options.save_best_only
        self.early_stopping_patience = options.early_stopping_patience
        self.restore_best_model = options.restore_best_model
        self.current_epoch = 1
        self.final_report = options.final_report

        train_vocab = train_iter.dataset.fields['words'].vocab.orig_stoi
        emb_vocab = train_iter.dataset.fields['words'].vocab.vectors_words
        self.train_stats = Stats(train_vocab=train_vocab,
                                 emb_vocab=emb_vocab,
                                 mask_id=constants.TAGS_PAD_ID)
        self.dev_stats = Stats(train_vocab=train_vocab,
                               emb_vocab=emb_vocab,
                               mask_id=constants.TAGS_PAD_ID)
        self.test_stats = Stats(train_vocab=train_vocab,
                                emb_vocab=emb_vocab,
                                mask_id=constants.TAGS_PAD_ID)
        self.train_stats_history = []
        self.dev_stats_history = []
        self.test_stats_history = []

    def train(self):

        start_time = time.time()
        for epoch in range(self.current_epoch, self.epochs + 1):
            logging.info('Epoch {} of {}'.format(epoch, self.epochs))
            self.current_epoch = epoch

            # Train a single epoch
            logging.info('Training...')
            self.train_epoch()

            # Perform an evaluation on dev set if it is available
            if self.dev_iter is not None:
                # Only perform if a checkpoint was reached
                if (self.dev_checkpoint_epochs > 0
                        and epoch % self.dev_checkpoint_epochs == 0):
                    logging.info('Evaluating...')
                    self.dev_epoch()

            # Perform an evaluation on test set if it is available
            if self.test_iter is not None:
                logging.info('Testing...')
                self.test_epoch()

            # Only save if an improvement occurred
            if self.save_best_only:
                if self.dev_stats.best_acc.epoch == epoch:
                    logging.info('Accuracy improved '
                                 'on epoch {}'.format(epoch))
                    self.save(epoch)
            else:
                # Otherwise, save if a checkpoint was reached
                if (self.save_checkpoint_epochs > 0
                        and epoch % self.save_checkpoint_epochs == 0):
                    self.save(epoch)

            # Stop training before the total number of epochs
            if self.early_stopping_patience > 0:
                # Only stop if the desired patience epochs was reached
                passed_epochs = epoch - self.dev_stats.best_acc.epoch
                if passed_epochs == self.early_stopping_patience:
                    logging.info('Stop training! No improvement on accuracy '
                                 'after {} epochs'.format(passed_epochs))
                    if self.restore_best_model:
                        if self.dev_stats.best_acc.epoch < epoch:
                            self.restore_epoch(self.dev_stats.best_acc.epoch)
                    break

        elapsed = time.time() - start_time
        hms = time.strftime("%Hh:%Mm:%Ss", time.gmtime(elapsed))
        logging.info('Training ended after {}'.format(hms))

        if self.final_report:
            logging.info('Training final report: ')
            report_stats_final(self.train_stats_history)
            if self.dev_iter:
                logging.info('Dev final report: ')
                report_stats_final(self.dev_stats_history)
            if self.test_iter:
                logging.info('Test final report: ')
                report_stats_final(self.test_stats_history)

    def train_epoch(self):
        self.scheduler_optim.step()
        self.model.train()
        self.train_stats.reset()
        indexes = []
        for i, batch in enumerate(self.train_iter, start=1):
            self.model.zero_grad()
            pred = self.model(batch)
            loss = self.model.loss(pred, batch.tags)
            loss.backward()
            self.optimizer.step()
            self.train_stats.update(loss.item(), pred, batch.tags)
            report_progress(i, len(self.train_iter), self.train_stats.loss / i)
            indexes.extend(batch.words)
        inv_vocab = self.train_iter.dataset.fields['words'].vocab.itos
        words = indexes_to_words(indexes, inv_vocab)
        self.train_stats.calc(self.current_epoch, words)
        self.train_stats_history.append(self.train_stats.to_dict())
        report_stats(self.train_stats)

    def dev_epoch(self):
        self._eval(self.dev_iter, self.dev_stats)
        self.dev_stats_history.append(self.dev_stats.to_dict())
        report_stats(self.dev_stats)

    def test_epoch(self):
        self._eval(self.test_iter, self.test_stats)
        self.test_stats_history.append(self.test_stats.to_dict())
        report_stats(self.test_stats)

    def _eval(self, dataset_iter, stats):
        self.model.eval()
        stats.reset()
        indexes = []
        with torch.no_grad():
            for i, batch in enumerate(dataset_iter, start=1):
                pred = self.model(batch)
                loss = self.model.loss(pred, batch.tags)
                stats.update(loss.item(), pred, batch.tags)
                report_progress(i, len(dataset_iter), stats.loss / i)
                indexes.extend(batch.words)
        inv_vocab = dataset_iter.dataset.fields['words'].vocab.itos
        words = indexes_to_words(indexes, inv_vocab)
        stats.calc(self.current_epoch, words)

    def save(self, current_epoch):
        epoch_dir = 'epoch_{}'.format(current_epoch)
        output_path = Path(self.output_dir, epoch_dir)
        output_path.mkdir(exist_ok=True)
        logging.info('Saving training state to {}'.format(output_path))
        models.save(output_path, self.model)
        optimizer.save(output_path, self.optimizer)
        scheduler.save(output_path, self.scheduler_optim)

    def load(self, directory):
        logging.info('Loading training state from {}'.format(directory))
        models.load_state(directory, self.model)
        optimizer.load_state(directory, self.optimizer)
        scheduler.load_state(directory, self.scheduler_optim)

    def restore_epoch(self, epoch):
        epoch_dir = 'epoch_{}'.format(epoch)
        self.load(str(Path(self.output_dir, epoch_dir)))

    def resume(self, epoch):
        self.restore_epoch(epoch)
        self.current_epoch = epoch
        self.train()
