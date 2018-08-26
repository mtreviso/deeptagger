import logging
from pathlib import Path

import torch
from deeptagger import constants
from deeptagger.stats import Stats


class Trainer:

    def __init__(
        self,
        model,
        train_iter,
        optimizer,
        options,
        dev_iter=None,
        test_iter=None
    ):
        self.model = model
        self.train_iter = train_iter
        self.dev_iter = dev_iter
        self.test_iter = test_iter
        self.optimizer = optimizer
        self.epochs = options.epochs
        self.output_dir = options.output_dir
        self.dev_checkpoint_epochs = options.dev_checkpoint_epochs
        self.save_checkpoint_epochs = options.save_checkpoint_epochs
        self.save_best_only = options.save_best_only
        self.early_stopping_patience = options.early_stopping_patience
        self.restore_best_model = options.restore_best_model
        self.emb_vocab = train_iter.dataset.fields['words'].vocab.vectors_words
        self.train_vocab = train_iter.dataset.fields['words'].vocab.orig_stoi

    def train(self):
        train_best_acc, train_best_epoch = 0, 0
        dev_best_acc, dev_best_epoch = 0, 0
        test_best_acc, test_best_epoch = 0, 0

        for epoch in range(1, self.epochs + 1):
            logging.info('Epoch {} of {}'.format(epoch, self.epochs))
            logging.info('Training...')

            # Train a single epoch
            stats = self.train_epoch()
            train_best_acc, train_best_epoch = self.report(
                stats, train_best_acc, train_best_epoch, epoch
            )
            del stats

            # Perform an evaluation on dev set if it is available
            if self.dev_iter is not None:
                # Only perform if a checkpoint was reached
                if (self.dev_checkpoint_epochs > 0
                        and epoch % self.dev_checkpoint_epochs == 0):
                    logging.info('Evaluating...')
                    stats = self.eval(self.dev_iter)
                    dev_best_acc, dev_best_epoch = self.report(
                        stats, dev_best_acc, dev_best_epoch, epoch
                    )
                    del stats

            # Perform an evaluation on test set if it is available
            if self.test_iter is not None:
                logging.info('Testing...')
                stats = self.eval(self.test_iter)
                test_best_acc, test_best_epoch = self.report(
                    stats, test_best_acc, test_best_epoch, epoch
                )
                del stats

            # Only save if an improvement occurred
            if self.save_best_only:
                if dev_best_epoch == epoch:
                    logging.info('Accuracy improved '
                                 'on epoch {}.'.format(epoch))
                    self.save(epoch)
            else:
                # Otherwise, save if a checkpoint was reached
                if (self.save_checkpoint_epochs > 0
                        and epoch % self.save_checkpoint_epochs == 0):
                    self.save(epoch)

            # Stop training before the total number of epochs
            if self.early_stopping_patience > 0:
                # Only stop if the desired patience epochs was reached
                if epoch - dev_best_epoch == self.early_stopping_patience:
                    logging.info('Stop training. No improvement on acc after '
                                 '{} epochs.'.format(epoch - dev_best_epoch))
                    if self.restore_best_model:
                        self.restore_epoch(dev_best_epoch)
                    break

        logging.info('Best accuracies: ')
        logging.info('Train acc on epoch {}: {:.4f}'.format(train_best_epoch,
                                                            train_best_acc))
        if self.dev_iter:
            logging.info('Dev acc on epoch {}: {:.4f}'.format(dev_best_epoch,
                                                              dev_best_acc))
        if self.test_iter:
            logging.info('Test acc on epoch {}: {:.4f}'.format(test_best_epoch,
                                                               test_best_acc))

    def train_epoch(self):
        stats = Stats(mask_id=constants.TAGS_PAD_ID)
        self.model.train()
        for i, batch in enumerate(self.train_iter):
            self.model.zero_grad()
            pred = self.model(batch)
            loss = self.model.loss(pred, batch)
            loss.backward()
            self.optimizer.step()
            stats.add(loss.item(), pred, batch.tags)
            n = len(self.train_iter)
            l_val = stats.loss / (i + 1)
            print('Loss ({}/{}): {:.4f}'.format(i + 1, n, l_val), end='\r')
        return stats

    def eval(self, dataset_iter):
        stats = Stats(mask_id=constants.TAGS_PAD_ID)
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataset_iter):
                pred = self.model(batch)
                loss = self.model.loss(pred, batch)
                stats.add(loss.item(), pred, batch.tags, words=batch.words)
                n = len(dataset_iter)
                l_val = stats.loss / (i + 1)
                print('Loss ({}/{}): {:.4f}'.format(i + 1, n, l_val), end='\r')
        return stats

    def save(self, current_epoch):
        output_path = Path(self.output_dir, 'epoch_{}'.format(current_epoch))
        output_path.mkdir(exist_ok=True)

        model_path = str(output_path / constants.MODEL)
        logging.info('Saving training state to {}'.format(output_path))
        self.model.save(model_path)

        optimizer_path = str(output_path / constants.OPTIMIZER)
        torch.save(self.optimizer.state_dict(), optimizer_path)

    def load(self, directory):
        logging.info('Loading training state from {}'.format(directory))
        root_path = Path(directory)

        model_path = root_path / constants.MODEL
        self.model.load(str(model_path))

        optimizer_path = root_path / constants.OPTIMIZER
        self.optimizer.load_state_dict(torch.load(str(optimizer_path)))

    def restore_epoch(self, epoch):
        epoch_dir = 'epoch_{}'.format(epoch)
        self.load(str(Path(self.output_dir, epoch_dir)))

    def report(self, stats, best_acc, best_epoch, epoch):
        acc = stats.accuracy()
        loss = stats.final_loss()
        acc_oov = stats.accuracy_oov(self.train_vocab)
        acc_emb = stats.accuracy_emb(self.emb_vocab)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
        logging.info('Loss: {:.4f}'.format(loss))
        logging.info('Accuracy: {:.4f} - '
                     'OOV: {:.4f} - '
                     'Emb: {:.4f}'.format(acc, acc_oov, acc_emb))
        logging.info('Best on epoch {}: {:.4f}'.format(best_epoch, best_acc))
        return best_acc, best_epoch
