import logging
from pathlib import Path

import torch
from deeptagger import constants
from deeptagger.stats import Stats


def report(stats, best_acc, best_epoch, epoch):
    acc = stats.accuracy()
    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
    logging.info('Loss: {:.4f}'.format(stats.final_loss()))
    logging.info('Accuracy: {:.4f}'.format(acc))
    logging.info('Best on epoch {}: {:.4f}'.format(best_epoch, best_acc))
    del stats
    return best_acc, best_epoch


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
        self.save_checkpoint_epochs = options.save_checkpoint_epochs

    def train(self):
        train_best_acc, train_best_epoch = 0, 0
        dev_best_acc, dev_best_epoch = 0, 0
        test_best_acc, test_best_epoch = 0, 0
        for epoch in range(1, self.epochs + 1):
            logging.info('Epoch {} of {}'.format(epoch, self.epochs))

            logging.info('Training...')
            stats = self.train_epoch()
            train_best_acc, train_best_epoch = report(
                stats, train_best_acc, train_best_epoch, epoch)

            if self.dev_iter is not None:
                logging.info('Evaluating...')
                stats = self.eval(self.dev_iter)
                dev_best_acc, dev_best_epoch = report(
                    stats, dev_best_acc, dev_best_epoch, epoch)

            if self.test_iter is not None:
                logging.info('Testing...')
                stats = self.eval(self.test_iter)
                test_best_acc, test_best_epoch = report(
                    stats, test_best_acc, test_best_epoch, epoch)

            if self.save_checkpoint_epochs == 0:
                continue
            if (
                epoch % self.save_checkpoint_epochs == 0 or
                dev_best_epoch == epoch
            ):
                self.save(epoch)

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
            j, n, l = i + 1, len(self.train_iter), stats.loss / (i + 1)
            print('Loss ({}/{}): {:.4f}'.format(j, n, l), end='\r')
        return stats

    def eval(self, dataset_iter):
        stats = Stats(mask_id=constants.TAGS_PAD_ID)
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataset_iter):
                pred = self.model(batch)
                loss = self.model.loss(pred, batch)
                stats.add(loss.item(), pred, batch.tags)
                j, n, l = i + 1, len(dataset_iter), stats.loss / (i + 1)
                print('Loss ({}/{}): {:.4f}'.format(j, n, l), end='\r')
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
