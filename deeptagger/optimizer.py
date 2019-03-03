from pathlib import Path

import torch
import adabound

from deeptagger import constants
from deeptagger import opts
from deeptagger.modules.optim.adamw import AdamW
from deeptagger.modules.optim.step_decay_optimizer import StepDecayOptimizer
from deeptagger.modules.optim.lr_scheduler import (NoamDecayScheduler,
                                                   ExpDecayScheduler,
                                                   RsqrtDecayScheduler)

available_optimizers = {
    'adam': torch.optim.Adam,
    'adadelta': torch.optim.Adadelta,
    'adagrad': torch.optim.Adagrad,
    'adamax': torch.optim.Adamax,
    'sparseadam': torch.optim.SparseAdam,
    'sgd': torch.optim.SGD,
    'asgd': torch.optim.ASGD,
    'rmsprop': torch.optim.RMSprop,
    'adabound': adabound.AdaBound,
    'adamw': AdamW,
}

available_step_decays = {
    'noam': NoamDecayScheduler,
    'exp': ExpDecayScheduler,
    'rsqrt': RsqrtDecayScheduler
}


def build_step_decay_wrapper(options, optim):
    kwargs = {}
    lr_step_decay_class = available_step_decays[options.lr_step_decay]

    if options.warmup_steps is not None:
        kwargs['warmup_steps'] = options.warmup_steps
    if options.decay_steps is not None:
        kwargs['decay_steps'] = options.decay_steps
    if options.lr_step_decay == 'noam':
        kwargs['model_size'] = max(options.hidden_size)
    elif options.lr_step_decay == 'exp':
        kwargs['initial_lr'] = options.learning_rate

    lr_step_decay = lr_step_decay_class(**kwargs)
    wrapped_optim = StepDecayOptimizer(optim, lr_step_decay)
    return wrapped_optim


def build(options, model_parameters):
    kwargs = {}
    optim_class = available_optimizers[options.optimizer]

    if options.learning_rate is not None:
        kwargs['lr'] = options.learning_rate
    if options.weight_decay is not None:
        kwargs['weight_decay'] = options.weight_decay
    if options.lr_decay is not None:
        kwargs['lr_decay'] = options.lr_decay
    if options.rho is not None:
        kwargs['rho'] = options.rho
    if options.beta0 is not None and options.beta1 is not None:
        kwargs['betas'] = (options.beta0, options.beta1)
    if options.momentum:
        kwargs['momentum'] = options.momentum
    if options.nesterov:
        kwargs['nesterov'] = options.nesterov
    if options.alpha:
        kwargs['alpha'] = options.alpha

    # learning rate is a required arg for sgd
    if options.optimizer == 'sgd' and options.learning_rate is None:
        kwargs['lr'] = 0.1

    parameters = filter(lambda p: p.requires_grad, model_parameters)
    optim = optim_class(parameters, **kwargs)

    # wrap optimizer inside the step decay optimizer
    if options.lr_step_decay is not None:
        optim = build_step_decay_wrapper(options, optim)

    return optim


def load_state(path, optim):
    optim_path = Path(path, constants.OPTIMIZER)
    optim.load_state_dict(torch.load(str(optim_path)))


def load(path, model_parameters):
    options = opts.load(path)
    optim = build(options, model_parameters)
    load_state(path, optim)
    return optim


def save(path, optim):
    optim_path = Path(path, constants.OPTIMIZER)
    torch.save(optim.state_dict(), str(optim_path))
