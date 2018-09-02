import torch.optim


def build_optimizer(options, model_parameters):
    kwargs = {}
    OptimizerClass = None
    if options.optimizer == 'adam':
        OptimizerClass = torch.optim.Adam
    elif options.optimizer == 'adadelta':
        OptimizerClass = torch.optim.Adadelta
    elif options.optimizer == 'adagrad':
        OptimizerClass = torch.optim.Adagrad
    elif options.optimizer == 'adamax':
        OptimizerClass = torch.optim.Adamax
    elif options.optimizer == 'sparseadam':
        OptimizerClass = torch.optim.SparseAdam
    elif options.optimizer == 'sgd':
        OptimizerClass = torch.optim.SGD
    elif options.optimizer == 'asgd':
        OptimizerClass = torch.optim.ASGD
    elif options.optimizer == 'rmsprop':
        OptimizerClass = torch.optim.RMSprop

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
    return OptimizerClass(parameters, **kwargs)
