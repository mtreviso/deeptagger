# from torch.optim import Optimizer


class StepDecayOptimizer:
    """Simple wrapper that implements learning rate decay during
    optimization steps.

    Args:
        optimizer (nn.optim): torch optimizer object
        lr (float): initial learning rate
        lr_scheduler_fn (optional): a callable that receives the current step
            as argument and returns a factor that will scale the lr
    """
    def __init__(self, optimizer, lr, lr_scheduler_fn=None):
        self.optimizer = optimizer
        self._lr = lr
        self._lr_scheduler_fn = lr_scheduler_fn
        self._step = 0

    def step(self):
        self._step += 1
        current_lr = self.learning_rate()
        for p in self.optimizer.param_groups:
            p['lr'] = current_lr
        self.optimizer.step()

    def learning_rate(self):
        if self._lr_scheduler_fn is None:
            return self._lr
        return self._lr * self._lr_scheduler_fn(self._step)

    def state_dict(self):
        return {
            'training_step': self._step,
            'optimizer': self.optimizer.state_dict()
        }

    def load_state_dict(self, state_dict):
        self._step = state_dict['training_step']
        self.optimizer.load_state_dict(state_dict['optimizer'])


if __name__ == '__main__':

    from deeptagger.modules.optim.lr_scheduler import (NoamDecayScheduler,
                                                       RsqrtDecayScheduler,
                                                       ExpDecayScheduler)
    import numpy as np
    from matplotlib import pyplot as plt

    # usage:
    # model = MyModel()
    # opt = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    # noam_scheduler = NoamDecayScheduler(4000, model_hidden_size)
    # step_optim = StepDecayOptimizer(opt, lr=2.0, noam_scheduler)

    lr = 1.0
    opts = [StepDecayOptimizer(None, lr, NoamDecayScheduler(4000, 512)),
            StepDecayOptimizer(None, lr, NoamDecayScheduler(8000, 512)),
            StepDecayOptimizer(None, lr, NoamDecayScheduler(4000, 256)),
            StepDecayOptimizer(None, lr, RsqrtDecayScheduler(1000)),
            StepDecayOptimizer(None, lr, ExpDecayScheduler(0.1, 8000))
            ]

    epoch_steps = 20000  # nb of steps for one epoch
    rates = []
    x = np.arange(1, epoch_steps)
    for _ in x:
        rs = []
        for opt in opts:
            opt._step += 1
            rs.append(opt.learning_rate())
        rates.append(rs)

    plt.semilogy(x, rates)
    plt.legend(["noam:512:4000",
                "noam:512:8000",
                "noam:256:4000",
                "rsqrt:1000",
                "exp:0.9:100"])
    plt.show()
