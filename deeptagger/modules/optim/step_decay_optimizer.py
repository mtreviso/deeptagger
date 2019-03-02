import math


class StepOptimizerLRScheduler:
    def __call__(self, step):
        raise NotImplementedError


class NoamDecayScheduler(StepOptimizerLRScheduler):
    """Implements learning rate decay from AIAYN paper."""

    def __init__(self, warmup_steps, model_size):
        self.warmup_steps = warmup_steps
        self.model_size = model_size

    def __call__(self, step):
        sqrt_model_size = math.pow(self.model_size, -0.5)
        sqrt_warmup_steps = math.pow(self.warmup_steps, -1.5)
        sqrt_step = math.pow(step, -0.5)
        return sqrt_model_size * min(sqrt_step, step * sqrt_warmup_steps)


class ExpDecayScheduler(StepOptimizerLRScheduler):
    """Adapted from opennmt-py"""

    def __init__(self, initial_lr, decay_steps, start_step=0):
        self.initial_lr = initial_lr
        self.decay_steps = decay_steps
        self.start_step = start_step

    def __call__(self, step):
        valid_steps = max(0, step - self.start_step + self.decay_steps)
        return math.pow(self.initial_lr, valid_steps // self.decay_steps)


class RsqrtDecayScheduler(StepOptimizerLRScheduler):
    """Adapted from opennmt-py"""

    def __init__(self, warmup_steps):
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        return 1.0 / math.sqrt(max(step, self.warmup_steps))


class StepOptimizerWrapper:
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

    import numpy as np
    from matplotlib import pyplot as plt

    # usage:
    # model = MyModel()
    # opt = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    # noam_scheduler = NoamDecayScheduler(4000, model_hidden_size)
    # step_optim = StepOptimizerWrapper(opt, lr=2.0, noam_scheduler)

    lr = 1.0
    opts = [StepOptimizerWrapper(None, lr, NoamDecayScheduler(4000, 512)),
            StepOptimizerWrapper(None, lr, NoamDecayScheduler(8000, 512)),
            StepOptimizerWrapper(None, lr, NoamDecayScheduler(4000, 256)),
            StepOptimizerWrapper(None, lr, RsqrtDecayScheduler(1000)),
            StepOptimizerWrapper(None, lr, ExpDecayScheduler(0.1, 8000))
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
