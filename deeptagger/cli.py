import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def configure_output(options):
    if not options.output_dir:
        output_time = time.strftime('%Y-%m-%d_%H:%M:%S')
        output_path = Path('runs', output_time)
        output_path.mkdir(parents=True, exist_ok=True)
        options.output_dir = str(output_path)
    else:
        Path(options.output_dir).mkdir(parents=True, exist_ok=True)


def configure_logger(options):
    logging.Formatter.converter = time.gmtime
    logging.Formatter.default_msec_format = '%s.%03d'
    log_format = '[%(asctime)s] %(levelname)s: %(message)s'
    if logging.getLogger().handlers:
        log_formatter = logging.Formatter(log_format)
        for handler in logging.getLogger().handlers:
            handler.setFormatter(log_formatter)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)

    log_level = logging.DEBUG if options.debug else logging.INFO
    logging.getLogger().setLevel(log_level)
    if options.output_dir is not None:
        fh = logging.FileHandler(os.path.join(options.output_dir, 'out.log'))
        fh.setLevel(log_level)
        logging.getLogger().addHandler(fh)


def configure_seed(options):
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    torch.cuda.manual_seed(options.seed)
