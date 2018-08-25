# lowercased special tokens
UNK = '<unk>'
PAD = '<pad>'
START = '<bos>'
STOP = '<eos>'

# special tokens id (don't edit this order)
UNK_ID = 0
PAD_ID = 1
START_ID = 2
STOP_ID = 3

# this should be set later
TAGS_PAD_ID = 0

# output_dir
OUTPUT_DIR = 'runs'

# default filenames
DATASET = 'dataset.torch'
OPTIMIZER = 'optim.torch'
MODEL = 'model.torch'
TRAINER = 'trainer.torch'


def set_tags_pad_id(pad_id):
    global TAGS_PAD_ID
    TAGS_PAD_ID = pad_id
