from torchtext import data
from deeptagger.constants import UNK, PAD, START, STOP
from deeptagger.vocabulary import Vocabulary


class WordsField(data.Field):
    """Defines a field for word tokens with default
       values from constant.py and with the vocabulary
       defined in vocabulary.py."""

    def __init__(self, unk_token=UNK, pad_token=PAD, init_token=START,
                 eos_token=STOP, batch_first=True, **kwargs):
        super().__init__(**kwargs)
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.init_token = init_token
        self.eos_token = eos_token
        self.batch_first = batch_first
        self.vocab_cls = Vocabulary


class TagsField(data.Field):
    """Defines a field for tag tokens by setting unk_token to None
       and pad_token to constants.PAD as default."""

    def __init__(self, unk_token=None, batch_first=True,
                 pad_token=PAD, **kwargs):
        super().__init__(**kwargs)
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.batch_first = batch_first
