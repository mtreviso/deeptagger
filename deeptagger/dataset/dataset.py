from deeptagger.dataset.corpus import Corpus
from deeptagger.dataset.modules.dataset import LazyDataset


def build(path, fields_tuples, options):
    def filter_len(x):
        return options.min_length <= len(x.words) <= options.max_length
    corpus = Corpus(fields_tuples, options.del_word, options.del_tag)
    corpus.read(path)

    if (options.use_prefixes or
            options.use_suffixes or
            options.use_caps):
        corpus.add_features(options.prefix_min_length,
                            options.prefix_max_length,
                            options.suffix_min_length,
                            options.suffix_max_length)
    return PoSDataset(corpus, filter_pred=filter_len)


def build_texts(texts, fields_tuples, options):
    def filter_len(x):
        return options.min_length <= len(x.words) <= options.max_length
    corpus = Corpus(fields_tuples, options.punctuations)
    corpus.add_texts(texts)
    if (options.use_prefixes or
            options.use_suffixes or
            options.use_caps):
        corpus.add_features(options.prefix_min_length,
                            options.prefix_max_length,
                            options.suffix_min_length,
                            options.suffix_max_length)
    return PoSDataset(corpus, filter_pred=filter_len)


class PoSDataset(LazyDataset):
    """Defines a dataset for PoS Tagging."""

    @staticmethod
    def sort_key(ex):
        return len(ex.words)

    def __init__(self, corpus, filter_pred=None):
        """Create a dataset from a list of Examples and Fields.

        Arguments:
            corpus: Corpus object.
            filter_pred (callable or None): Use only examples for which
                filter_pred(example) is True, or use all examples if None.
                Default is None.
        """
        # if we use LazyBucketIterator instead:
        # examples = iter(corpus)
        examples = list(corpus)
        fields = corpus.attr_fields
        super().__init__(examples, fields, filter_pred)

    def get_loss_weights(self):
        from sklearn.utils.class_weight import compute_class_weight
        tags_vocab = self.fields['tags'].vocab.stoi
        y = []
        for ex in self.examples:
            ex_classes = [tags_vocab[t] for t in ex.tags]
            y.extend(ex_classes)
        classes = list(set(y))
        return compute_class_weight('balanced', classes, y)
