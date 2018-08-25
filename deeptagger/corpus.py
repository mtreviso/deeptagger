from torchtext import data
import logging
from random import shuffle as shuffle_list


class Corpus:

    def __init__(self, fields, delimiter_word=' ', delimiter_tag='/'):
        """Base class for a PoS Corpus.
        Arguments:
            fields: a list of tuples where the first element is an
                    attr name and the second is a torchtext's Field object
            delimiter_word: char that delimiters tokens
            delimiter_tag: char that delimiters word tokens from tags tokens
        """
        # list of fields containing the same number of examples
        self.fields_examples = []
        # list of name of attrs and their corresponding torchtext fields
        self.attr_fields = fields
        # delimiters
        self.del_word = delimiter_word
        self.del_tag = delimiter_tag
        # the number of examples in the corpus
        self.nb_examples = 0

    def read(self, filepath):
        """
        filepath: path to a file with the following format:
                  words are delimited by `delimiter_word` and
                  tags are delimited from words by `delimiter_tag`
                  e.g. The_ART princess_S is_V pretty_ADJ
                  where delimiter_word=' ' and delimiter_tag='_'
        """
        # warning for two well known Brazilian Portuguese PoS corpus
        if 'macmorpho' in filepath and self.del_tag != '_':
            logging.warning('Default MacMorpho delimiter tag is `_`, '
                            'but you passed `{}`'.format(self.del_tag))
        if 'tychobrahe' in filepath and self.del_tag != '/':
            logging.warning('Default TychoBrahe delimiter tag is `/`, '
                            'but you passed `{}`'.format(self.del_tag))

        # load the file and fill words and tags examples
        words_for_example = []
        tags_for_example = []
        with open(filepath, 'r', encoding='utf8') as f:
            for line in f:
                words_tags_line = self._normalize(line)
                words, tags = zip(
                    *list(
                        map(
                            lambda x: x.split(self.del_tag),
                            words_tags_line.split(self.del_word)
                        )
                    )
                )
                words_for_example.append(' '.join(words))
                tags_for_example.append(' '.join(tags))
        self.fields_examples = [words_for_example, tags_for_example]

        # then add each corresponding sentence from each field
        nb_lines = [len(fe) for fe in self.fields_examples]
        # assert files have the same size
        assert min(nb_lines) == max(nb_lines)
        self.nb_examples = nb_lines[0]

    def __iter__(self):
        # you may want to use list(dataset) before passing to torchtext's
        # iterators
        for j in range(self.nb_examples):
            fields_values_for_example = [self.fields_examples[i][j]
                                         for i in range(len(self.attr_fields))]
            yield data.Example.fromlist(fields_values_for_example,
                                        self.attr_fields)

    def _normalize(self, text):
        return text.strip()

    def split(self, ratio=0.8, shuffle=True):
        first_corpus = Corpus(self.attr_fields, self.del_word, self.del_tag)
        second_corpus = Corpus(self.attr_fields, self.del_word, self.del_tag)
        words, tags = self.fields_examples
        limit = int(self.nb_examples * ratio)
        id_list = list(range(self.nb_examples))
        if shuffle is True:
            shuffle_list(id_list)
        first_words = [words[idx] for idx in id_list[:limit]]
        first_tags = [tags[idx] for idx in id_list[:limit]]
        second_words = [words[idx] for idx in id_list[limit:]]
        second_tags = [tags[idx] for idx in id_list[limit:]]
        first_corpus.fields_examples = [first_words, first_tags]
        first_corpus.nb_examples = len(first_words)
        second_corpus.fields_examples = [second_words, second_tags]
        second_corpus.nb_examples = len(second_words)
        return first_corpus, second_corpus


# if __name__ == '__main__':
#     import sys
#     filepath = sys.argv[1]
#     words_field = data.Field()
#     tags_field = data.Field(unk_token=None, pad_token=None)
#     fields = [('words', words_field), ('tags', tags_field)]

#     corpus = Corpus(fields, delimiter_word=' ', delimiter_tag='_')
#     corpus.read(filepath)
#     for ex in corpus:
#         print(ex.words)
#         print(ex.tags)
#         break

#     c1, c2 = corpus.split(shuffle=False)
#     print(corpus.nb_examples, c1.nb_examples, c2.nb_examples)
#     print(next(iter(c1)).words)
#     print(next(iter(c2)).words)

#     assert(c1.fields_examples[0][0] == corpus.fields_examples[0][0])
#     mid = c1.nb_examples
#     assert(c2.fields_examples[0][0] == corpus.fields_examples[0][mid])
