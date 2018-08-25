import numpy as np

from numpy.lib.stride_tricks import as_strided
import torch
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


def unroll(list_of_lists, rec=False):
    """
    :param list_of_lists: a list that contains lists
    :param rec: unroll recursively
    :return: a flattened list
    """
    if not isinstance(list_of_lists[0], (np.ndarray, list)):
        return list_of_lists
    new_list = [item for l in list_of_lists for item in l]
    if rec and isinstance(new_list[0], (np.ndarray, list)):
        return unroll(new_list, rec=rec)
    return new_list


def pad_sequences(sequences, maxlen=None, mask_value=0):
    """
    :param sequences: list of sequence of ids
    :param maxlen: if not specified, maxlen is max sentence length
    :param mask_value: the value to be used for padding
    :return: a np array with shape (nb_samples, maxlen)
    """
    dtype = 'int32'
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max([len(s) for s in sequences])
    x = (np.ones((nb_samples, maxlen)) * mask_value).astype(dtype)
    for idx, s in enumerate(sequences):
        x[idx, : len(s)] = s
    return x


def pad_sequences_3d(sequences, maxlen=None, mask_value=0):
    """
    :param sequences: list of sequence of ids
    :param maxlen: if not specified, maxlen is max sentence length
    :param mask_value: the value to be used for padding
    :return: a np array with shape (nb_samples, maxlen)
    """
    dtype = 'int32'
    nb_samples = len(sequences)
    nb_features = len(sequences[0][0])
    if maxlen is None:
        maxlen = np.max([len(s) for s in sequences])

    x = (np.ones((nb_samples, maxlen, nb_features)) * mask_value).astype(dtype)
    for idx, s in enumerate(sequences):
        x[idx, : len(s)] = s
    return x


def unpad_sequences(padded_sequences, map_with=None, mask_value=0):
    """
    :param padded_sequences: a np array that was padded using pad_sequences
    :param map_with: a list of lists with the same nb of elements
                     as :padded_sequences:
    :param mask_value: the value to be used for padding in case map_with is
                       not provided
    :return: a list with the original sequence structure
    """
    unpadded = []
    if map_with is not None:
        for i, sequence in enumerate(map_with):
            unpadded.extend(padded_sequences[i, : len(sequence)].tolist())
    else:
        for sequence in padded_sequences:
            unpadded.extend(sequence[sequence != mask_value].tolist())
    return np.array(unpadded)


def vectorize(tensor, one_hot_dim=None):
    """
    :param tensor: numpy array of sequences of ids
    :param one_hot_dim: if not specified, max value in tensor + 1 is used
    :return:
    """
    if not one_hot_dim:
        one_hot_dim = tensor.max() + 1

    if len(tensor.shape) == 1:
        # It's a vector; return a 2d tensor
        tensor_2d = np.zeros((tensor.shape[0], one_hot_dim), dtype=np.bool8)
        for i, val in np.ndenumerate(tensor):
            tensor_2d[i, val] = 1
        return tensor_2d

    tensor_3d = np.zeros(
        (tensor.shape[0], tensor.shape[1], one_hot_dim), dtype=np.bool8
    )
    for (i, j), val in np.ndenumerate(tensor):
        if val < one_hot_dim:
            tensor_3d[i, j, val] = 1
    return tensor_3d


def unvectorize(tensor):
    """
    :param tensor: numpy array of sequences of ids
    :return: the row indices that maximizes this tensor
    """
    return tensor.argmax(axis=-1)


def unconvolve_sequences(window):
    """
    :param window: a numpy array of sequences of ids that was windowed
    :return: the middle column
    """
    if len(window.shape) == 1:
        # it is already a vector
        return window
    middle = window.shape[1] // 2
    return window[:, middle]


def unconvolve_sequences_3d(window):
    """
    :param window: a 1-hot numpy array of sequences of ids
    :return: the middle column of the window without unvectorized
    """
    seq = np.ones((window.shape[0], 1))
    middle = window.shape[1] // 2
    for i, row in enumerate(window):
        seq[i] = unvectorize(row[middle])
    return seq


def convolve_sequence(sequence, window_size, pad_value=0):
    pad = np.ones(window_size // 2, dtype=np.int) * pad_value

    def pad_sequence(sequence):
        sequence = np.array(sequence)
        if pad_value is not None:
            sequence = np.hstack((pad, sequence))
            sequence = np.hstack((sequence, pad))
        return sequence

    return ngrams_via_striding(pad_sequence(sequence), window_size)


def convolve_sequences(sequences, window_size, pad_value=0):
    """
    Convolve around each element in each sequence.
    :param sequences: list of lists with possibly varying sizes
    :param window_size: if odd, align elements at the center
    :param pad_value: padding values for windows of elements
    at the start and end of sequences; if equal to None, the corresponding
    padding is not added (see consequence on the return value)
    :return: convolved sequences of size (number of words, window_size)
    """
    if pad_value is not None:
        lines = sum(len(ws) for ws in sequences)
    else:
        lines = sum(len(ws) - window_size + 1 for ws in sequences)
    window_array = np.zeros((lines, window_size), dtype=np.int)

    i = 0
    for seq in sequences:
        vectors = convolve_sequence(seq, window_size, pad_value=pad_value)
        window_array[i: i + len(vectors), :] = vectors
        i += len(vectors)
    return window_array


def ngrams_via_striding(array, order):
    # https://gist.github.com/mjwillson/060644552eb037ebb3e7
    itemsize = array.itemsize
    assert array.strides == (itemsize,)
    return as_strided(
        array, (max(array.size + 1 - order, 0), order), (itemsize, itemsize)
    )


def apply_packed_sequence(rnn, embedding, lengths):
    """ Runs a forward pass of embeddings through an rnn using packed sequence.
    Args:
       rnn: The RNN that that we want to compute a forward pass with.
       embedding (FloatTensor b x seq x dim): A batch of sequence embeddings.
       lengths (LongTensor batch): The length of each sequence in the batch.

    Returns:
       output: The output of the RNN `rnn` with input `embedding`
    """
    # Sort Batch by sequence length
    lengths_sorted, permutation = torch.sort(lengths, descending=True)
    embedding_sorted = embedding[permutation]

    # Use Packed Sequence
    embedding_packed = pack(embedding_sorted, lengths_sorted, batch_first=True)
    outputs_packed, _ = rnn(embedding_packed)
    outputs_sorted, _ = unpack(outputs_packed, batch_first=True)

    # Restore original order
    _, permutation_rev = torch.sort(permutation, descending=False)
    outputs = outputs_sorted[permutation_rev]
    return outputs


def stop_to_pad(target, stop, pad):
    """Replaces STOP tokens with PAD.

    Because of different length of sequences,
    there might be remaining STOP tokens.
    Replacing them with PAD ensures they are ignored during loss
    computation.

    """
    return target.masked_fill(target == stop, pad)
