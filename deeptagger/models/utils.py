import copy

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def unmask(tensor, mask):
    """
    Unmask a tensor and convert it back to a list of lists.
    :param tensor: a torch.tensor object
    :param mask: a torch.tensor object with 1 indicating a valid position
                 and 0 elsewhere
    :return: a list of lists with variable length
    """
    lengths = mask.int().sum(dim=-1).tolist()
    return [x[:lengths[i]].tolist() for i, x in enumerate(tensor)]


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


def clones(module, N):
    """Produce N identical layers."""
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def sequence_mask(lengths, max_len=None):
    """Creates a boolean mask from sequence lengths."""
    if max_len is None:
        max_len = lengths.max()
    aranges = torch.arange(max_len).repeat(lengths.shape[0], 1)
    return aranges < lengths.unsqueeze(1)


def unsqueeze_as(tensor, as_tensor, dim=-1):
    """Expand new dimensions based on a template tensor along `dim` axis."""
    x = tensor
    while x.dim() < as_tensor.dim():
        x = x.unsqueeze(dim)
    return x


def unsqueeze_to_ndim(tensor, ndim, dim=1):
    """Expand `ndim` new dimensions along `dim` axis."""
    x = tensor
    while x.dim() < ndim:
        x = tensor.unsqueeze(dim)
    return x


def make_mergeable_tensors(t1, t2):
    """Expand a new dimension in t1 and t2 such that both have the same timesteps:
        t1.shape = [bs, n, d1] and t2.shape = [bs, m, d2]
        x1.shape = [bs, n, 1, d1] and x2.shape = [bs, 1, m, d2]
        x1.shape = [bs, n, m, d1] and x2.shape = [bs, n, m, d2]
    """
    x1 = unsqueeze_to_ndim(t1, 4, 1) if t1.dim() > 4 else t1
    x2 = unsqueeze_to_ndim(t2, 4, 1) if t2.dim() > 4 else t2
    new_shape = [-1] * (x1.dim() + 1)
    new_shape[-2] = x1.shape[-2]
    new_shape[-3] = x2.shape[-2]
    x1 = x1.unsqueeze(-2).transpose(-2, -3).expand(new_shape)
    x2 = x2.unsqueeze(-2).expand(new_shape)
    return x1, x2


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


def indexes_to_words(indexes, itos):
    """
    Transofrm indexes to words using itos list
    :param indexes: list of lists of ints
    :param itos: list mapping integer to string
    :return: list of lists of strs
    """
    words = []
    for sample in indexes:
        words.append([itos[i] for i in sample])
    return words
