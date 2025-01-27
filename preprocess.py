import logging
import random
import torch
import gzip


def read_corpus(path):
    """
    read raw text file
    max word len is 28
    """
    data = []
    if path.endswith('gz'):
        fin = gzip.open(path, 'rt', encoding='utf-8')
    else: fin = open(path, 'r', encoding='utf-8')
    for line in fin:
        sent = list()
        for token in line.strip().split():
            sent.append(token.lower())
        data.append(sent)
    fin.close()
    return data

def create_one_batch(x, word2id, sort=True, device='cpu'):
    batch_size = len(x)
    lst = list(range(batch_size))
    if sort:
        lst.sort(key=lambda l: -len(x[l]))

    x = [x[i] for i in lst]
    lens = [len(x[i]) for i in lst]
    max_len = max(lens)

    batch_w = torch.LongTensor(batch_size, max_len).fill_(0).to(device)
    for i, x_i in enumerate(x):
        for j, x_ij in enumerate(x_i):
            assert x_ij in word2id
            batch_w[i][j] = word2id.get(x_ij)

    masks = [torch.LongTensor(batch_size, max_len).fill_(0).to(device), [], []]

    for i, x_i in enumerate(x):
        for j in range(len(x_i)):
            masks[0][i][j] = 1
            if j + 1 < len(x_i):
                masks[1].append(i * max_len + j)
            if j > 0:
                masks[2].append(i * max_len + j)

    assert len(masks[1]) <= batch_size * max_len
    assert len(masks[2]) <= batch_size * max_len

    masks[1] = torch.LongTensor(masks[1]).to(device)
    masks[2] = torch.LongTensor(masks[2]).to(device)

    return batch_w, lens, masks


# shuffle training examples and create mini-batches
def create_batches(
        x, batch_size, word2id, eval=False, perm=None, 
        shuffle=True, sort=True, device="cpu", eval_device="cpu"
    ):

    if eval:
        device = eval_device

    lst = perm or list(range(len(x)))
    if shuffle:
        random.shuffle(lst)

    if sort:
        lst.sort(key=lambda l: -len(x[l]))

    sorted_x = [x[i] for i in lst]

    sum_len = 0.0
    # TODO refactor using a Batch class
    batches_w, batches_c, batches_var_c, batches_lens, batches_masks, batch_indices = [], [], [], [], [], []
    size = batch_size
    cur_len = 0
    start_id = 0
    end_id = 0
    for sorted_index in range(len(sorted_x)):
        if cur_len == 0:
            cur_len = len(sorted_x[sorted_index])
            if len(sorted_x) > 1:
                continue
        if cur_len != len(sorted_x[sorted_index]) or sorted_index - start_id == batch_size or sorted_index == len(sorted_x)-1:
            if sorted_index != len(sorted_x) - 1:
                end_id = sorted_index
            else:
                end_id = None

            if (end_id is None and len(sorted_x[sorted_index]) == cur_len) or end_id is not None:
                bw, blens, bmasks = create_one_batch(sorted_x[start_id: end_id], word2id, sort=sort,
                                                         device=device)
                batch_indices.append(lst[start_id:end_id])
                sum_len += sum(blens)
                batches_w.append(bw)
                batches_lens.append(blens)
                batches_masks.append(bmasks)
                start_id = end_id
                cur_len = len(sorted_x[sorted_index])
            else:
                end_id = sorted_index
                bw, blens, bmasks = create_one_batch(sorted_x[start_id: end_id], word2id, sort=sort,
                                                         device=device)
                batch_indices.append(lst[start_id:end_id])
                sum_len += sum(blens)
                batches_w.append(bw)
                batches_lens.append(blens)
                batches_masks.append(bmasks)

                bw, blens, bmasks = create_one_batch(sorted_x[-1:], word2id, sort=sort,
                                                         device=device)
                batch_indices.append(lst[-1:])
                sum_len += sum(blens)
                batches_w.append(bw)
                batches_lens.append(blens)
                batches_masks.append(bmasks)

    nbatch = len(batch_indices)

    logging.info("{} batches, avg len: {:.1f}, max len {}, min len {}.".format(nbatch, sum_len / len(x), len(sorted_x[0]),
                                                                               len(sorted_x[-1])))

    if sort:
        perm = list(range(nbatch))
    random.shuffle(perm)
    batches_w = [batches_w[i] for i in perm]
    batches_lens = [batches_lens[i] for i in perm]
    batches_masks = [batches_masks[i] for i in perm]
    batch_indices = [batch_indices[i] for i in perm]

    return batches_w, batches_lens, batches_masks, batch_indices


def read_markers(fname):
    markers = [0]
    with open(fname) as fh:
        for l in fh:
            marker = int(l.strip().split(' ')[1])
            markers.append(marker)
    return markers

