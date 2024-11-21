import torch
import time
import logging
import numpy as np

DEBUG = False

def printDebug(*args, **kwargs):
    if DEBUG:
        print("DEBUG: ", end="")
        print(*args, **kwargs)


def train_model(
        epoch, model, optimizer, train, max_grad_norm
    ):
    """
    Training model for one epoch
    """
    torch.autograd.set_detect_anomaly(True)
    model.train()
    total_loss, total_tag = 1e-7, 1e-7
    cnt = 0
    start_time = time.time()
    train_w, _, _, train_lens, _, _ = train
    max_cnt = len(train_w)
    tenths = list([int(max_cnt / 10) * i for i in range(1, 10)])
    # TODO what does lst accomplish?
    lst = list(range(len(train_w)))
    train_w = [train_w[l] for l in lst]
    train_lens = [train_lens[l] for l in lst]

    for ws, lens in zip(train_w, train_lens):
        cnt += 1
        optimizer.zero_grad()

        loss = model.forward(ws)
        loss.backward()
        total_loss += loss.item()

        total_tag += sum(lens)

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        global_step = max_cnt * epoch + cnt

        if cnt in tenths:
            logging.info("Epoch={} iter={} lr={:.5f} train loss={:.4f} time={:.2f}s".format(
                epoch, cnt, optimizer.param_groups[0]['lr'],
                total_loss / total_tag, time.time() - start_time
            ))

            start_time = time.time()
            model.writer.add_scalar('train_accumulative/average_total_loss', total_loss / total_tag, global_step)

    model.writer.add_scalar('train_epochwise/average_total_loss', total_loss / total_tag, epoch)
    return optimizer


def parse_dataset(model, dataset, epoch, section='dev'):
    model.eval()
    with torch.no_grad():
        train_w, _, _, train_lens, _, train_indices = dataset
        trees = [None] * sum([len(x) for x in train_indices])
        total_structure_loss = 0
        total_num_tags = sum([sum(x) for x in train_lens])
        for batch_index, (w, indices) in enumerate(zip(train_w, train_indices)):
            if batch_index == 0:
                structure_loss, v_treelist = model.parse(w, set_grammar=True)
            else:
                structure_loss, v_treelist = model.parse(w, set_grammar=False)
            for t_id, t in zip(indices, v_treelist):
                trees[t_id] = t
            total_structure_loss += structure_loss
        if model.writer is not None:
            model.writer.add_scalar(section+'_epochwise/average_structure_loss', total_structure_loss / total_num_tags, epoch)
            logging.info(
                'Epoch {} EVALUATION | Structure loss {:.4f} '.format(epoch, total_structure_loss))
        printDebug("parse_dataset train_w:", train_w)
    model.train()
    return total_structure_loss, trees


def likelihood_dataset(model, dataset, epoch, section='dev'):
    model.eval()
    with torch.no_grad():
        train_w, _, _, train_lens, _, _ = dataset
        total_structure_loss = 0
        total_num_tags = sum([sum(x) for x in train_lens])
        for batch_index, w in enumerate(train_w):
            if batch_index == 0:
                structure_loss = model.likelihood(w, set_grammar=True)
            else:
                structure_loss = model.likelihood(w, set_grammar=False)
            total_structure_loss += structure_loss
        model.writer.add_scalar(section+'_epochwise/average_structure_loss', total_structure_loss / total_num_tags, epoch)
        logging.info(
            'Epoch {} EVALUATION | Structure loss {:.4f} '.format(epoch, total_structure_loss))
    model.train()
    return total_structure_loss


def eval_model(model, valid):
    model.eval()
    if model.config['classifier']['name'].lower() == 'cnn_softmax' or \
            model.config['classifier']['name'].lower() == 'sampled_softmax':
        model.classify_layer.update_embedding_matrix()
    total_loss, total_tag = 0.0, 0
    valid_w, valid_c, valid_lens, valid_masks = valid
    for w, c, lens, masks in zip(valid_w, valid_c, valid_lens, valid_masks):
        loss_forward, loss_backward = model.forward(w, c, masks)
        total_loss += loss_forward.data[0]
        n_tags = sum(lens)
        total_tag += n_tags
    model.train()
    return np.exp(total_loss / total_tag)
