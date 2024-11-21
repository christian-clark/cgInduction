import os, sys, gzip, shutil, argparse, time, random, logging, json, bidict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from configparser import ConfigParser
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import preprocess, postprocess, model_use
from top_models import TopModel
from eval.eval_access import eval_access
from cg_inducer import BasicCGInducer


DEFAULT_CONFIG = {
    "DEFAULT": {
        "model": "my_model",
        "seed": -1,
        "device": "cpu",
        "eval_device": "cpu",
        "optimizer": "adam",
        "max_grad_norm": 5,
        "learning_rate": 0.0001,
        "batch_size": 2,
        "max_vocab_size": 150000,
        "eval_steps": 2,
        "eval_start_epoch": 1,
        "start_epoch": 0,
        "max_epoch": 20,
        "logfile": "log.txt.gz",
        "model_type": "word",
        "rnn_hidden_dim": 512,
        "state_dim": 64,
        "eval_patient": 5
    }
}


DEBUG = True

def printDebug(*args, **kwargs):
    if DEBUG:
        print("DEBUG: ", end="")
        print(*args, **kwargs)


def random_seed(seed_value, use_cuda):
    printDebug("seeding to value {}".format(seed_value))
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars


def setup(eval_only=False):
    printDebug("main: setting up...")
    top_config = ConfigParser()
    top_config.read_dict(DEFAULT_CONFIG)
    if len(sys.argv) == 2:
        overrides = []
    elif len(sys.argv[2].split("=")) == 1:
        top_config.read(sys.argv[2])
        overrides = sys.argv[3:]
    else:
        overrides = sys.argv[2:]
    config = top_config["DEFAULT"]

    # any args after the config file override key-value pairs
    for kv in overrides:
        k, v = kv.split("=")
        config[k] = v

    # set seed before anything else.
    if config.getint("seed") < 0: # random seed if seed is set to negative values
        seed = int(int(time.time()) * random.random())
        config["seed"] = str(seed)
    printDebug("main: seed is value {}".format(config.getint("seed")))
    random_seed(config.getint("seed"), use_cuda=config["device"]=="cuda")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if eval_only:
        # testing must use an existing model
        assert "model_path" in config and os.path.exists(config["model_path"])
    else:
        if "model_path" not in config:
            model_path = os.path.join("outputs", config["model"])
            # NOTE: not sure what this line accomplishes; maybe useful
            # for parallel runs?
            time.sleep(random.uniform(0, 5))
            for i in range(100):
                checking_path = model_path+'_'+str(i)
                if not os.path.exists(checking_path):
                    config["model_path"] = checking_path
                    break
        else:
            if os.path.exists(config["model_path"]):
                shutil.rmtree(config["model_path"])
        os.makedirs(config["model_path"])

    printDebug("main: setup A")

    config_file = os.path.join(config["model_path"], "config.ini")
    with open(config_file, 'w') as cf:
        top_config.write(cf)

    logfile_fh = gzip.open(os.path.join(config["model_path"], config["logfile"]), 'wt')
    writer = SummaryWriter(os.path.join(config["model_path"], "tensorboard"), flush_secs=10)
    filehandler = logging.StreamHandler(logfile_fh)
    streamhandler = logging.StreamHandler(sys.stdout)
    handler_list = [filehandler, streamhandler]
    logging.basicConfig(level='INFO', format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', handlers=handler_list)

    # Dump configurations
    config_str = str(dict(config))
    logging.info(config_str)
    writer.add_text('args', config_str)

    assert (config["device"] == "cuda" and torch.cuda.is_available()) \
        or config["device"] == "cpu"

    # TODO may want to read in the koren phonetics arg from config
    train_sents = preprocess.read_corpus(config["train_sents"])

    logging.info('training instance: {}, training tokens: {}.'.format(
        len(train_sents), sum([len(s) - 1 for s in train_sents])
    ))

    #with open(config["train_gold_path"]) as tfh:
    #    train_tree_list = [x.strip() for x in tfh]

    #train_data, valid_data, train_tree_list, valid_tree_list = preprocess.divide(
    #    train_data,
    #    int(config["valid_size"]),
    #    train_tree_list,
    #    include_valid_in_train=False,
    #    all_train_as_valid=True
    #) # INCLUDE VALID IN TRAIN TO REDUCE TIME

    valid_sents_path = config.get("valid_sents", fallback=None)
    if valid_sents_path is None:
        valid_sents = None
    else:
        valid_sents = preprocess.read_corpus(valid_sents_path)

    valid_trees_path = config.get("valid_trees", fallback=None)
    if valid_trees_path is None:
        valid_trees = None
    else:
        with open(valid_trees_path) as f:
            valid_trees = [t.strip() for t in f]

    logging.info('training instance: {}, training tokens after division: {}.'.format(len(train_sents), sum([len(s) - 1 for s in train_sents])))
    logging.info('valid instance: {}, valid tokens: {}.'.format(len(valid_sents), sum([len(s) - 1 for s in valid_sents])))

    word_lexicon = bidict.bidict()

    # Maintain the vocabulary. vocabulary is used in either WordEmbeddingInput or softmax classification
    # NOTE: min count of 1 is currently hard-coded
    logging.warning('enforcing minimum count of 1')
    vocab = preprocess.get_truncated_vocab(
        train_sents, 1, config.getint("max_vocab_size")
    )

    # Ensure index of '<oov>' is 0
    special_words = [preprocess.OOV, preprocess.BOS, preprocess.EOS, preprocess.PAD, preprocess.LRB, preprocess.RRB]
    special_chars = [preprocess.BOS, preprocess.EOS, preprocess.OOV, preprocess.PAD, preprocess.BOW, preprocess.EOW]

    for special_word in special_words:
        if special_word not in word_lexicon:
            word_lexicon[special_word] = len(word_lexicon)

    for word, _ in vocab:
        if word not in word_lexicon:
            word_lexicon[word] = len(word_lexicon)

    logging.info('Vocabulary size: {0}'.format(len(word_lexicon)) + '; Max length: {}'.format(max([len(x) for x in word_lexicon])))
    logging.info('Vocabulary: {}'.format(word_lexicon))

    # Character Lexicon
    char_lexicon = bidict.bidict()

    for sentence in train_sents:
        for word in sentence:
            for ch in word:
                if ch not in char_lexicon:
                    char_lexicon[ch] = len(char_lexicon)

    for special_char in special_chars:
        if special_char not in char_lexicon:
            char_lexicon[special_char] = len(char_lexicon)

    logging.info('Char embedding size: {0}'.format(len(char_lexicon)))


    # training batch size for the pre training is 8 times larger than in eval
    train = preprocess.create_batches(
        train_sents,
        config.getint("batch_size"),
        word_lexicon,
        char_lexicon,
        device=config["device"],
    )

    logging.info('Evaluate every {0} epochs.'.format(config["eval_steps"]))

    if valid_sents is not None:
        valid = preprocess.create_batches(
            valid_sents,
            config.getint("batch_size"),
            word_lexicon,
            char_lexicon,
            eval=True,
            device=config["device"],
            eval_device=config["eval_device"]
        )
    else:
        valid = None

    logging.info('vocab size: {0}'.format(len(word_lexicon)))

    parser = BasicCGInducer(
        config,
        num_words=len(word_lexicon)
    )


    logging.info(
        "Total number of categories: {}".format(parser.qall)
    )
    logging.info(
        "Number of generated categories: {}".format(parser.qgen)
    )
    logging.info(
        "Examples of categories: {}".format(list(parser.ix2cat.items())[:100])
    )
    model = TopModel(parser, writer)

    printDebug("main: setup A0")
    logging.info(str(model))
    num_grammar_params = 0
    for param in model.parameters():
        num_grammar_params += param.numel()
    logging.info("Parser has {} parameters".format(num_grammar_params))
    printDebug("main: setup A1")

    model = model.to(config["device"])

    printDebug("main: setup B")
    # TODO hard-coded to use Adam?
    optimizer = optim.Adam(model.parameters(), lr=config.getfloat("learning_rate"))

    with open(os.path.join(config["model_path"], "char.dic"), 'w', encoding="utf-8") as fpo:
        for ch, i in char_lexicon.items():
            print("{0}\t{1}".format(ch, i), file=fpo)

    with open(os.path.join(config["model_path"], "word.dic"), 'w', encoding="utf-8") as fpo:
        for w, i in word_lexicon.items():
            print("{0}\t{1}".format(w, i), file=fpo)

    if "checkpoint" in config and config["checkpoint"] != "":
        checkpoint = torch.load(config["checkpoint"])
        model.load_state_dict(checkpoint)
        logging.info('Model loaded from {}.'.format(config["checkpoint"]))
    elif eval_only:
        # if doing eval only and checkpoint isn't specified, assume it's
        # model.pth
        checkpoint = torch.load(config["model_path"] + "/model.pth")
    printDebug("main: setup C")

    return config, model, optimizer, valid_sents, valid_trees, valid, train, logfile_fh


def train():
    config, model, optimizer, valid_sents, valid_trees, valid, train, logfile_fh = setup(eval_only=False)
    printDebug("main: done setting up...")
    torch.autograd.set_detect_anomaly(True)
    best_eval_likelihood = -1e+8
    patient = 0

    # evaluate the untrained model (written in log as epoch -1)
#    if valid_sents is not None:
#        model.to(config["eval_device"])
#        _, trees = model_use.parse_dataset(model, valid, -1)
#        valid_pred_trees = postprocess.print_trees(
#            trees, valid_sents, -1, config["model_path"]
#        )
#        if valid_trees is not None:
#            eval_access(valid_pred_trees, valid_trees, model.writer, -1)
#        model.to(config["device"])
        
    printDebug("main: starting train loop...")
    for epoch in range(config.getint("start_epoch"), config.getint("max_epoch")):
        optimizer = model_use.train_model(
            epoch, model, optimizer, train, config.getfloat("max_grad_norm")
        )

        if ((epoch - config.getint("eval_start_epoch")) % config.getint("eval_steps") == 0 \
                or epoch + 1 == config.getint("max_epoch")) \
            and epoch >= config.getint("eval_start_epoch"):

            logging.info("EVALING.")

            if valid_sents:
                model.to(config["eval_device"])
                total_eval_likelihoods, trees = model_use.parse_dataset(model, valid, epoch)
                valid_pred_trees = postprocess.print_trees(
                    trees, valid_sents, epoch, config["model_path"]
                )
                if valid_trees is not None:
                    eval_access(valid_pred_trees, valid_trees, model.writer, -1)
                model.to(config["device"])

            else:
                total_eval_likelihoods = model_use.likelihood_dataset(model, train, epoch)

            if total_eval_likelihoods > best_eval_likelihood:
                logging.info("Better model found based on likelihood: {}! vs {}".format(total_eval_likelihoods, best_eval_likelihood))
                best_eval_likelihood = total_eval_likelihoods
                patient = 0
                model_save_path = os.path.join(config["model_path"], "model.pth")
                torch.save(model.state_dict(), model_save_path)

            else:
                patient += 1
                if patient >= config.getint("eval_patient"):
                    break

            if config.getboolean("dump_grammar"):
                logging.info("======== START GRAMMAR DUMP ========")
                #torch.set_printoptions(precision=2, linewidth=120)
                torch.set_printoptions(sci_mode=False, precision=2, linewidth=300)
                logging.info("word_emb for important words")
                # indices 0-5 are things like <oov>
                # dim: words x preds
                logging.info(model.inducer.word_emb.weight[6:].softmax(dim=1))
                logging.info("all_cats: {}".format(model.inducer.all_cats))
                logging.info("gen_cats: {}".format(model.inducer.gen_cats))
                logging.info("ix2cat: {}".format(model.inducer.ix2cat))
                logging.info("ix2cat_gen: {}".format(model.inducer.ix2cat_gen))

                logging.info("root probs")
                logging.info(torch.exp(model.inducer.parser.root_probs))

                logging.info("split probs")
                logging.info(torch.exp(model.inducer.parser.split_probs))

                qgen = model.inducer.qgen
                # dim: qall x 4qgen
                rule_scores = model.inducer.rule_mlp(model.inducer.fake_emb)
                rule_scores[:, :qgen] += model.inducer.rfunc_mask
                rule_scores[:, qgen:2*qgen] += model.inducer.lfunc_mask
                rule_scores[:, 2*qgen:3*qgen] += model.inducer.mod_mask[None, :]
                rule_scores[:, 3*qgen:] += model.inducer.mod_mask[None, :]
                rule_probs = rule_scores.softmax(dim=1)
                logging.info("rule_probs_Aa")
                logging.info(rule_probs[:, :qgen])
                logging.info("rule_probs_Ab")
                logging.info(rule_probs[:, qgen:2*qgen])
                logging.info("rule_probs_Ma")
                logging.info(rule_probs[:, 2*qgen:3*qgen])
                logging.info("rule_probs_Mb")
                logging.info(rule_probs[:, 3*qgen:])


                logging.info("combined G Aa probs")
                logging.info(torch.exp(model.inducer.parser.full_G[:, :qgen]))
                logging.info("combined G Ab probs")
                logging.info(torch.exp(model.inducer.parser.full_G[:, qgen:2*qgen]))
                logging.info("combined G Ma probs")
                logging.info(torch.exp(model.inducer.parser.full_G[:, 2*qgen:3*qgen]))
                logging.info("combined G Mb probs")
                logging.info(torch.exp(model.inducer.parser.full_G[:, 3*qgen:]))

                word_dist = torch.exp(model.inducer.emit_prob_model.dist)
                logging.info("word_dist shape")
                logging.info(word_dist.shape)
                logging.info("word_dist")
                logging.info(word_dist)
                logging.info("======== END GRAMMAR DUMP ========")

    model.writer.close()
    logfile_fh.close()



def test():
    config, model, _, valid_sents, valid_trees, valid, _, logfile_fh = setup(eval_only=True)
    logging.info('EVALING.')
    model.to(config["eval_device"])
    _, trees = model_use.parse_dataset(model, valid, 0)
    valid_pred_trees = postprocess.print_trees(
        trees, valid_sents, 0, config["model_path"]
    )
    eval_access(valid_pred_trees, valid_trees, model.writer, 0)
    #model.to(config["device"])
    model.writer.close()
    logfile_fh.close()


if __name__ == "__main__":
    printDebug("kicking off main script...")
    if len(sys.argv) >= 3 and sys.argv[1] == 'train':
        train()
        logging.shutdown()
    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        test()
    else:
        print('Usage: {0} [train|test] [config] [overrides]'.format(sys.argv[0]), file=sys.stderr)
