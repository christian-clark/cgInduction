# Categorial Grammar Induction
The categorial grammar induction system presented in "Categorial Grammar Induction from Raw Data" by Clark and Schuler (2023).

Major dependencies include:

- Conda
- Python 3.7+
- PyTorch 1.7.0+
- TensorBoard 2.3.0+
- Bidict

## Setup
The environment for running the induction system can be set up in [Conda](https://docs.conda.io/en/latest/) using the provided YAML file:

```
conda env create -f environment.yml
conda activate cgInduction
```


## Training
`main.py` is the main training script.
Sample command:

```
python main.py train \
    model=adamPrim2Depth2 \
    train_sents=adam.senttoks \
    valid_sents=adam.senttoks \
    valid_trees=adam.senttrees \
    num_primitives=2 \
    max_func_depth=2
```

The default model configuration is defined in `DEFAULT_CONFIG` at the beginning of this script.
Values in the default configuration can be overridden on the command line:

```
python main.py train \
    model=adamPrim2Depth2 \
    train_sents=adam.senttoks \
    valid_sents=adam.senttoks \
    valid_trees=adam.senttrees \
    num_primitives=2 \
    max_func_depth=2 \
    device="cuda" \
    eval_device="cuda" \
    learning_rate=0.05
```

Configuration can also be specified in an INI file, with additional overrides optionally following:

```
python main.py train config.ini \
    device="cuda" \
    eval_device="cuda"
```

Command-line overrides take first precedence, followed by the INI file, followed by `DEFAULT_CONFIG`.

Sample INI files can be found in the `ini` directory.

## Configuration
- `model`: Output directory name
- `model_path`: Path to output directory
- `train_sents`: File with sentences for training, one per line
- `valid_sents`: File with sentences for evaluation, one per line
- `valid_trees`: File with annotated trees for evaluation sentences, one per line in Penn Treebank format
- `num_primitives`: The number of primitives used in induced categories
- `max_func_depth`: The maximum category depth (described in the paper)
- `category_list`: As an alternative to providing `num_primitives` and `max_func_depth`, a file containing a predetermined set of categories (one per line) may be provided. If this is provided then `num_primitives` and `max_func_depth` should not be given, and vice versa
- `seed`: Random seed used by `random`, `numpy`, etc. When this is set to -1, a different seed will be chosen for each run
- `device`: The device (CPU or GPU) to be used by PyTorch during training
- `eval_device`: The device (CPU or GPU) to be used by PyTorch during testing
- `optimizer`: Optimization algorithm, e.g. Adam
- `max_grad_norm`: Maximum gradient norm
- `learning_rate`
- `batch_size`
- `max_vocab_size`
- `eval_steps`: How many epochs are completed between each evaluation
- `eval_start_epoch`: First epoch when evaluation is performed
- `start_epoch`: First training epoch (typically 0)
- `max_epoch`: Maximum number of training epochs
- `logfile`: Filename where model info is logged
- `model_type`: `word` or `char` for a word-level or character-level model (the ACL Findings paper uses word-level)
- `rnn_hidden_dim`: Hidden dimension of the RNN used in the character-level model
- `state_dim`: Hidden dimension used in the MLPs
- `eval_patient`: Maximum number of epochs showing no improvement before training is halted
