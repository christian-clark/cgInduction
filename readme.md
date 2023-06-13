# Categorial Grammar Induction
The categorial grammar induction system presented in Clark and Schuler (2023).

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
python main.py train
```

The default model configuration is defined in `DEFAULT_CONFIG` at the beginning of this script.
Values in the default configuration can be overridden on the command line:

```
python main.py train
```

Configuration can also be specified in an INI file, with additional overrides on the command line:

```
python main.py train
```

Command-line key-value pairs take first precedence, followed by the INI file, followed by `DEFAULT_CONFIG`.

Sample INI files can be found in the `ini` directory.

## Configuration

- `seed`: Random seed used by `random`, `numpy`, etc. When this is set to -1, a different seed will be chosen for each run
- `device`: The device (CPU or GPU) to be used by PyTorch during training
- `eval_device`: The device (CPU or GPU) to be used by PyTorch during testing
- TO BE CONTINUED

