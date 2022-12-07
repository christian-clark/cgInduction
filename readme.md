# Categorial Grammar Induction
An unsupervised categorial grammar induction system based on [Jin et al. (2021)](https://aclanthology.org/2021.findings-emnlp.371/).

Major dependencies include:

- Python 3.7+
- PyTorch 1.7.0+
- TensorBoard 2.3.0+
- Bidict

## Training
`main.py` is the main training script. Sample commands for training the induction model can be found under the `exps` directory:

```
python main.py train --seed -1 \
                     --train_path data/Jong.010322.linetoks \
                     --train_gold_path data/Jong.010322.linetrees \
                     --model_type char --model char_jong_c90 \
                     --num_nonterminal 45 --num_preterminal 45 \
                     --state_dim 128 --rnn_hidden_dim 512 \
                     --max_epoch 45 --batch_size 2 \
                     --optimizer adam --device cuda \
                     --eval_device cuda --eval_steps 2 \
                     --eval_start_epoch 1 --eval_parsing
```
