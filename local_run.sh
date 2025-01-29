TRAIN=/mnt/c/Users/cecla/Documents/git/cgInduction/inputs/catsEatAnts
DEV=$TRAIN
MODEL=catsEatAnts
MODE=train
LOSS_TYPE=marginal
#LOSS_TYPE=best_parse
# 40 mins for 5000 epochs
MAX_EPOCH=10
# "default" learning rate
LEARNING_RATE=0.0001
#LEARNING_RATE=1
EVAL_STEPS=2
MODEL_TYPE=word
DUMP_GRAMMAR=yes
BATCH_SIZE=100
#USE_ENTROPY_LOSS=yes
ASSOC_ARG1=/mnt/c/Users/cecla/Documents/git/cgInduction/inputs/assoc_arg1
ASSOC_ARG2=/mnt/c/Users/cecla/Documents/git/cgInduction/inputs/assoc_arg2
PREDICATES=/mnt/c/Users/cecla/Documents/git/cgInduction/inputs/predicates
CAT_LIST=/mnt/c/Users/cecla/Documents/git/cgInduction/inputs/categories
DEVICE=cpu
EVAL_DEVICE=cpu

python3 /mnt/c/Users/cecla/Documents/git/cgInduction/main.py $MODE \
    dump_grammar=$DUMP_GRAMMAR \
    learning_rate=$LEARNING_RATE \
    eval_steps=$EVAL_STEPS \
    max_epoch=$MAX_EPOCH \
    train_sents=$TRAIN \
    valid_sents=$DEV \
    loss_type=$LOSS_TYPE \
    category_list=$CAT_LIST \
    batch_size=$BATCH_SIZE \
    associations_arg1=$ASSOC_ARG1 \
    associations_arg2=$ASSOC_ARG2 \
    predicates=$PREDICATES \
    device=$DEVICE \
    eval_device=$EVAL_DEVICE \
    use_entropy_loss=$USE_ENTROPY_LOSS \
    model=$MODEL
