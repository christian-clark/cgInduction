#!/bin/bash
##SBATCH --partition=schuler
##SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
##SBATCH --mem=300gb
#SBATCH --mem=50gb
##SBATCH --time=120:00:00
#SBATCH --time=1:00:00

GPU=0
set -x
set -e
if [ $GPU -eq 0 ]; then
    DEVICE=cpu
    EVAL_DEVICE=cpu
else
    DEVICE=cuda
    EVAL_DEVICE=cuda
fi


MODEL=sampling/test
TRAIN=/home/clark.3664/projects/cg_induction/data/childes/adam.20.senttoks
DEV=$TRAIN
DEV_TREES=/home/clark.3664/projects/cg_induction/data/childes/adam.20.senttrees
MAX_EPOCH=5
EVAL_STEPS=2
STATE_DIM=16
LEARNING_RATE=0.0001
#CAT_LIST=/home/clark.3664/projects/cg_induction/categories/p_q/no_scores/maxCats500/p0.5_q0.1
#CAT_LIST=/home/clark.3664/projects/cg_induction/categories/p_q/no_scores/maxCats100/p0.5_q0.1
CAT_LIST=/home/clark.3664/git/temp2/cgInduction/categories/toy_depth2
SAMPLE_COUNT=3
COOCCURRENCE_SCORES_DIR=/home/clark.3664/git/temp2/cgInduction/cooccurrences


#python3 /home/clark.3664/git/cgInduction/main.py train \
python3 /home/clark.3664/git/temp2/cgInduction/main.py train \
    learning_rate=$LEARNING_RATE \
    state_dim=$STATE_DIM \
    max_epoch=$MAX_EPOCH \
    train_sents=$TRAIN \
    valid_sents=$DEV \
    valid_trees=$DEV_TREES \
    batch_size=2 \
    category_list=$CAT_LIST \
    device=$DEVICE \
    eval_device=$EVAL_DEVICE \
    sample_count=$SAMPLE_COUNT \
    cooccurrence_scores_dir=$COOCCURRENCE_SCORES_DIR \
    model=$MODEL

    #state_dim=64 \
    #seed=$SEED \
    #category_list=/home/clark.3664/projects/cg_induction/categories/grid_search/just_cat/$f \
    #arg_depth_penalty=$ARG_DEPTH_PENALTY \
    #labeled_eval=no \
    #/home/clark.3664/git/temp/cgInduction/config.ini \


#TRAIN=/home/clark.3664/projects/cg_induction/data/childes/adam.entire.senttoks
#TRAIN=/home/clark.3664/projects/cg_induction/data/childes/adam.20.senttoks
#DEV_TREES=/home/clark.3664/projects/cg_induction/data/childes/adam.entire.senttrees
#DEV_TREES=/home/clark.3664/projects/cg_induction/data/childes/adam.20.senttrees
#CAT_FILES=(
#    maxCats100/p0.001_q0.5
#)
#CAT_LIST=/home/clark.3664/projects/cg_induction/categories/p_q/no_scores/maxCats1000/p0.5_q0.1
#CAT_LIST=/home/clark.3664/projects/cg_induction/categories/p_q/no_scores/maxCats1500/p0.5_q0.1
#CAT_LIST=/home/clark.3664/projects/cg_induction/categories/p_q/no_scores/maxCats2500/p0.5_q0.01
#CAT_LIST=/home/clark.3664/projects/cg_induction/categories/p_q/no_scores/maxCats100/p0.5_q0.1
#CAT_LIST=/home/clark.3664/projects/cg_induction/categories/grid_search/just_cat/bc2_nbc2_mc11

