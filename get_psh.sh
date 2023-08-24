echo "step 1"
python -c "from eval.evalb_unlabeled import eval_rvm_et_al; eval_rvm_et_al(['-g/home/clark.3664/projects/cg_induction/data/adam.entire.senttrees', '-p/home/clark.3664/projects/cg_induction/outputs/adam/emnlp/gridsearch_bc16_nbc16_mc80_0/e5.vittrees.gz'])" >> gold_pred_cc.txt
echo "step 2"
python get_psh.py gold_pred_cc.txt >> cc_psh.txt
