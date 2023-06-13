from eval.evalb_unlabeled import eval_rvm_et_al, nont_heatmap
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("goldtrees")
    parser.add_argument("predtrees")
    args = parser.parse_args()

#    p, r, f, hom, rh = eval_rvm_et_al(["--gold", args.goldtrees, "--pred", args.predtrees])
#    print("precision:", p)
#    print("recall:", r)
#    print("hom:", hom)
#    print("RH:", rh)

    nont_heatmap(
        ["--gold", args.goldtrees,
         "--pred", args.predtrees]
    )
