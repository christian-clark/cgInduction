from eval.evalb_unlabeled import eval_rvm_et_al
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("goldtrees")
    parser.add_argument("predtrees")
    args = parser.parse_args()

#    run_analysis(["--type", "hm",
#                  "--gold", args.goldtrees,
#                  "--pred", args.predtrees,
#                  "--save", args.outfile]
#    )

    p, r, f, hom, rh = eval_rvm_et_al(["--gold", args.goldtrees, "--pred", args.predtrees])

    print("precision:", p)
    print("recall:", r)
    print("hom:", hom)
    print("RH:", rh)

