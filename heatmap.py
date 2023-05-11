#from eval.evalb_unlabeled_heatmap import run_analysis, nont_heatmap
from eval.evalb_unlabeled import nont_heatmap
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("goldtrees")
    parser.add_argument("predtrees")
    parser.add_argument("outfile")
    args = parser.parse_args()

#    run_analysis(["--type", "hm",
#                  "--gold", args.goldtrees,
#                  "--pred", args.predtrees,
#                  "--save", args.outfile]
#    )

    nont_heatmap(["--gold", args.goldtrees,
                  "--pred", args.predtrees]
    )
