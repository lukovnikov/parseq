import pandas as pd
import csv
import io
import numpy as np
import fire


def run(p=""):
    df = pd.read_csv(p)
    print(df)
    print(df.columns)

    # replace mcdX with mcd
    repld = {"cfq/mcd1": "cfq/mcd", "cfq/mcd2": "cfq/mcd", "cfq/mcd3": "cfq/mcd",
             "scan/mcd1": "scan/mcd", "scan/mcd2": "scan/mcd", "scan/mcd3": "scan/mcd"}
    df["dataset"] = df["dataset"].replace(repld)

    groupbycols = [x for x in df.columns if not (x in ["gpu", "seed", "Sweep", "Runtime", "Created", "Name", "State", "Nodes", "User", "Tags"] or x.endswith("acc") or x.endswith("entropy") or x.endswith("_aucroc") or x.endswith("_aucpr") or "_fpr" in x or x.endswith("_loss") or x.endswith("nll") or x.endswith("_loss_gru") or x.endswith("_acc_gru"))]
    print(groupbycols)
    # groupbycols = ["mcdropout", "dropout", "batsize", "dataset", "lr", "enclrmul", "gpu", "hdim", "gradnorm", "maxsize", "mode", "numheads", "numlayers", "patience", "smoothing", "warmup", "worddropout"]
    aggdf = df.groupby(groupbycols).aggregate([np.mean, np.std])
    aggdf = aggdf.reset_index()
    print(aggdf)
    print(aggdf.columns)

    print(aggdf.sort_values("dataset"))
    print(aggdf)

    # # drop cols which have all the same values:
    # nunique = aggdf.apply(pd.Series.nunique)
    # cols_to_drop = nunique[nunique == 1].index
    # aggdf = aggdf.drop(cols_to_drop, axis=1)

    # print("unique datasets")
    print()
    print(" SCAN RESULTS ")
    print()
    for mcdrop in aggdf["mcdropout"].unique():
        for method in ["decnll", "entropy", "maxnll", "sumnll"]:
            head = f"-mcdropout:{mcdrop}-method:{method}"
            values = []
            try:
                for dataset in ["scan/add_jump", "scan/add_turn_left", "scan/length", "scan/mcd"]:
                    for metric in ["aucroc", "aucpr", "fpr90"]:
                        colname = (f"{method}_{metric}", "mean")
                        row = aggdf[(aggdf["mcdropout"] == mcdrop) & (aggdf["dataset"] == dataset)]
                        value = row[colname]
                        assert(len(value) == 1)
                        values.append(f"{float(value)*100:.1f}")
                print(head)
                print(" & " + " & ".join(values) + " \\\\")
            except KeyError as e:
                pass

    print()
    print(" CFQ RESULTS ")
    print()
    for mcdrop in aggdf["mcdropout"].unique():
        for method in ["decnll", "entropy", "maxnll", "sumnll"]:
            try:
                head = f"-mcdropout:{mcdrop}-method:{method}"
                values = []
                for dataset in ["cfq/mcd"]:
                    for metric in ["aucroc", "aucpr", "fpr90"]:
                        colname = (f"{method}_{metric}", "mean")
                        row = aggdf[(aggdf["mcdropout"] == mcdrop) & (aggdf["dataset"] == dataset)]
                        value = row[colname]
                        assert(len(value) == 1)
                        values.append(f"{float(value)*100:.1f}")
                print(head)
                print(" & " + " & ".join(values) + " \\\\")
            except KeyError as e:
                pass



if __name__ == '__main__':
    fire.Fire(run)