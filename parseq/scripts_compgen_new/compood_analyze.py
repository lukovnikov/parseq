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

    groupbycols = [x for x in df.columns if not (x in ["gpu", "seed", "Sweep", "Runtime", "Created", "Name", "State", "Nodes", "User", "Tags"] or x.endswith("acc") or x.endswith("entropy") or x.endswith("_aucroc") or x.endswith("_aucpr") or "_fpr" in x or x.endswith("_loss") or x.endswith("nll"))]
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
    for dataset in aggdf["dataset"].unique():
        print(f"Dataset: {dataset}")
        xdf = aggdf[aggdf["dataset"] == dataset]
        print(xdf)
    print(aggdf["valid_treeacc", "mean"])
    print(aggdf.iloc[0])


if __name__ == '__main__':
    fire.Fire(run)