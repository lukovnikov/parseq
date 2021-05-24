import pandas as pd
import csv
import io
import numpy as np
import fire


def run(p=""):
    df = pd.read_csv(p)
    df = df.replace(np.nan, -999, regex=True)
    print(df)
    print(df.columns)

    # replace mcdX with mcd
    df = df[df["State"] == "finished"]

    averagecols = ['seed', 'gpu', 'Name', 'State', 'Notes', 'User', 'Tags', 'Created', 'Updated', 'Runtime', 'GPU Type', 'evaltrain', 'final_test_steps_used', 'final_test_tree_acc', 'final_train_CE', 'final_train_steps_used', 'final_train_tree_acc', 'final_valid_steps_used', 'final_valid_tree_acc', 'testcode', 'version', 'train_CE', 'train_stepsused', 'train_tree_acc', 'valid_stepsused', 'valid_tree_acc']

    groupbycols = [x for x in df.columns if not x in averagecols    #["gpu", "seed", "Sweep", "Runtime", "Created", "Name", "State", "Nodes", "User", "Tags"]
             ]
    print(groupbycols)
    # groupbycols = ["mcdropout", "dropout", "batsize", "dataset", "lr", "enclrmul", "gpu", "hdim", "gradnorm", "maxsize", "mode", "numheads", "numlayers", "patience", "smoothing", "warmup", "worddropout"]
    aggdf = df.groupby(groupbycols).aggregate([np.mean, np.std])
    aggdf = aggdf.reset_index()
    print(aggdf)
    print(aggdf.columns)

    # # drop cols which have all the same values:
    # nunique = aggdf.apply(pd.Series.nunique)
    # cols_to_drop = nunique[nunique == 1].index
    # aggdf = aggdf.drop(cols_to_drop, axis=1)

    # print("unique datasets")
    domains = ["calendar", "blocks", "housing", "restaurants", "publications", "recipes", "socialnetwork", "basketball"]
    domains2 = ["calendar", "restaurants", "publications", "recipes"]

    settings = set()
    for i in range(len(aggdf)):
        setting = tuple(sorted([(k, aggdf[k][i]) for k in aggdf.columns if k[0] not in averagecols and k[0] != "domain"]))
        settings.add(setting)

    seldf = aggdf
    for setting in settings:
        for k, v in setting:
            assert k[1] == ''
            seldf = seldf[seldf[k[0]] == v]
        means, stds = [], []
        steps_means, steps_stds = [], []
        for domain in domains:
            testtreeacc_mean = seldf[seldf["domain"]==domain]["final_test_tree_acc"]["mean"]
            testtreeacc_std = seldf[seldf["domain"] == domain]["final_test_tree_acc"]["std"]
            means.append(testtreeacc_mean.iloc[0] if len(testtreeacc_mean) == 1 else -1)
            stds.append(testtreeacc_std.iloc[0] if len(testtreeacc_std) == 1 else -1)
            teststeps_mean = seldf[seldf["domain"]==domain]["final_test_steps_used"]["mean"]
            teststeps_std = seldf[seldf["domain"] == domain]["final_test_steps_used"]["std"]
            steps_means.append(teststeps_mean.iloc[0] if len(teststeps_mean) == 1 else -1)
            steps_stds.append(teststeps_std.iloc[0] if len(teststeps_std) == 1 else -1)
        means.append(np.mean(means))
        means.append(np.mean(stds))

        steps_means.append(np.mean(teststeps_mean))

        print("RESULTS FOR:")
        print(setting)
        print('(a)')
        print(domains)
        line = ' & ' + ' & '.join([f"{m*100:.1f}" for m in means[:-1]]) + f' $\\pm$ {100*means[-1]:.1f}' + ' \\\\'
        print(line)


        means, stds = [], []
        steps_means, steps_stds = [], []
        for domain in domains2:
            testtreeacc_mean = seldf[seldf["domain"]==domain]["final_test_tree_acc"]["mean"]
            testtreeacc_std = seldf[seldf["domain"] == domain]["final_test_tree_acc"]["std"]
            means.append(testtreeacc_mean.iloc[0] if len(testtreeacc_mean) == 1 else -1)
            stds.append(testtreeacc_std.iloc[0] if len(testtreeacc_std) == 1 else -1)
            teststeps_mean = seldf[seldf["domain"]==domain]["final_test_steps_used"]["mean"]
            teststeps_std = seldf[seldf["domain"] == domain]["final_test_steps_used"]["std"]
            steps_means.append(teststeps_mean.iloc[0] if len(teststeps_mean) == 1 else -1)
            steps_stds.append(teststeps_std.iloc[0] if len(teststeps_std) == 1 else -1)
        means.append(np.mean(means))
        # means.append(np.mean(stds))

        steps_means.append(np.mean(teststeps_mean))

        print('(b)')
        print(domains2)
        line = ' & ' + ' & '.join([f"{m*100:.1f} & {s:.1f} ($\\times$ ? )" for m, s in zip(means, steps_means)]) + '\\\\'
        print(line)



        seldf = aggdf



if __name__ == '__main__':
    fire.Fire(run)