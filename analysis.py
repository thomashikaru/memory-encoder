import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from scipy import stats


def make_graph(x, y, prefix, scatter=True):
    plt.clf()
    g = sns.jointplot(data=df1, x=x, y=y, kind="hex", color="#4CB391",)
    r, p = stats.pearsonr(df1[x], df1[y])
    print(f"{x}: {r:.3e} {p:.3e}")
    sns.regplot(
        data=df1, x=x, y=y, ax=g.ax_joint, scatter=scatter, x_bins=10,
    )
    plt.savefig(f"img/{prefix}{x}", dpi=100, bbox_inches="tight")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="test_output.csv")
    parser.add_argument("--out_prefix", default="")
    args = parser.parse_args()

    # read in stimuli and create mappings from word to features
    df_stim = pd.read_csv("embeddings/word_stimuli.csv")
    meaning_dict = dict(zip(df_stim["word_lower"], df_stim["# meanings (human)"]))
    synonym_dict = dict(zip(df_stim["word_lower"], df_stim["# synonyms (human)"]))
    freq_dict = dict(zip(df_stim["word_lower"], df_stim["Log Subtlex frequency"]))
    concreteness_dict = dict(zip(df_stim["word_lower"], df_stim["Concreteness"]))
    familiarity_dict = dict(zip(df_stim["word_lower"], df_stim["Familiarity"]))
    distinctiveness_dict = dict(
        zip(df_stim["word_lower"], df_stim["GloVe distinctiveness"])
    )

    # read in test results
    df = pd.read_csv(args.csv)

    # apply predictors
    df["num_meanings"] = df["word"].replace(meaning_dict)
    df["num_synonyms"] = df["word"].replace(synonym_dict)
    df["frequency"] = df["word"].replace(freq_dict)
    df["concreteness"] = df["word"].replace(concreteness_dict)
    df["familiarity"] = df["word"].replace(familiarity_dict)
    df["distinctiveness"] = df["word"].replace(distinctiveness_dict)

    # calculate scores
    df["false_negative"] = ((df["label"] == 1) & (df["response"] == 0)).astype(int)
    df["false_positive"] = ((df["label"] == 0) & (df["response"] == 1)).astype(int)
    df["true_negative"] = ((df["label"] == 0) & (df["response"] == 0)).astype(int)
    df["true_positive"] = ((df["label"] == 1) & (df["response"] == 1)).astype(int)
    df["correct"] = (df["label"] == df["response"]).astype(int)

    # filter for only target repeats
    df = df[df.label == 1]

    cols = [
        "label",
        "false_positive",
        "false_negative",
        "true_positive",
        "true_negative",
    ]
    print(df[cols].describe())

    # get average scores per word
    df1 = df.groupby("word").mean().reset_index()

    # make graphs
    make_graph("num_meanings", "correct", args.out_prefix)
    make_graph("num_synonyms", "correct", args.out_prefix)
    make_graph("concreteness", "correct", args.out_prefix)
    make_graph("frequency", "correct", args.out_prefix)
    make_graph("familiarity", "correct", args.out_prefix)
    make_graph("distinctiveness", "correct", args.out_prefix)
