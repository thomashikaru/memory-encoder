from lib2to3.pgen2.tokenize import StopTokenizing
from os import stat, stat_result, sysconf_names
from turtle import ycor
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
    plt.savefig(f"img/{prefix}{x}", dpi=100)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="test_output.csv")
    parser.add_argument("--out_prefix", default="")
    args = parser.parse_args()

    df_stim = pd.read_csv("embeddings/word_stimuli.csv")
    meaning_dict = dict(zip(df_stim["word_lower"], df_stim["# meanings (human)"]))
    synonym_dict = dict(zip(df_stim["word_lower"], df_stim["# synonyms (human)"]))
    freq_dict = dict(zip(df_stim["word_lower"], df_stim["Log Subtlex frequency"]))
    concreteness_dict = dict(zip(df_stim["word_lower"], df_stim["Concreteness"]))
    familiarity_dict = dict(zip(df_stim["word_lower"], df_stim["Familiarity"]))
    distinctiveness_dict = dict(
        zip(df_stim["word_lower"], df_stim["GloVe distinctiveness"])
    )

    df = pd.read_csv(args.csv)

    df["num_meanings"] = df["word"].replace(meaning_dict)
    df["num_synonyms"] = df["word"].replace(synonym_dict)
    df["frequency"] = df["word"].replace(freq_dict)
    df["concreteness"] = df["word"].replace(concreteness_dict)
    df["familiarity"] = df["word"].replace(familiarity_dict)
    df["distinctiveness"] = df["word"].replace(distinctiveness_dict)

    df["false_negative"] = ((df["label"] == 1) & (df["response"] == 0)).astype(int)
    df["false_positive"] = ((df["label"] == 0) & (df["response"] == 1)).astype(int)
    df["true_negative"] = ((df["label"] == 0) & (df["response"] == 0)).astype(int)
    df["true_positive"] = ((df["label"] == 1) & (df["response"] == 1)).astype(int)
    df["correct"] = (df["label"] == df["response"]).astype(int)

    df = df[df.label == 1]

    print(
        df[
            [
                "label",
                "false_positive",
                "false_negative",
                "true_positive",
                "true_negative",
            ]
        ].describe()
    )

    df1 = df.groupby("word").mean().reset_index()

    make_graph("num_meanings", "correct", args.out_prefix)
    make_graph("num_synonyms", "correct", args.out_prefix)
    make_graph("concreteness", "correct", args.out_prefix)
    make_graph("frequency", "correct", args.out_prefix)
    make_graph("familiarity", "correct", args.out_prefix)
    make_graph("distinctiveness", "correct", args.out_prefix)
