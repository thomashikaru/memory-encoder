from os import sysconf_names
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="test_output.csv")
    parser.add_argument("--out_prefix", default="")
    args = parser.parse_args()

    df_stim = pd.read_csv("embeddings/word_stimuli.csv")
    meaning_dict = dict(zip(df_stim["word_lower"], df_stim["# meanings (human)"]))
    synonym_dict = dict(zip(df_stim["word_lower"], df_stim["# synonyms (human)"]))

    df = pd.read_csv(args.csv)

    df["num_meanings"] = df["word"].replace(meaning_dict)
    df["num_synonyms"] = df["word"].replace(synonym_dict)
    df["false_negative"] = (df["label"] == 1) & (df["response"] == 0)
    df["false_positive"] = (df["label"] == 0) & (df["response"] == 1)
    df["true_negative"] = (df["label"] == 0) & (df["response"] == 0)
    df["true_positive"] = (df["label"] == 1) & (df["response"] == 1)
    df["correct"] = (df["label"] == df["response"]).astype(int)

    # print(df[df["label"] == 1])

    # print(df[df["label"] == 1].groupby("word").mean())

    df1 = df.groupby("word").mean().reset_index()
    print(df1)

    plt.clf()
    p = sns.regplot(
        data=df1, x="num_meanings", y="correct", x_bins=np.linspace(1, 3, 10)
    )
    plt.savefig(f"img/{args.out_prefix}num_meanings", dpi=100)

    plt.clf()
    p = sns.regplot(
        data=df1, x="num_synonyms", y="correct", x_bins=np.linspace(1, 4, 10)
    )
    plt.savefig(f"img/{args.out_prefix}num_synonyms", dpi=100)
