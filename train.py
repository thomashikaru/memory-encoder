import torch
from encoder import MemoryEncoder, MemoryVAE
import matplotlib.pyplot as plt
import pandas as pd
import utils
import itertools
from tqdm import tqdm
import logging
import gensim, gensim.downloader
import numpy as np
import seaborn as sns
import random
import argparse
from sklearn.model_selection import train_test_split
from collections import defaultdict
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.metrics import matthews_corrcoef as mcc


class MemoryModel:
    def __init__(self, dims, vae=False, noise=0.01, K=0.4) -> None:
        if vae:
            self.ae = MemoryVAE(dimensionality=dims)
        else:
            self.ae = MemoryEncoder(dimensionality=dims)
        self.history = set()
        self.history_latent = []
        self.noise = noise
        self.K = K

    def reset(self):
        model.history = set()
        model.history_latent = []

    def test_word_basic(self, word, input):
        """simulates presentation of a single word stimulus using basic strategy
        - the word is passed through the autoencoder, yielding a reconstruction
        - the closest word in embedding space to the reconstruction is found
        - this is stored in the memory buffer
        - model reports a repeat if the current word is already present in the buffer

        Args:
            word (str): current word stimulus
            input (tensor): tensor corresponding to word embedding of word

        Returns:
            Tuple[str, boolean]: closest word and boolean response (true if repeat)
        """
        latent, reconstructed = self.ae(input)

        closest = w2v.most_similar(
            positive=[reconstructed.detach().numpy()], negative=[], topn=1
        )[0][0]

        is_repeat = word in self.history
        self.history.add(closest)

        return closest, is_repeat

    def test_word_latent(self, word, input):
        """simulates presentation of a single word stimulus using latent strategy
        - the word is passed through the autoencoder, yielding latent and reconstructed vectors
        - random noise is added to the content of the memory buffer
        - the contents of the memory buffer are passed through the decoder
        - cosine similarity of input to vectors decoded from noisy latent vectors
        - the latent vector is added to the memory buffer


        Args:
            word (str): input word
            input (tensor): input word embedding

        Returns:
            Tuple[str, boolean]: empty and boolean response (true if repeat)
        """
        latent, reconstructed = self.ae(input)
        latent = latent.detach().numpy()

        closest = ""

        if len(self.history_latent) == 0:
            self.history_latent.append(latent)
            return closest, False

        buffer = np.array(self.history_latent)
        buffer += np.random.normal(0, self.noise, size=buffer.shape)
        self.latent_history = list(buffer)

        decoded = np.array([self.ae.decode(torch.Tensor(x)).numpy() for x in buffer])

        cos_sims = cosine_similarity(input.reshape(1, -1), decoded)

        is_repeat = cos_sims.max() > self.K

        self.history_latent.append(latent)

        return closest, is_repeat

    def save(self, filename):
        torch.save(self.ae, filename)

    def load(self, filename):
        self.ae = torch.load(filename)


def get_embeddings_gpt(
    model, tokenizer, sents, sentence_embedding, lower=True, verbose=True
):
    """
    :param sents: list of strings
    :param sentence_embedding: string denoting how to obtain sentence embedding

    hidden_states (in Transformers==4.1.0) is a 3D tensor of dims: [batch, tokens, emb size]

    Compute activations of hidden units
    Returns dict with key = layer, item = 2D array of stimuli x units
    """

    model.eval()  # does not make a difference
    n_layer = model.config.n_layer
    max_n_tokens = model.config.n_positions
    states_sentences = defaultdict(list)
    if verbose:
        print(f"Computing activations for {len(sents)} sentences")

    for count, sent in enumerate(sents):
        if lower:
            sent = sent.lower()
        input_ids = torch.tensor(tokenizer.encode(sent))

        start = max(0, len(input_ids) - max_n_tokens)
        if start > 0:
            logging.warn(f"Stimulus too long! Truncated the first {start} tokens")
        input_ids = input_ids[start:]
        result_model = model(input_ids, output_hidden_states=True, return_dict=True)
        hidden_states = result_model["hidden_states"]

        #### NOTE:
        # hidden_state[n_layer][word][emb for token]
        for i in range(n_layer):  # for each layer
            dim = 0
            state = None
            hidden_state_shape = hidden_states[i].shape
            if hidden_state_shape[1] <= 1:
                state = hidden_states[i].squeeze().detach().numpy()
            elif sentence_embedding == "last-tok":  # last token
                state = hidden_states[i].squeeze()[-1, :].detach().numpy()
            elif sentence_embedding == "avg-tok":  # mean over tokens
                state = torch.mean(hidden_states[i].squeeze(), dim=dim).detach().numpy()
            elif sentence_embedding == "sum-tok":  # sum over tokens
                state = torch.sum(hidden_states[i].squeeze(), dim=dim).detach().numpy()
            else:
                print("Sentence embedding method not implemented")
            states_sentences[i].append(np.array(state))
    return states_sentences


def get_embeddings_word2vec(words):
    a = []
    oov_words = []
    for word in words:
        try:
            vec = w2v[str(word)]
            a.append(vec)
        except KeyError:
            a.append(np.full(300, 0))
            oov_words.append(word)
    return np.array(a), oov_words


# glove
def read_glove_vectors_from_file(filename):
    d = {}
    with open(filename, "rt") as f:
        for line in f:
            word, *rest = line.split()
            d[word] = np.array(list(map(float, rest)))
    return d


def get_embeddings_glove(words):
    a = []
    oov_words = []
    for word in words:
        try:
            vec = glove[word]
            a.append(vec)
        except KeyError:
            a.append(np.full(300, 0))
            oov_words.append(word)
    return np.array(a), oov_words


# create a sequence of training stimuli containing some repeats
def create_train_seq(df, repeat_freq=0.3, length=1000):
    df = df.sample(n=length)

    words = list(df["word_lower"])
    embeddings = list(df["embedding"])
    repeat = [0] * len(words)

    for i in range(len(words)):
        if random.random() > repeat_freq:
            continue
        if repeat[i] == 1:
            continue
        idx = min(len(words) - 1, i + random.randint(5, 100))
        if repeat[idx] == 1:
            continue
        words[idx] = words[i]
        embeddings[idx] = embeddings[i]
        repeat[idx] = 1
    return words, embeddings, repeat


def train(model, df, args):

    epoch_losses = []
    loss_function1 = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.ae.parameters(), lr=args.lr)

    for epoch in tqdm(range(args.epochs)):

        losses = []
        # model.memory.reset_parameters()

        words, embeddings, labels = create_train_seq(df)

        for word, word_embedding, label in zip(words, embeddings, labels):

            optimizer.zero_grad()

            input = torch.Tensor(word_embedding)
            logging.debug(input.size())
            logging.debug(input)

            # Output of Autoencoder
            latent, decoded = model.ae(input)
            logging.debug(decoded.size())
            logging.debug(decoded[:10])

            # Calculating the loss function
            loss = loss_function1(decoded, input)

            loss.backward()
            optimizer.step()

            # Storing the losses in a list for plotting
            losses.append(loss.detach().numpy())

        epoch_loss = np.mean(losses)
        logging.info(f"Epoch {epoch} Loss: {epoch_loss}")
        epoch_losses.append(epoch_loss)

    p = sns.lineplot(x=list(range(len(epoch_losses))), y=epoch_losses)
    p.set_xlabel("Epoch")
    p.set_ylabel("Loss")
    plt.savefig("losses", dpi=100)


def train_vae(model, df, args):
    optimizer = torch.optim.Adam(model.ae.parameters(), lr=args.lr)

    epoch_losses = []

    for epoch in tqdm(range(args.epochs)):

        losses = []
        words, embeddings, labels = create_train_seq(df)

        for word, word_embedding, label in zip(words, embeddings, labels):
            input = torch.Tensor(word_embedding)
            optimizer.zero_grad()
            _, x_hat = model.ae(input)
            loss = ((input - x_hat) ** 2).sum() + model.ae.kl
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().numpy())

        epoch_loss = np.mean(losses)
        logging.info(f"Epoch {epoch} Loss: {epoch_loss}")
        epoch_losses.append(epoch_loss)

    identifier = f"lr{args.lr:.0e}_epochs{args.epochs}_inner64_vae{args.vae}_generic{args.train_generic}_{args.emb_method}"

    p = sns.lineplot(x=list(range(len(epoch_losses))), y=epoch_losses)
    p.set_xlabel("Epoch")
    p.set_ylabel("Loss")
    plt.savefig(f"img/{identifier}", dpi=100)


def test(model, df, use_latent):
    with torch.no_grad():
        logging.debug(
            f"{'Word':<15}{'Input':<18}{'Label':<8}{'Closest':<20}{'Repeat?':<8}"
        )
        output = defaultdict(list)
        model.reset()
        words, embeddings, labels = create_train_seq(df, repeat_freq=0.4)
        for word, word_embedding, label in zip(words, embeddings, labels):
            input = torch.Tensor(word_embedding)
            if use_latent:
                closest, is_repeat = model.test_word_latent(word, input)
            else:
                closest, is_repeat = model.test_word(word, input)
            logging.debug(
                f"{word:<15}{str(input[0]):<18}{label:<8}{closest:<20}{is_repeat:<8}"
            )
            output["word"].append(word)
            output["label"].append(label)
            output["closest"].append(closest)
            output["response"].append(int(is_repeat))

        output_df = pd.DataFrame(output)
        return output_df
        # output_df.to_csv("test_output.csv", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--test_runs", type=int, default=10)
    parser.add_argument("--vae", action="store_true")
    parser.add_argument("--train_generic", action="store_true")
    parser.add_argument("--use_latent", action="store_true")
    parser.add_argument("--dims", type=int, default=300)
    parser.add_argument("--emb_method", default="w2v")
    parser.add_argument("--generic_train_size", type=int, default=10000)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--K", type=float, default=0.4)
    args = parser.parse_args()

    # setup
    logging.basicConfig(level=logging.INFO)
    sns.set_style("dark")

    # conventional word embedding models

    # load experimental data
    df_test = pd.read_csv("embeddings/word_stimuli.csv")
    words = df_test["word_lower"]
    logging.info(f"Number of words: {len(words)}")

    if args.emb_method == "w2v":
        w2v = gensim.downloader.load("word2vec-google-news-300")
        word_embeddings, oov_words = get_embeddings_word2vec(words)
    elif args.emb_method == "glove":
        glove = read_glove_vectors_from_file(
            "../../sentence_memorability/embeddings/glove.42B.300d.txt"
        )
        word_embeddings, oov_words = get_embeddings_glove(words)

    logging.info(oov_words)
    df_test["embedding"] = list(word_embeddings)
    train_dfs = [df_test]

    # load training data
    if args.train_generic and not args.test_only:
        df_train = pd.read_csv(
            "embeddings/unigram_freq.csv", names=["word_lower", "freq"]
        ).iloc[: args.generic_train_size, :]
        words = df_train["word_lower"]
        logging.info(f"Number of words: {len(words)}")
        if args.emb_method == "w2v":
            word_embeddings, oov_words = get_embeddings_word2vec(words)
        elif args.emb_method == "glove":
            word_embeddings, oov_words = get_embeddings_glove(words)
        df_train["embedding"] = list(word_embeddings)
        train_dfs.append(df_train)

    # Model Initialization
    model = MemoryModel(dims=args.dims, vae=args.vae, noise=args.noise, K=args.K)
    logging.debug(model.ae)

    identifier = f"lr{args.lr:.0e}_epochs{args.epochs}_inner64_vae{args.vae}_generic{args.train_generic}_{args.emb_method}"
    checkpoint_name = f"checkpoints/{identifier}.pt"

    # train model
    if not args.test_only:
        if args.vae:
            train_vae(model, pd.concat(train_dfs), args)
        else:
            train(model, pd.concat(train_dfs), args)
        model.save(checkpoint_name)

    # load model from checkpoint
    model.load(checkpoint_name)

    # test model on test set
    logging.info("\n\nTesting on Test Set")

    output_dfs = []
    for i in range(args.test_runs):
        output_dfs.append(test(model, df_test, args.use_latent))
    output_df = pd.concat(output_dfs)
    print("MCC:", mcc(output_df["label"], output_df["response"]))
    output_df.to_csv(
        f"csv/{identifier}-latent{args.use_latent}.csv", index=False,
    )
