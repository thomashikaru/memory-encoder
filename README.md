# Word Memorability Autoencoder Model

## Steps

1. Get a corpus of English words
1. Calculate synonymy and polysemy scores
1. Get word embeddings
    1. word2vec
    1. GloVe
1. Train autoencoder model
    1. 1 epoch has N word embeddings presented sequentially
    1. Run T epochs
    1. Loss function: reconstruction loss (MSE between input and output)
    1. Backpropagate through autoencoder parameters
1. Evaluate model
    1. Test on words in Mahowald dataset
    1. Get recognition accuracy for repeat test words
    1. Look for evidence of synonymy and polysemy effects