# Word Memorability Autoencoder Model

## Steps

1. Get a corpus of English words
1. Calculate synonymy and polysemy scores
1. Get word embeddings
    1. word2vec
    1. GloVe
    1. BERT
    1. GPT-2
1. Train autoencoder model
    1. 1 epoch has N word embeddings presented sequentially
    1. After each epoch, wipe the memory but preserve encoder, decoder, and memory unit weights
    1. Run T epochs
    1. Loss function: joint reconstruction loss and binary cross entropy loss for recognition
    1. Backpropagate through autoencoder and memory unit parameters
    1. Model should learn to reconstruct the input while using the memory unit to recognize repeats
1. Evaluate model
    1. Test on both seen and unseen words (separately)
    1. Get recognition accuracy for test words
    1. Look for evidence of synonymy and polysemy effects