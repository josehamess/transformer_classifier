# transformer_classifier
Testing a transformer against a standard fully connected neural network for classifying yelp reviews.

Over 40,000 yelp reviews were used to train the models and 8000 to test. Gensim Word2Vec model was used to create the initial uncontextualised word
embeddings and vectorise the texts.

Embeddings size used: 90 (due to lack of computational power)
Token lengths used: 150

For the FCNN (fully connected neural network) the texts were compressed into 1D vectors by simple averaging over tokens in the text.

Accuracy of Transformer was 0.92 compared to 0.85 for the FCNN, demostrating the power of the transformer. However, the computational power
needed to train the Transformer vastly outweighs the 7% improvement in classification. This is because the transformer has over 400,000 trainable parameters compared to around 60,000 for the FCNN. Without the use of a GPU this training would take far too long.
