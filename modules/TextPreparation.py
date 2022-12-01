import numpy as np
import pandas as pd
from modules.TextCleaner import Cleaner
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn


class TextPreparation(Cleaner):
    def __init__(self):
        super().__init__()
    

    def split_data(self, data, split_ratio):

        # splits data in train and test sets #
        # returns train and test datasets #

        inds = np.array(range(0, data.shape[0]))
        np.random.shuffle(inds)
        train_inds = inds[0:int(np.floor(data.shape[0] * split_ratio))]
        test_inds = inds[int(np.floor(data.shape[0] * split_ratio)):]
        return data.loc[train_inds, :], data.loc[test_inds, :]
    

    def rebalance(self, data):

        # rebalance train set so equal number of classes #
        # return rebalanced train set #

        class_1, class_0 = data[data['sentiment'] == 1], data[data['sentiment'] == 0]
        diff = len(class_1) - len(class_0)
        if diff > 0:
            extra_samples = class_0.sample(diff, replace=True)
        else:
            extra_samples = class_1.sample(abs(diff), replace=True)
        rebalanced_data = pd.concat([data, extra_samples], axis=0).reset_index()
        
        return rebalanced_data


    def create_word_embeddings(self, data, emb_size, w_size, min_count, save):
        
        # use gensim to create word embeddings #
        # returns keyed array of word embeddings #

        data['sentences'] = data['text'].apply(lambda x, save = False: self.get_sentences(x, save)[0][0:-1])
        sentences = []
        for i in range(data.shape[0]):
            text = data.iloc[i, -1]
            for sentence in text:
                if len(sentence) > 1:
                    sentences.append(sentence)
        print(f'num sentences: {len(sentences)}')
        model = Word2Vec(sentences=sentences,
                        vector_size=emb_size,
                        window=w_size,
                        min_count=min_count)
        embeddings = model.wv
        if save == True:
            with open('data/embeddings', 'wb') as fp:
                pickle.dump(embeddings, fp)
        return embeddings
    

    def create_pos_encodings(self, emb_size):

        # create a positional encoding array #
        # returns posiitonal encoding array #

        position = torch.arange(5000).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * (-np.log(10000.0) / emb_size))
        positions = torch.zeros(1, 5000, emb_size)
        positions[0, :, 0::2] = torch.sin(position * div_term)
        positions[0, :, 1::2] = torch.cos(position * div_term)
        
        return positions


    def vectorise_texts(self, text, embeddings, tokens_len):

        # converts ngrams into embeddings #
        # returns the text as a 2d array #
        
        vectorised_text = np.zeros((tokens_len, (len(embeddings[0]))))
        ngram_counter = 0
        for ngram in text:
            if ngram_counter == tokens_len:
                break
            if ngram in embeddings:
                vectorised_text[ngram_counter, :] = embeddings[ngram]
                ngram_counter += 1
        return vectorised_text
    

    def compress_texts(self, text):

        # compresses texts into one vector for standard NN training #
        # returns text as a 1D vector #

        summed_ngrams = np.sum(text, axis=0)
        summed_encodings = np.sum(text, axis=1)

        return summed_ngrams / len(summed_encodings[summed_encodings != 0])