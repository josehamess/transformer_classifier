import regex as re
import pickle


class Cleaner():
    def __init__(self, *kwargs):
        self.kwargs = kwargs

    def formatting_cleaner(self, text):

        # removes format tokens from text #
        # returns the cleaned text #

        text = re.sub(r'\[[0-9]"\]', '', text)
        text = re.sub(r'<br>', '', text)
        text = re.sub(r'-\n', '', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\t', '', text)
        text = re.sub(r'\d', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text
    

    def decapitalise(self, text):

        # removes capital letters from text #
        # returns clean list of text #
        
        return text.lower()
    

    def punctuation_removal(self, text):

        # removes punctuation from text #
        # returns new list of text #

        punctuation = ['[', ']', '(', ')', '*', '%',
                        '$', '£', '@', ',', '/', '-',
                        '"', "'", '+', '=', ':', ';']
        for char in punctuation:
            text = text.replace(char, '')
        clean_text = re.sub(r'\s+', ' ', text)
    
        return clean_text
    

    def sentence_splitter(self, text):

        # splits text into sentences #
        # return list of sentences #

        return re.split(r'[.?!]', text)
    

    def stop_word_removal(self, sentences):

        # splits sentences into words and removes stop words #
        # returns list of clean sentences #

        stop_words = []

        cleaned_sentences = []
        ngram_counts = {}
        for sentence in sentences:
            split_sentence = sentence.split(' ')
            cleaned_sentence = []
            for ngram in split_sentence:
                if ngram not in stop_words and len(ngram) > 0:
                    cleaned_sentence.append(ngram)
                    if ngram in ngram_counts:
                        ngram_counts[ngram] += 1
                    else:
                        ngram_counts[ngram] = 1
            cleaned_sentences.append(cleaned_sentence)

        return cleaned_sentences, ngram_counts
    

    def remove_sentence_punctuation(self, text):

        # remove puctuation that hasnt been removed #
        # return cleaned text #

        text = text.replace('.', '')
        text = text.replace('?', '')
        text = text.replace('!', '')
        clean_text = re.sub(r'\s+', ' ', text)
        
        return clean_text
    

    def clean_text(self, text):

        # extracts clean text from text #
        # returns cleaned text #

        text = self.punctuation_removal(self.decapitalise(self.formatting_cleaner(text)))
        text, _ = self.stop_word_removal([self.remove_sentence_punctuation(text)])

        return text[0]


    def get_sentences(self, text, save):

        # extracts sentences from text #
        # returns cleaned sentences #

        text = self.punctuation_removal(self.decapitalise(self.formatting_cleaner(text)))
        sentences, ngram_counts = self.stop_word_removal(self.sentence_splitter(text))

        if save == True:
            with open('data/sentences', 'wb') as fp:
                pickle.dump(sentences, fp)

        return sentences, ngram_counts