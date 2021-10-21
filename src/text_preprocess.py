import re

import numpy as np
import torchtext
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder
from spacy.lang.en import English


class TFIDFTokenize(TransformerMixin):
    """
    TF-IDF tokenization of both title and description fields and combine both the fields before feed to model.
    """
    def __init__(self):
        print("TF-IDF Tokenizer")
        self.tfidf = TfidfVectorizer(sublinear_tf=True,
                                     min_df=5, norm='l2',
                                     encoding='utf-8',
                                     ngram_range=(1, 3),
                                     stop_words='english')
        self.label_encoder = LabelEncoder()
        self.column_transform = ColumnTransformer(
                                    [('tfidf_title', self.tfidf, 'title'),
                                        ('tfidf_description', self.tfidf, 'description')])

    def fit(self, X, y=None):
        self.column_transform.fit(X)
        if y is not None:
            self.label_encoder.fit(y)

    def fit_transform(self, X, y=None):
        X = self.column_transform.fit_transform(X)
        if y is not None:
            y = self.label_encoder.fit_transform(y)
            return X, y
        return X, None

    def transform(self, X, y=None):
        X = self.column_transform.transform(X)
        if y is not None:
            y = self.label_encoder.transform(y)
            return X, y
        return X, None


class MeanEmbeddingTransformer(TransformerMixin):
    """
    Using Glove embedding from torchtext library.
    """
    def __init__(self):
        self._E = torchtext.vocab.GloVe(name="6B",  # trained on Wikipedia 2014 corpus of 6 billion words
                                        dim=50)  # embedding size = 100

    def _doc_mean(self, doc):
        return np.mean(np.array([self._E[w].numpy() for w in doc]), axis=0)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._doc_mean(doc) for doc in X])

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class TextPreprocess:
    """
    Text pre-processing steps applied before using Glove embedding. Following are the pre-processing steps applied

    1. Tokenization
    2. Stopwords removal
    3. Removing digits
    4. Removing punctuations
    """
    def __init__(self):
        pass

    def _tokenize(self, text):

        nlp = English()
        tokenizer = nlp.tokenizer
        tokens = tokenizer(text)
        tokens_list = []
        for token in tokens:
            if (not token.is_stop) and \
                    (not token.is_punct) and \
                    (not token.is_digit) and \
                    (len(token) > 3):
                tokens_list.append(token.lower_)
        return tokens_list

    def _punctuation(self, tokens):
        if isinstance((tokens), (str)):
            tokens = re.sub('<[^>]*>', '', tokens)
            tokens = re.sub('[\W]+', '', tokens.lower())
            return tokens
        if isinstance((tokens), (list)):
            return_list = []
            for i in range(len(tokens)):
                temp_text = re.sub('<[^>]*>', '', tokens[i])
                temp_text = re.sub('[\W]+', '', temp_text.lower())
                return_list.append(temp_text)
            return (return_list)
        else:
            pass

    def _pipelinize(self, function, active=True):
        def list_comprehend_a_function(list_or_series, active=True):
            if active:
                return [function(i) for i in list_or_series]
            else:  # if it's not active, just pass it right back
                return list_or_series

        return FunctionTransformer(list_comprehend_a_function, validate=False, kw_args={'active': active})

    def __call__(self, text):
        estimators = [('tokenizer', self._pipelinize(self._tokenize)),
                      ('puntuation', self._pipelinize(self._punctuation))]
        pipe = Pipeline(estimators)
        return pipe.transform([text])[0]


def tokenize_and_transform(X, y=None):
    text_process = TextPreprocess()
    label_encoder = LabelEncoder()
    X = X[['title', 'description']].values
    title = X[:, 0]
    description = X[:, 1]
    title = [text_process(doc) for doc in title]
    description = [text_process(doc) for doc in description]
    met = MeanEmbeddingTransformer()
    test = met.fit_transform(title)
    test2 = met.fit_transform(description)
    print(test.shape)
    print(test2.shape)
    X_transform = np.concatenate([met.fit_transform(title),
                                  met.fit_transform(description)],
                                 axis=1)
    if y is not None:
        y = label_encoder.fit_transform(y)
        return X_transform, y
    return X_transform, None