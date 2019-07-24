import os
import re
import nltk
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from tqdm import tqdm
from bpemb import BPEmb


_wnl = nltk.WordNetLemmatizer()
bpemb_en = BPEmb(lang="en", dim=50, vs=200000)

master_pos_tags = ['NN', 'VBG', 'RB']


# Helper Methods

def normalize_word(w):
    return _wnl.lemmatize(w).lower()

def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]

def get_tokenized_pos(s, pos_tags = master_pos_tags):
    return [word for (word, token) in nltk.pos_tag(get_tokenized_lemmas(s)) if token in pos_tags]

def get_tokenized_quotes(s):
    words_within_quotes = re.findall(r"(['\"])(.*?)\1", s)
    return [token for word in words_within_quotes for token in get_tokenized_lemmas(word[1])]

def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]

def get_tokenized_encoder(s):
    return [word.replace("▁","") for word in bpemb_en.encode(s)]

def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output

def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output

def append_chargrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
        if gram in text_body[:100]:
            grams_first_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features


def append_ngrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in ngrams(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    return features

def body_split_sentences(bodies):
    body_part1 = []
    body_part2 = []
    body_part3 = []
    body_part4 = []
    for body in tqdm(bodies):

        sentences = re.split(r'[.?!]\s*', body)
        split_size = int(len(sentences)/4)
        i=0
        body_part1.append(" ".join(sentences[i:i+split_size]))
        i += split_size
        body_part2.append(" ".join(sentences[i:i+split_size]))
        i += split_size
        body_part3.append(" ".join(sentences[i:i+split_size]))
        i += split_size
        body_part4.append(" ".join(sentences[i:i+split_size]))

    return body_part1,body_part2,body_part3,body_part4


def gen_or_load_feats(feat_fn, headlines, bodies, feature_file):
    if not os.path.isfile(feature_file):
        feats = feat_fn(headlines, bodies)
        np.save(feature_file, feats)

    return np.load(feature_file)


# Feature methods

def word_overlap_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_lemmas(clean_headline)
        clean_body = get_tokenized_lemmas(clean_body)
        features = [
            len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
        X.append(features)
    return X


def refuting_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in _refuting_words]
        X.append(features)
    return X

def polarity_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

    def calculate_polarity(text):
        tokens = get_tokenized_lemmas(text)
        return sum([t in _refuting_words for t in tokens]) % 2
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        features = []
        features.append(calculate_polarity(clean_headline))
        features.append(calculate_polarity(clean_body))
        X.append(features)
    return np.array(X)


def hand_features(headlines, bodies):

    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in clean(headline).split(" "):
            if headline_token in clean(body):
                bin_count += 1
            if headline_token in clean(body)[:255]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in remove_stopwords(clean(headline).split(" ")):
            if headline_token in clean(body):
                bin_count += 1
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def count_grams(headline, body):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph

        clean_body = clean(body)
        clean_headline = clean(headline)
        features = []
        features = append_chargrams(features, clean_headline, clean_body, 2)
        features = append_chargrams(features, clean_headline, clean_body, 8)
        features = append_chargrams(features, clean_headline, clean_body, 4)
        features = append_chargrams(features, clean_headline, clean_body, 16)
        features = append_ngrams(features, clean_headline, clean_body, 2)
        features = append_ngrams(features, clean_headline, clean_body, 3)
        features = append_ngrams(features, clean_headline, clean_body, 4)
        features = append_ngrams(features, clean_headline, clean_body, 5)
        features = append_ngrams(features, clean_headline, clean_body, 6)
        return features

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        X.append(binary_co_occurence(headline, body)
                 + binary_co_occurence_stops(headline, body)
                 + count_grams(headline, body))


    return X


# Custom Features

def word_overlap_quotes_features(headlines, bodies):

    """ Method to calculate the intersection over union of tokens in headline and
    body. Tokens for headline are extracted based on the text between single or double quotes. """
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_quotes(clean_headline)
        clean_body = get_tokenized_lemmas(clean_body)
        features = [
            len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
        X.append(features)
    return X

def word_overlap_pos_features(headlines, bodies):

    """ Method to calculate the intersection over union of tokens in headline and
    body extracted based on the specific parts of speech tagging. """
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_pos(clean_headline)
        clean_body = get_tokenized_pos(clean_body)
        features = [
            len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)) + 0.000001)]
        X.append(features)
    return X


def word_overlap_split_bodies_features(headlines, bodies):

    """ Method to calculate the intersection over union of tokens in headline and
    4 body parts consisting of equal number of sentences.
    It returns 4 features based on pair of headline and body parts. """
    body_part1,body_part2,body_part3,body_part4 = body_split_sentences(bodies)
    X1 = word_overlap_pos_features(headlines, body_part1)
    print ("\n\n X1 matrix size - \n")
    print(len(X1))

    X2 = word_overlap_pos_features(headlines, body_part2)
    print ("\n\n X2 matrix size - \n")
    print(len(X2))

    X3 = word_overlap_pos_features(headlines, body_part3)
    print ("\n\n X3 matrix size - \n")
    print(len(X3))

    X4 = word_overlap_pos_features(headlines, body_part4)
    print ("\n\n X4 matrix size - \n")
    print(len(X4))

    return np.c_[X1, X2, X3, X4]

def word_tfidf_features(headlines, bodies):

    """ Method to calculate cosine similarity between the tfidf vectors for headline and body.
    Vocab size is for TFIDF Vectorizer is calculated by taking 5000 most occurring words in headline and body.
    It returns a single feature which is the cosine similarity value for a headline and body vector."""
    total_vocab = [get_tokenized_pos(clean(line)) for line in tqdm(headlines+bodies)]
    print ("\n\n total vocab size - \n")
    print(len(total_vocab))

    total_vocab_flatten = [word for subword in total_vocab for word in subword]
    word_counter = Counter(total_vocab_flatten)
    most_occur = word_counter.most_common(5000)

    vocab = [wd for wd,count in most_occur]
    print ("\n\n extracted vocab size - \n")
    print(len(vocab))

    tfidf_vectorizer = TfidfVectorizer(use_idf=True, vocabulary=vocab, analyzer='word', tokenizer=get_tokenized_lemmas)

    headlines_tfidf = tfidf_vectorizer.fit_transform(headlines)
    headlines_matrix = headlines_tfidf.toarray()
    print ("\n\n headline matrix size - \n")
    print(headlines_matrix.shape)

    bodies_tfidf = tfidf_vectorizer.fit_transform(bodies)
    bodies_matrix = bodies_tfidf.toarray()
    print ("\n\n body matrix size - \n")
    print(bodies_matrix.shape)

    similarity_df = cosine_similarity(headlines_matrix, bodies_matrix)
    X = np.diagonal(similarity_df)

    return X

def word_tfidf_pos_ss_features(headlines, bodies):

    """ Method to calculate cosine similarity between the tfidf vectors for headline and body.
    Method splits the body in 4 parts containing equal number of sentences.
    Vocab size is for TFIDF Vectorizer is calculated by taking 5000 most occurring words in headline and body.
    It returns a single feature which is the max cosine similarity value for a headline and body vector pair."""
    total_vocab = [get_tokenized_pos(clean(line)) for line in tqdm(headlines+bodies)]
    total_vocab_flatten = [word for subword in total_vocab for word in subword]

    word_counter = Counter(total_vocab_flatten)
    most_occur = word_counter.most_common(5000)
    vocab = [wd for wd,count in most_occur]

    tfidf_vectorizer = TfidfVectorizer(use_idf=True, vocabulary=vocab, analyzer='word', tokenizer=get_tokenized_lemmas)
    headlines_tfidf = tfidf_vectorizer.fit_transform(headlines)
    headlines_matrix = headlines_tfidf.toarray()
    print ("\n\n headline matrix size - \n")
    print(headlines_matrix.shape)

    body_part1,body_part2,body_part3,body_part4 = body_split_sentences(bodies)

    body_part1_tfidf = tfidf_vectorizer.fit_transform(body_part1)
    body_part1_matrix = body_part1_tfidf.toarray()
    print ("\n\n body 1 matrix size "+ str(len(body_part1)) +" - \n")
    print(body_part1_matrix.shape)

    body_part2_tfidf = tfidf_vectorizer.fit_transform(body_part2)
    body_part2_matrix = body_part2_tfidf.toarray()
    print ("\n\n body 2 matrix size "+ str(len(body_part2)) +" - \n")
    print(body_part2_matrix.shape)

    body_part3_tfidf = tfidf_vectorizer.fit_transform(body_part3)
    body_part3_matrix = body_part3_tfidf.toarray()
    print ("\n\n body 3 matrix size "+ str(len(body_part3)) +" - \n")
    print(body_part3_matrix.shape)

    body_part4_tfidf = tfidf_vectorizer.fit_transform(body_part4)
    body_part4_matrix = body_part4_tfidf.toarray()
    print ("\n\n body 4 matrix size "+ str(len(body_part4)) +" -\n")
    print(body_part4_matrix.shape)

    similarity_df1 = cosine_similarity(headlines_matrix, body_part1_matrix)
    X1 = np.diagonal(similarity_df1)

    similarity_df2 = cosine_similarity(headlines_matrix, body_part2_matrix)
    X2 = np.diagonal(similarity_df2)

    similarity_df3 = cosine_similarity(headlines_matrix, body_part3_matrix)
    X3 = np.diagonal(similarity_df3)

    similarity_df4 = cosine_similarity(headlines_matrix, body_part4_matrix)
    X4 = np.diagonal(similarity_df4)

    X =  [max(b1,b2,b3,b4) for b1,b2,b3,b4 in zip(X1,X2,X3,X4)]
    print ("\n\n X matrix size - \n")
    print(len(X))
    return X

def word_overlap_bpe_features(headlines, bodies):
    """ Method to calculate intersection over union of tokens encoded using byte-pair encoding library. """
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = " ".join(get_tokenized_pos(clean_headline))
        clean_body = " ".join(get_tokenized_pos(clean_body))
        clean_headline = get_tokenized_encoder(clean_headline)
        clean_body = get_tokenized_encoder(clean_body)
        features = [
            len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)) + 0.00001)]
        X.append(features)
    return X