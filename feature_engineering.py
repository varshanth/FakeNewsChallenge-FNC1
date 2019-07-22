import os
import re
import nltk
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from tqdm import tqdm


_wnl = nltk.WordNetLemmatizer()
# master_words = [ 'CD','JJ', 'JJR', 'JJS',
#              'NN', 'NNS', 'NNP', 'NNPS',
#              'RB', 'RBR', 'RBS', 'VB',
#              'VBG','VBD', 'VBP', 'VBZ']

master_words = ['NN', 'VBG', 'RB']


def normalize_word(w):
    return _wnl.lemmatize(w).lower()


def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]

def get_tokenized_pos(s):
    return [word for (word, token) in nltk.pos_tag(get_tokenized_lemmas(s)) if token in master_words]

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

def word_tfidf_bpe_features(headlines, bodies):

    # total_vocab = [get_tokenized_pos(clean(line)) for line in tqdm(headlines+bodies)]
    total_vocab = [word.replace("▁","") for line in tqdm(headlines+bodies) for word in bpemb_en.encode(line)]
    # total_vocab_flatten = [word for subword in total_vocab for word in subword]
    print ("\n\n Total Vocab size - \n")
    print(len(total_vocab))

    word_counter = Counter(total_vocab)
    most_occur = word_counter.most_common(8000)
    vocab = [wd for wd,count in most_occur]

    tfidf_vectorizer = TfidfVectorizer(use_idf=True, vocabulary=vocab, analyzer='word', tokenizer=get_tokenized_encoder)
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


def word_tfidf_bodies(bodies):
    body_part1 = []
    body_part2 = []
    body_part3 = []
    body_part4 = []
    for body in tqdm(bodies):
        split_size = int(len(body)/4)
        i=0
        body_part1.append(body[i:i+split_size])
        i += split_size
        body_part2.append(body[i:i+split_size])
        i += split_size
        body_part3.append(body[i:i+split_size])
        i += split_size
        body_part4.append(body[i:i+split_size])

    return body_part1,body_part2,body_part3,body_part4

def word_overlap_split_bodies_features(headlines, bodies):

    body_part1,body_part2,body_part3,body_part4 = word_tfidf_bodies(bodies)
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




def gen_or_load_feats(feat_fn, headlines, bodies, feature_file):
    if not os.path.isfile(feature_file):
        feats = feat_fn(headlines, bodies)
        np.save(feature_file, feats)

    return np.load(feature_file)




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


def word_overlap_pos_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_pos(clean_headline)
        clean_body = get_tokenized_pos(clean_body)
        features = [
            len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
        X.append(features)
    return X

def word_overlap_quotes_features(headlines, bodies):
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

def word_tfidf_features(headlines, bodies):

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
