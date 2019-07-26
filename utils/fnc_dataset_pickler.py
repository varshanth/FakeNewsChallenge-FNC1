import pickle
import argparse
from csv import DictReader
import re
import nltk
from tqdm import tqdm

HEADLINE_MAX = 50
BODY_MAX = 500

# Function to read the csv line by line and each line will be treated
# as a dictionary element. The csv file must have the headers to be used as
# keys to the dictionary
def read_csv_into_rows(csv_file):
    rows = []
    with open(csv_file, 'r', encoding='utf-8') as table:
        content = DictReader(table)
        for line in content:
            # Each row is a dictionary object
            rows.append(line)
    return rows

def get_pos(tokens):
    pos_tagged_tokens = nltk.pos_tag(tokens)
    return pos_tagged_tokens

# Function used to filter the words in the title/headline, tokenize the words
# and return the desired number of tokens
def tokenize_and_limit(sentence, limit):
    sentence = re.sub('-', ' ', sentence)
    filtered_tokens = re.findall('[a-zA-Z]+|[0-9]+|[\!\?\.\',\"]+', sentence)
    tokenized = [tok.lower() for tok in filtered_tokens]
    if len(tokenized) < limit:
        tokenized += ['<pad>'] * (limit - len(tokenized))
    tokenized = tokenized[:limit]
    return tokenized

def pos_tag_tokenize_and_pickle_fnc_dataset(stances_csv, bodies_csv, pkl_path):
    print('-------Started Pickling Dataset--------')
    print('Reading the stances and bodies')
    stances = read_csv_into_rows(stances_csv)
    article_rows = read_csv_into_rows(bodies_csv)
    articles = {article['Body ID']:article['articleBody'] for article in article_rows}
    print('Tokenizing and POS Tagging the datapoints')
    datapoints = [{
        'h' : get_pos(tokenize_and_limit(stance['Headline'], HEADLINE_MAX)),
        'b' : get_pos(tokenize_and_limit(articles[stance['Body ID']], BODY_MAX)),
        'y' : stance['Stance'],
        } for stance in tqdm(stances)]
    print('Dumping the output to pkl')
    with open(pkl_path, 'wb') as to_pkl_fp:
        pickle.dump(datapoints, to_pkl_fp)
    print('--------End Pickling Dataset---------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Tokenize FNC Dataset & Pickle Datapoints")
    parser.add_argument('-stances_csv', type=str, default=None, required=True,
            help='Path to the Stances CSV')
    parser.add_argument('-bodies_csv', type=str, default=None, required=True,
            help='Path to the Bodies CSV')
    parser.add_argument('-pkl_path', type=str, default=None, required=True,
            help='Path to output the pickle file')
    args = parser.parse_args()
    pos_tag_tokenize_and_pickle_fnc_dataset(args.stances_csv,
            args.bodies_csv, args.pkl_path)


