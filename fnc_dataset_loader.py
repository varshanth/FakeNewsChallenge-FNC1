import re
import os
import random
import torch
from torchtext import data
from csv import DictReader

LEN_HEADLINE = 15
LEN_BODY = 60

TRAIN_PATH = {
        'stances': 'fnc-1/train_stances.csv',
        'bodies' : 'fnc-1/train_bodies.csv'
        }

TEST_PATH = {
        'stances' : 'fnc-1/competition_test_stances.csv',
        'bodies'  : 'fnc-1/competition_test_bodies.csv'
        }

def read_csv_into_rows(csv_file):
    rows = []
    with open(csv_file, 'r', encoding='utf-8') as table:
        content = DictReader(table)
        for line in content:
            # Each row is a dictionary object
            rows.append(line)
    return rows


def tokenize_and_limit(sentence, limit):
    sentence = re.sub('-', ' ', sentence)
    filtered_tokens = re.findall('[a-zA-Z]+|[0-9]+|[\!\?\.\',\"]+', sentence)
    tokenized = [tok.lower() for tok in filtered_tokens]
    if len(tokenized) < limit:
        tokenized += ['<pad>'] * (limit - len(tokenized))
    tokenized = tokenized[:limit]
    return tokenized


def get_FNC_1_fields():
    text_field = data.Field(batch_first = True)
    # LabelField sets sequential=False, is_target=True
    label_field = data.LabelField()
    # torch.float is req below since CosineEmbeddingLoss "y" accepts float
    condition_field = data.LabelField(use_vocab = False, dtype = torch.float)
    return {'text_field'   : text_field,
            'label_field'  : label_field,
            'condition_field': condition_field}


class FNC_1(data.Dataset):
    def __init__(self, train_flag, text_field, label_field, condition_field, condition, examples=None, **kwargs):
        """Create the FNC dataset instance given fields.
        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            condition_field: The field that will be used to indicate the conditioning label
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field), ('label', label_field), ('condition', condition_field)]

        if examples is None:
            stances_path = TRAIN_PATH['stances'] if train_flag else TEST_PATH['stances']
            bodies_path = TRAIN_PATH['bodies'] if train_flag else TEST_PATH['bodies']

            stances = read_csv_into_rows(stances_path)
            article_rows = read_csv_into_rows(bodies_path)
            articles = {article['Body ID']:article['articleBody'] for article in article_rows}

            datapoints = [{
                'h' : tokenize_and_limit(stance['Headline'], LEN_HEADLINE),
                'b' : tokenize_and_limit(articles[stance['Body ID']], LEN_BODY),
                'y' : stance['Stance'],
                'c' : -1. if stance['Stance'] == condition else 1.}
                for stance in stances]

            if condition != 'unrelated' and train_flag:
                print(condition)
                print('Filtering out unrelated datapoints')
                # Filter out the unrelated datapoints (unless it is the test set
                datapoints = list(filter(lambda dp: dp['y'] != 'unrelated', datapoints))

            examples = [data.Example.fromlist(
                [" ".join(datapoint['h']+datapoint['b']),
                datapoint['y'],
                datapoint['c']],
                fields) for datapoint in datapoints]

        super(FNC_1, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, train_flag, text_field, label_field, condition_field, condition, dev_ratio=.07, shuffle=True, **kwargs):
        """Create dataset objects for splits of the FNC_1 dataset.
        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            condition_field: The field that will be used as the conditioning label
            condition: Which label to use to condition the net
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        examples = cls(True, text_field, label_field, condition_field, condition, **kwargs).examples
        if shuffle: random.shuffle(examples)
        dev_index = -1 * int(dev_ratio*len(examples))

        return (cls(True, text_field, label_field, condition_field, condition, examples=examples[:dev_index]),
                cls(True, text_field, label_field, condition_field, condition, examples=examples[dev_index:]))
