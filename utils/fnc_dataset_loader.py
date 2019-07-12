import re
import os
import random
import torch
from torchtext import data
from csv import DictReader
import sys
# The following will tell python to look for packages in the above folder as well
sys.path.append('..')
from dl_approach_cfg import TRAIN_PATH, TEST_PATH, LEN_HEADLINE, LEN_BODY

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

# Function to create & retrieve the data Fields used for the FNC_1 dataset
def get_FNC_1_fields():
    headline_field = data.Field(batch_first = True)
    body_field = data.Field(batch_first = True)
    # LabelField sets sequential=False, is_target=True
    label_field = data.LabelField()
    # torch.float is req below since CosineEmbeddingLoss "y" accepts float
    condition_field = data.LabelField(use_vocab = False, dtype = torch.float)
    return {'headline_field'   : headline_field,
            'body_field' : body_field,
            'label_field'  : label_field,
            'condition_field': condition_field}

### MAIN DATASET USED FOR THE MODEL ##
class FNC_1(data.Dataset):
    def __init__(self, train_flag,
            headline_field, body_field, label_field, condition_field,
            condition, examples=None, **kwargs):
        """Create the conditioned FNC dataset instance given fields.
        Arguments:
            headline_field: The field that will be used for headline data.
            body_field: The field that will be used for body data.
            label_field: The field that will be used for label data.
            condition_field: The field that will be used to indicate the conditioning label
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('headline', headline_field),
                  ('body', body_field),
                  ('label', label_field),
                  ('condition', condition_field)]

        if examples is None:
            # When the dataset creation is called standalone without the datapoints
            # being rendered
            stances_path = TRAIN_PATH['stances'] if train_flag else TEST_PATH['stances']
            bodies_path = TRAIN_PATH['bodies'] if train_flag else TEST_PATH['bodies']

            stances = read_csv_into_rows(stances_path)
            article_rows = read_csv_into_rows(bodies_path)
            articles = {article['Body ID']:article['articleBody'] for article in article_rows}

            datapoints = [{
                'h' : tokenize_and_limit(stance['Headline'], LEN_HEADLINE),
                'b' : tokenize_and_limit(articles[stance['Body ID']], LEN_BODY),
                'y' : stance['Stance'],
                # Conditioning on the field that other fields must be opposite to
                # This is used for the CosineEmbeddingLoss where the vectors representing
                # opposing headlines and bodies will be trained to be orthogonal to each
                # other.
                'c' : -1. if stance['Stance'] == condition else 1.}
                for stance in stances]

            if condition != 'unrelated':
                print('Filtering out unrelated datapoints')
                # Filter out the unrelated datapoints. When we condition on fields other
                # than unrelated, unrelated datapoints will add noise to the conditioning
                datapoints = list(filter(lambda dp: dp['y'] != 'unrelated', datapoints))
            else:
                # When the conditioning is on unrelated, all the other datapoints apart
                # from unrelated datapoints are considered related
                for i in range(len(datapoints)):
                    if datapoints[i]['y'] != 'unrelated':
                        datapoints[i]['y'] = 'related'

            examples = [data.Example.fromlist([
                " ".join(datapoint['h']),
                " ".join(datapoint['b']),
                datapoint['y'],
                datapoint['c']],
                fields) for datapoint in datapoints]

        super(FNC_1, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, train_flag, headline_field, body_field, label_field,
        condition_field, condition, dev_ratio=.07, shuffle=True, **kwargs):
        """Create dataset objects for splits of the FNC_1 dataset.
        Arguments:
            headline_field: The field that will be used for the headline.
            body_field: The field that will be used for the body.
            label_field: The field that will be used for label data.
            condition_field: The field that will be used as the conditioning label
            condition: Which label to use to condition the net
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        examples = cls(True, headline_field, body_field, label_field, condition_field,
            condition, **kwargs).examples
        if shuffle: random.shuffle(examples)
        dev_index = -1 * int(dev_ratio*len(examples))

        # Create train & test splits by calling the original class method to create
        # instances, but this time with the data already in hand
        return (cls(True, headline_field, body_field, label_field, condition_field,
            condition, examples=examples[:dev_index]),
            cls(True, headline_field, body_field, label_field, condition_field,
                condition, examples=examples[dev_index:]))

# This class should be used when evaluating the entire test dataset (including unrelated)
# when the model is conditioned for 'disagree'. The unrelated datapoints will be hard
# coded to have the true label as 'discuss'. We use this dataset to basically just retain
# the datapoint indices of the non 'unrelated' labels within the test set. A disagree
# conditioned model would predict a 'related' label but obviously never an unrelated label
# for all the datapoints. Hence the predictions for 'unrelated' datapoints would be
# nonsensical/meaningless
# THIS DATASET SHOULD NOT BE USED FOR TRAINING
class FNC_1_TEST_Unrelated_Is_Discuss(data.Dataset):
    def __init__(self, headline_field, body_field, label_field, condition_field,
        examples=None, **kwargs):
        """Create the FNC TEST dataset instance given fields.
        Arguments:
            headline_field: The field that will be used for headline data.
            body_field: The field that will be used for body data.
            label_field: The field that will be used for label data.
            condition_field: The field that will be used to indicate the conditioning label
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [
            ('headline', headline_field),
            ('body', body_field),
            ('label', label_field),
            ('condition', condition_field)]

        if examples is None:
            stances_path = TEST_PATH['stances']
            bodies_path = TEST_PATH['bodies']

            stances = read_csv_into_rows(stances_path)
            article_rows = read_csv_into_rows(bodies_path)
            articles = {article['Body ID']:article['articleBody'] for article in article_rows}

            datapoints = [{
                'h' : tokenize_and_limit(stance['Headline'], LEN_HEADLINE),
                'b' : tokenize_and_limit(articles[stance['Body ID']], LEN_BODY),
                'y' : 'discuss' if stance['Stance'] == 'unrelated' else stance['Stance'],
                'c' : -1. if stance['Stance'] == 'disagree' else 1.}
                for stance in stances]

            examples = [data.Example.fromlist([
                " ".join(datapoint['h']),
                " ".join(datapoint['b']),
                datapoint['y'],
                datapoint['c']],
                fields) for datapoint in datapoints]

        super(FNC_1_TEST_Unrelated_Is_Discuss, self).__init__(examples, fields, **kwargs)


# This class is used to give the full untouched FNC_1 Traindataset. This class
# can be used traditionally for building the vocabulary for the model
class FNC_1_Train_Untouched(data.Dataset):
    def __init__(self, headline_field, body_field, label_field, condition_field,
        examples=None, **kwargs):
        """Create the FNC TRAIN dataset instance given fields.
        Arguments:
            headline_field: The field that will be used for headline data.
            body_field: The field that will be used for body data.
            label_field: The field that will be used for label data.
            ### condition field IGNORED
            condition_field: The field that will be used to indicate the conditioning label
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [
            ('headline', headline_field),
            ('body', body_field),
            ('label', label_field),
            ('condition', condition_field)]

        if examples is None:
            stances_path = TRAIN_PATH['stances']
            bodies_path = TRAIN_PATH['bodies']

            stances = read_csv_into_rows(stances_path)
            article_rows = read_csv_into_rows(bodies_path)
            articles = {article['Body ID']:article['articleBody'] for article in article_rows}

            datapoints = [{
                'h' : tokenize_and_limit(stance['Headline'], LEN_HEADLINE),
                'b' : tokenize_and_limit(articles[stance['Body ID']], LEN_BODY),
                'y' : stance['Stance'],
                'c' : None} # IGNORED
                for stance in stances]

            examples = [data.Example.fromlist([
                " ".join(datapoint['h']),
                " ".join(datapoint['b']),
                datapoint['y'],
                datapoint['c']],
                fields) for datapoint in datapoints]

        super(FNC_1_Train_Untouched, self).__init__(examples, fields, **kwargs)

