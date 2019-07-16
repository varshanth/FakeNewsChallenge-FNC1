import re
import os
import random
import torch
from torchtext import data
from csv import DictReader
import pickle
import sys
import nltk
# The following will tell python to look for packages in the above folder as well
sys.path.append('..')
from dl_approach_cfg import TRAIN_PATH, TEST_PATH, LEN_HEADLINE, LEN_BODY

def get_pos_filtered_tokens(pos_tagged_tokens):
    accepted_pos = {
            'CD',
            'JJ', 'JJR', 'JJS',
            'NN', 'NNS', 'NNP', 'NNPS',
            'RB', 'RBR', 'RBS', 'VB',
            'VBG','VBD', 'VBP', 'VBZ',
            }
    filt_tokens = [tok_pos[0] for tok_pos in pos_tagged_tokens
                   if tok_pos[1] in accepted_pos]
    return filt_tokens

def filter_and_limit_tokens(pos_tagged_tokens, apply_pos_filter, limit):
    if apply_pos_filter:
        output_tokens = get_pos_filtered_tokens(pos_tagged_tokens)
    else:
        output_tokens = [tok_pos[0] for tok_pos in pos_tagged_tokens]
    if len(output_tokens) < limit:
        output_tokens += ['<pad>'] * (limit - len(output_tokens))
    output_tokens = output_tokens[:limit]
    return output_tokens

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
            condition, apply_pos_filter=False, examples=None, **kwargs):
        """Create the conditioned FNC dataset instance given fields.
        Arguments:
            headline_field: The field that will be used for headline data.
            body_field: The field that will be used for body data.
            label_field: The field that will be used for label data.
            condition_field: The field that will be used to indicate the conditioning label
            condition: Which label to use to condition the net
            apply_pos_filter: Apply POS filter on the tokens or not
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
            datapoints_path = TRAIN_PATH if train_flag else TEST_PATH

            with open(datapoints_path, 'rb') as pkl_fp:
                # datapoints = [{'h': [(headline_token, POS)*], 'b' : [(body_token, POS)*], 'y' : stance}*]
                datapoints = pickle.load(pkl_fp)

            if condition != 'unrelated':
                print('Filtering out unrelated datapoints')
                # Filter out the unrelated datapoints. When we condition on fields other
                # than unrelated, unrelated datapoints will add noise to the conditioning
                datapoints = list(filter(lambda dp: dp['y'] != 'unrelated', datapoints))
            else:
                # When the conditioning is on unrelated, all the other datapoints apart
                # from unrelated datapoints are considered related
                for datapoint in datapoints:
                    if datapoint['y'] != 'unrelated':
                        datapoint['y'] = 'related'

            for datapoint in datapoints:
                datapoint['h'] = filter_and_limit_tokens(
                        datapoint['h'], apply_pos_filter, LEN_HEADLINE)
                datapoint['b'] = filter_and_limit_tokens(
                        datapoint['b'], apply_pos_filter, LEN_BODY)
                # Conditioning on the field that other fields must be opposite to
                # This is used for the CosineEmbeddingLoss where the vectors representing
                # opposing headlines and bodies will be trained to be orthogonal to each
                # other.
                datapoint['c'] = -1. if datapoint['y'] == condition else 1.

            examples = [data.Example.fromlist([
                " ".join(datapoint['h']),
                " ".join(datapoint['b']),
                datapoint['y'],
                datapoint['c']],
                fields) for datapoint in datapoints]

        super(FNC_1, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, train_flag, headline_field, body_field, label_field,
        condition_field, condition, apply_pos_filter = False,
        dev_ratio=.008, shuffle=True, **kwargs):
        """Create dataset objects for splits of the FNC_1 dataset.
        Arguments:
            headline_field: The field that will be used for the headline.
            body_field: The field that will be used for the body.
            label_field: The field that will be used for label data.
            condition_field: The field that will be used as the conditioning label
            condition: Which label to use to condition the net
            apply_pos_filter: Apply POS filter on the tokens or not
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        examples = cls(True, headline_field, body_field, label_field, condition_field,
            condition, apply_pos_filter, **kwargs).examples
        if shuffle: random.shuffle(examples)
        dev_index = -1 * int(dev_ratio*len(examples))

        # Create train & test splits by calling the original class method to create
        # instances, but this time with the data already in hand
        return (cls(True, headline_field, body_field, label_field, condition_field,
            condition, apply_pos_filter, examples=examples[:dev_index]),
            cls(True, headline_field, body_field, label_field, condition_field,
                condition, apply_pos_filter, examples=examples[dev_index:]))

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
        apply_pos_filter = False, examples=None, **kwargs):
        """Create the FNC TEST dataset instance given fields.
        Arguments:
            headline_field: The field that will be used for headline data.
            body_field: The field that will be used for body data.
            label_field: The field that will be used for label data.
            condition_field: The field that will be used to indicate the conditioning label
            apply_pos_filter: Apply POS filter on the tokens or not
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [
            ('headline', headline_field),
            ('body', body_field),
            ('label', label_field),
            ('condition', condition_field)]

        if examples is None:
            datapoints_path = TEST_PATH
            with open(datapoints_path, 'rb') as pkl_fp:
                datapoints = pickle.load(pkl_fp)

            for datapoint in datapoints:
                datapoint['h'] = filter_and_limit_tokens(
                        datapoint['h'], apply_pos_filter, LEN_HEADLINE)
                datapoint['b'] = filter_and_limit_tokens(
                        datapoint['b'], apply_pos_filter, LEN_BODY)
                datapoint['y'] = 'discuss' if datapoint['y'] == 'unrelated' else datapoint['y']
                # Conditioning on the field that other fields must be opposite to
                # This is used for the CosineEmbeddingLoss where the vectors representing
                # opposing headlines and bodies will be trained to be orthogonal to each
                # other.
                datapoint['c'] = -1. if datapoint['y'] == 'disagree' else 1.

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
        apply_pos_filter=False, examples=None, **kwargs):
        """Create the FNC TRAIN dataset instance given fields.
        Arguments:
            headline_field: The field that will be used for headline data.
            body_field: The field that will be used for body data.
            label_field: The field that will be used for label data.
            ### condition field IGNORED
            condition_field: The field that will be used to indicate the conditioning label
            apply_pos_filter: Apply POS filter on the tokens or not
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [
            ('headline', headline_field),
            ('body', body_field),
            ('label', label_field),
            ('condition', condition_field)]

        if examples is None:
            datapoints_path = TRAIN_PATH
            with open(datapoints_path, 'rb') as pkl_fp:
                datapoints = pickle.load(pkl_fp)

            for datapoint in datapoints:
                datapoint['h'] = filter_and_limit_tokens(
                        datapoint['h'], apply_pos_filter, LEN_HEADLINE)
                datapoint['b'] = filter_and_limit_tokens(
                        datapoint['b'], apply_pos_filter, LEN_BODY)
                datapoint['y'] = 'discuss' if datapoint['y'] == 'unrelated' else datapoint['y']
                datapoint['c'] = None # THIS FIELD IS IGNORED

            examples = [data.Example.fromlist([
                " ".join(datapoint['h']),
                " ".join(datapoint['b']),
                datapoint['y'],
                datapoint['c']],
                fields) for datapoint in datapoints]

        super(FNC_1_Train_Untouched, self).__init__(examples, fields, **kwargs)

