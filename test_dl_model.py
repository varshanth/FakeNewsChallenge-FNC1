import torch
import torchtext.data as data
from fnc_dataset_loader import FNC_1, FNC_1_Train_Untouched, FNC_1_TEST_Unrelated_Is_Discuss, get_FNC_1_fields
import argparse
from dl_approach_cfg import TRAIN_CFG, DATA_CFG, NET_CFG, EMBED_CFG
from custom_cnn_model import ConditionedCNNClassifier
from train_test_utils import train_model, test_model, report_fnc1_score, get_test_predictions


def test_fnc1_model(weights_file, condition, device):
    print('--------------Start--------------')
    print('Preparing Train, Val & Test Sets')

    fields = get_FNC_1_fields()
    full_train_data = FNC_1_Train_Untouched(**fields)
    train_data, val_data = FNC_1.splits(True, condition=condition, **fields)
    test_data = FNC_1(False, condition=condition, **fields)
    # Build the vocabulary from all the data
    fields['headline_field'].build_vocab(full_train_data,
                                     max_size = DATA_CFG['MAX_VOCAB_SIZE'])

    fields['body_field'].build_vocab(full_train_data,
                                     max_size = DATA_CFG['MAX_VOCAB_SIZE'])
    # Build labels only from conditioned data
    fields['label_field'].build_vocab(train_data, val_data)
    test_iter = data.BucketIterator(
            test_data,
            batch_size = len(test_data),
            device = device,
            sort_key = lambda x: len(x.headline+x.body),
            sort_within_batch=False,
            shuffle=False)

    print(f'Training: {len(full_train_data)} Test: {len(test_data)}')

    print('Getting Model')

    NET_CFG['num_classes'] = len(fields['label_field'].vocab)
    EMBED_CFG['H_V'] = len(fields['headline_field'].vocab)
    EMBED_CFG['B_V'] = len(fields['body_field'].vocab)
    model = ConditionedCNNClassifier(NET_CFG, EMBED_CFG)

    print('Testing Model Selected')
    model.load_state_dict(torch.load(weights_file))
    model.to(device)
    test_loss, test_acc = test_model(model, test_iter, device)
    print(f'Test Loss: {test_loss} Accuracy: {test_acc}')
    report_fnc1_score(model, test_iter, fields['label_field'])
    print('--------------End--------------')


def get_predictions_from_FNC_1_Test(weights_file, device):
    print('--------------Start--------------')
    print('Preparing Train, Val & Test Sets')

    fields = get_FNC_1_fields()
    full_train_data = FNC_1_Train_Untouched(**fields)
    train_data, val_data = FNC_1.splits(True, condition='disagree', **fields)
    test_dummy_data = FNC_1(False, condition='disagree', **fields)
    test_data = FNC_1_TEST_Unrelated_Is_Discuss(**fields)
    # Build the vocabulary from all the data
    fields['headline_field'].build_vocab(full_train_data,
                                     max_size = DATA_CFG['MAX_VOCAB_SIZE'])

    fields['body_field'].build_vocab(full_train_data,
                                     max_size = DATA_CFG['MAX_VOCAB_SIZE'])
    # Build labels only from the conditioned data
    fields['label_field'].build_vocab(train_data, val_data)
    test_iter =data.BucketIterator(
            test_data,
            batch_size = len(test_data),
            device = device,
            sort_key = lambda x: len(x.headline+x.body),
            sort_within_batch=False,
            shuffle=False)

    print(f'Training: {len(full_train_data)} Test: {len(test_data)}')
    print('Getting Model')

    NET_CFG['num_classes'] = len(fields['label_field'].vocab)
    EMBED_CFG['H_V'] = len(fields['headline_field'].vocab)
    EMBED_CFG['B_V'] = len(fields['body_field'].vocab)
    model = ConditionedCNNClassifier(NET_CFG, EMBED_CFG)
    print('Testing Model Selected')
    model.load_state_dict(torch.load(weights_file))
    model.to(device)
    return get_test_predictions(model, test_iter, fields['label_field'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN Based FNC Classifier')
    parser.add_argument('-weights_file', type=str, required = True, default=None, help = 'Path to Weights File')
    args = parser.parse_args()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CONDITION = 'disagree'
    test_fnc1_model(args.weights_file, CONDITION, DEVICE)
    # predictions = get_predictions_from_FNC_1_Test(args.weights_file, DEVICE)
