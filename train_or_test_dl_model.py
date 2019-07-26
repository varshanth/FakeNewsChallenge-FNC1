import torch
import torchtext.data as data
from utils.fnc_dataset_loader import FNC_1, FNC_1_Train_Untouched, get_FNC_1_fields
import argparse
from dl_approach_cfg import *
from models.custom_cnn_model import ConditionedCNNClassifier, ConditionedSharedCNNClassifier
from utils.train_test_utils import train_model, test_model, report_fnc1_score

parser = argparse.ArgumentParser(description='CNN Based FNC Classifier')
parser.add_argument('-test', action='store_true', default=False, help = 'Activate Testing')
parser.add_argument('-weights_file', type=str, default=None, help = 'Path to Weights File')
parser.add_argument('-condition', type=str, default='unrelated', help='Label to Condition Network')
parser.add_argument('-apply_pos_filter', action='store_true', default=False, help = 'Apply POS filters')
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    print('--------------Start--------------')
    print('Preparing Train, Val & Test Sets')

    fields = get_FNC_1_fields()
    # Load the conditioned dataset
    train_data, val_data = FNC_1.splits(True, condition=args.condition,
            apply_pos_filter=args.apply_pos_filter, **fields)
    # Load the full training data to build the vocabulary
    full_train_data = FNC_1_Train_Untouched(
            apply_pos_filter=args.apply_pos_filter, **fields)
    test_data = FNC_1(False, condition=args.condition,
            apply_pos_filter=args.apply_pos_filter, **fields)
    # Build the vocabulary from all the data
    fields['headline_field'].build_vocab(full_train_data,
                                     max_size = DATA_CFG['MAX_VOCAB_SIZE'],
                                     vectors = DATA_CFG['VECTORS'],
                                     unk_init = torch.Tensor.normal_)

    fields['body_field'].build_vocab(full_train_data,
                                     max_size = DATA_CFG['MAX_VOCAB_SIZE'],
                                     vectors = DATA_CFG['VECTORS'],
                                     unk_init = torch.Tensor.normal_)
    # Build the labels from the conditioned data
    fields['label_field'].build_vocab(train_data, val_data)
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
            (train_data, val_data, test_data),
            batch_sizes = (TRAIN_CFG['BATCH_SIZE'], len(val_data), len(test_data)),
            device = DEVICE,
            sort_key = lambda x: len(x.headline+x.body),
            sort_within_batch=False)

    print(f'Training: {len(train_data)} Validation: {len(val_data)} Test: {len(test_data)}')

    print('Getting Model')

    TRAIN_CFG['WEIGHTS_PATH'] += f'_{args.condition}'
    VANILLA_COND_CNN_NET_CFG['num_classes'] = len(fields['label_field'].vocab)
    EMBED_CFG['H_V'] = len(fields['headline_field'].vocab)
    EMBED_CFG['B_V'] = len(fields['body_field'].vocab)
    model = ConditionedCNNClassifier(VANILLA_COND_CNN_NET_CFG, EMBED_CFG)

    if not args.test:
        print('Train Model Selected')
        model.h_embedding.weight.data.copy_(fields['headline_field'].vocab.vectors)
        model.b_embedding.weight.data.copy_(fields['body_field'].vocab.vectors)
        H_PAD_IDX = fields['headline_field'].vocab.stoi[fields['headline_field'].pad_token]
        B_PAD_IDX = fields['body_field'].vocab.stoi[fields['body_field'].pad_token]
        H_UNK_IDX = fields['headline_field'].vocab.stoi[fields['headline_field'].unk_token]
        B_UNK_IDX = fields['body_field'].vocab.stoi[fields['body_field'].unk_token]
        model.h_embedding.weight.data[H_UNK_IDX] = torch.zeros(EMBED_CFG['D'])
        model.b_embedding.weight.data[B_UNK_IDX] = torch.zeros(EMBED_CFG['D'])
        model.h_embedding.weight.data[H_PAD_IDX] = torch.zeros(EMBED_CFG['D'])
        model.b_embedding.weight.data[B_PAD_IDX] = torch.zeros(EMBED_CFG['D'])
        model.to(DEVICE)
        print('Training Model')
        train_model(model, train_iter, val_iter, DEVICE, TRAIN_CFG)

    else:
        print('Testing Model Selected')
        model.load_state_dict(torch.load(args.weights_file))
        test_loss, test_acc = test_model(model, test_iter, DEVICE, TRAIN_CFG)
        print(f'Test Loss: {test_loss} Accuracy: {test_acc}')
        report_fnc1_score(model, test_iter, fields['label_field'])

    print('--------------End--------------')

