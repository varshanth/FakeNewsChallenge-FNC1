import torch
import torchtext.data as data
from fnc_dataset_loader import FNC_1, get_FNC_1_fields
import argparse
from dl_approach_cfg import TRAIN_CFG, DATA_CFG, NET_CFG, EMBED_CFG
from custom_cnn_model import ConditionedCNNClassifier
from train_test_utils import train_model, test_model, report_fnc1_score

parser = argparse.ArgumentParser(description='CNN Based FNC Classifier')
parser.add_argument('-test', action='store_true', default=False, help = 'Activate Testing')
parser.add_argument('-weights_file', type=str, default=None, help = 'Path to Weights File')
parser.add_argument('-condition', type=str, default='unrelated', help='Label to Condition Network')
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    print('--------------Start--------------')

    print('Preparing Train, Val & Test Sets')

    fields = get_FNC_1_fields()
    train_data, val_data = FNC_1.splits(True, condition=args.condition, **fields)
    test_data = FNC_1(False, condition=args.condition, **fields)
    # Build the vocabulary from all the data
    fields['text_field'].build_vocab(train_data, val_data, test_data,
                                     max_size = DATA_CFG['MAX_VOCAB_SIZE'],
                                     vectors = DATA_CFG['VECTORS'],
                                     unk_init = torch.Tensor.normal_)

    fields['label_field'].build_vocab(train_data, val_data)
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
            (train_data, val_data, test_data),
            batch_sizes = (TRAIN_CFG['BATCH_SIZE'], len(val_data), len(test_data)),
            device = DEVICE,
            sort_key = lambda x: len(x.text),
            sort_within_batch=False)

    print(f'Training: {len(train_data)} Validation: {len(val_data)} Test: {len(test_data)}')

    print('Getting Model')

    NET_CFG['num_classes'] = len(fields['label_field'].vocab)
    EMBED_CFG['V'] = len(fields['text_field'].vocab)
    model = ConditionedCNNClassifier(NET_CFG, EMBED_CFG)

    if not args.test:
        print('Train Model Selected')
        model.embedding.weight.data.copy_(fields['text_field'].vocab.vectors)
        PAD_IDX = fields['text_field'].vocab.stoi[fields['text_field'].pad_token]
        UNK_IDX = fields['text_field'].vocab.stoi[fields['text_field'].unk_token]
        model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBED_CFG['D'])
        model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBED_CFG['D'])
        model.to(DEVICE)
        print('Training Model')
        train_model(model, train_iter, val_iter, DEVICE, TRAIN_CFG)

    else:
        print('Testing Model Selected')
        model.load_state_dict(torch.load(args.weights_file))
        test_loss, test_acc = test_model(model, test_iter, DEVICE)
        print(f'Test Loss: {test_loss} Accuracy: {test_acc}')
        report_fnc1_score(model, test_iter, fields['label_field'])

