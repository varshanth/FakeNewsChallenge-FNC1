import torch
import torch.nn as nn
import torch.optim as optim
import time
from early_stopping import EarlyStoppingWithSaveWeights
from utils.score import report_score


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1) # get the index of the max probability
    correct = max_preds.eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])


def run_epoch(model, train_iter, criteria, optimizer):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in train_iter:
        optimizer.zero_grad()
        predictions, h_vec, b_vec = model(batch.text)

        classif_loss = criteria['classif'](predictions, batch.label)
        condition_field_alignment_loss = criteria['condition_field_align'](
                h_vec, b_vec, batch.condition)
        total_loss = classif_loss + condition_field_alignment_loss
        acc = categorical_accuracy(predictions, batch.label)

        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(train_iter), epoch_acc / len(train_iter)


def evaluate(model, iterator, criteria):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions, h_vec, b_vec = model(batch.text)
            classif_loss = criteria['classif'](predictions, batch.label)
            condition_field_alignment_loss = criteria['condition_field_align'](
                    h_vec, b_vec, batch.condition)
            total_loss = classif_loss + condition_field_alignment_loss
            acc = categorical_accuracy(predictions, batch.label)

            epoch_loss += total_loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def train_model(model, train_iter, val_iter, device, train_cfg):
    optimizer = optim.Adam(model.parameters(), lr = train_cfg['LR'])
    es = EarlyStoppingWithSaveWeights(
            model,
            train_cfg['PATIENCE'],
            train_cfg['WEIGHTS_PATH'])
    optimizer = optim.Adam(model.parameters(), train_cfg['LR'])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                             step_size = train_cfg['LR_DECAY_STEPS'],
                                             gamma = train_cfg['LR_DECAY_GAMMA'])
    classif_criterion = nn.CrossEntropyLoss()
    condition_field_align_criterion = nn.CosineEmbeddingLoss(margin=0.0)

    classif_criterion = classif_criterion.to(device)
    condition_field_align_criterion = condition_field_align_criterion.to(device)

    criteria = {
            'classif' : classif_criterion,
            'condition_field_align' : condition_field_align_criterion,
            }

    for epoch in range(train_cfg['N_EPOCHS']):
        start_time = time.time()
        train_loss, train_acc = run_epoch(model, train_iter, criteria, optimizer)
        val_loss, val_acc = evaluate(model, val_iter, criteria)
        end_time = time.time()
        lr_scheduler.step()
        ep_m, ep_s = epoch_time(start_time, end_time)

        print(f'Epoch {epoch+1} : {ep_m}m {ep_s}s')
        print(f'Training Loss: {train_loss} Accuracy: {train_acc}')
        print(f'Validation Loss: {val_loss} Accuracy: {val_acc}')
        if es.step(val_loss):
            break


def test_model(model, test_iter, device):
    classif_criterion = nn.CrossEntropyLoss()
    condition_field_align_criterion = nn.CosineEmbeddingLoss()

    classif_criterion = classif_criterion.to(device)
    condition_field_align_criterion = condition_field_align_criterion.to(device)

    criteria = {
            'classif' : classif_criterion,
            'condition_field_align' : condition_field_align_criterion,
            }
    test_loss, test_acc = evaluate(model, test_iter, criteria)
    return test_loss, test_acc


def get_test_predictions(model, test_iter, label_field):
    # Assuming test iteration will iterate only once
    model.eval()
    with torch.no_grad():
        for batch in test_iter:
            predictions, h_vec, b_vec = model(batch.text)
            predictions = predictions.argmax(dim = 1).cpu().detach().numpy()
            actual = batch.label.cpu().detach().numpy()
            predictions = [label_field.vocab.itos[pred] for pred in predictions]
            return predictions

def report_fnc1_score(model, test_iter, label_field):
    model.eval()
    with torch.no_grad():
        for batch in test_iter:
            predictions, h_vec, b_vec = model(batch.text)
            predictions = predictions.argmax(dim = 1).cpu().detach().numpy()
            actual = batch.label.cpu().detach().numpy()
            predictions = [label_field.vocab.itos[pred]
                    if label_field.vocab.itos[pred] != 'related'
                    else 'discuss' # THIS IS A RANDOM LABEL ASSIGNED FOR SCORING PURPOSE
                    for pred in predictions]
            actual = [label_field.vocab.itos[act]
                    if label_field.vocab.itos[act] != 'related'
                    else 'discuss' # THIS IS A RANDOM LABEL ASSIGNED FOR SCORING PURPOSE
                    for act in actual]
            report_score(actual, predictions)

