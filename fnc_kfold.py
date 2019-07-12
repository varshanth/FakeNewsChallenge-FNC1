import sys
import pandas as pd
import numpy as np
import pickle
import os

# from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features, word_overlap_pos_features, word_overlap_quotes_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from utils.system import parse_params, check_version
from test_dl_model import get_predictions_from_FNC_1_Test
import argparse
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")
    X_overlap_pos = gen_or_load_feats(word_overlap_pos_features, h, b, "features/overlap_pos."+name+".npy")
    X_overlap_quotes = gen_or_load_feats(word_overlap_quotes_features, h, b, "features/overlap_quotes."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap, X_overlap_pos, X_overlap_quotes]
    return X,y

if __name__ == "__main__":
    check_version()
    params = parse_params()
    print('Running Conditioned CNN on FNC1 Dataset')
    dl_model_pred = get_predictions_from_FNC_1_Test(params.dl_weights_file, DEVICE)

    #Load the training dataset and generate folds
    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    stances = pd.DataFrame(competition_dataset.stances)
    # stances['Body ID'].to_csv(r'baseline-results/body_id.csv', index=False, header=False)
    # stances['Headline'].to_csv(r'baseline-results/headline.csv', index=False, header=False)
    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))


    best_score = 0
    best_fold = None

    if not os.path.exists(params.gb_weights_file):
        print(f'{params.gb_weights_file} Not Found. Training From Scratch')
        # classifiers = {model_rf :(0, None), model_gdb : (0, None)}
        # # Classifier for each fold
        # for classifier in classifiers:

        for fold in fold_stances:
            ids = list(range(len(folds)))
            del ids[fold]

            X_train = np.vstack(tuple([Xs[i] for i in ids]))
            y_train = np.hstack(tuple([ys[i] for i in ids]))

            X_test = Xs[fold]
            y_test = ys[fold]

            clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
            clf.fit(X_train, y_train)

            predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
            actual = [LABELS[int(a)] for a in y_test]

            fold_score, _ = score_submission(actual, predicted)
            max_fold_score, _ = score_submission(actual, actual)

            score = fold_score/max_fold_score

            print("Score for fold "+ str(fold) + " was - " + str(score))
            if score > best_score:
                best_score = score
                best_fold = clf
        pickle.dump(best_fold, open(params.gb_weights_file, 'wb'))

    best_fold = pickle.load(open(params.gb_weights_file, 'rb'))
    #Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]

    print("Scores on the dev set")
    report_score(actual,predicted)
    print("")
    print("")


    #Run on competition dataset
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    predicted_combined = [a if a == "unrelated" else aD for a,aD in zip(predicted, dl_model_pred)]
    actual = [LABELS[int(a)] for a in y_competition]
    report_score(actual, predicted_combined)

    predicted_df = pd.DataFrame(
            {'gb_pred': predicted,
             'dl_pred': dl_model_pred,
             'actual' : actual})
    predicted_df.to_csv(r'comparison.csv', index=False, header=True)

    print("Scores on the test set")
