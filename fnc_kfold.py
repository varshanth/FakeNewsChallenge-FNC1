import sys
import pandas as pd
import numpy as np
import pickle
import os

# from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats, word_tfidf_features, word_overlap_split_bodies_features
from feature_engineering import word_overlap_features, word_overlap_pos_features, word_overlap_quotes_features, word_tfidf_pos_ss_features, word_overlap_bpe_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, LABELS_RELATED, score_submission, score_cal

from utils.system import parse_params, check_version
from test_dl_model import get_predictions_from_FNC_1_Test
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = parse_params()

def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        if params.run_2_class:
            if name != 'competition': y.append(LABELS_RELATED.index(stance['Stance']))
            else : y.append(LABELS.index(stance['Stance']))
        else :
            y.append(LABELS.index(stance['Stance']))

        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")
    X_overlap_quotes = gen_or_load_feats(word_overlap_quotes_features, h, b, "features/overlap_quotes."+name+".npy")
    X_overlap_pos = gen_or_load_feats(word_overlap_pos_features, h, b, "features/overlap_pos."+name+".npy")
    X_overlap_pos_sentence = gen_or_load_feats(word_overlap_split_bodies_features,  h, b, "features/overlap_pos_sentence_split_bodies."+name+".npy")
    X_tfidf = gen_or_load_feats(word_tfidf_features, h, b, "features/tfidf_pos."+name+".npy")
    X_tfidf_max = gen_or_load_feats(word_tfidf_pos_ss_features, h, b, "features/tfidf_pos_max."+name+".npy")
    X_overlap_bpe_SS = gen_or_load_feats(word_overlap_bpe_features,  h, b, "features/overlap_bpe_nltk_tag3."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap, X_overlap_quotes, X_overlap_pos, X_overlap_pos_sentence, X_tfidf, X_tfidf_max, X_overlap_bpe_SS]
    return X,y

if __name__ == "__main__":
    check_version()

    print('Running Conditioned CNN on FNC1 Dataset')
    dl_model_pred, _unused1, _unused2 = get_predictions_from_FNC_1_Test(
            params.dl_weights_file, params.apply_pos_filter, DEVICE)

    #Load the training dataset and generate folds
    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    stances = pd.DataFrame(competition_dataset.stances)
    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))

    best_score = 0
    best_fold = None
    fold_score = 0.0
    max_fold_score = 1.0
    predicted = []
    actual = []
    if not os.path.exists(params.gb_weights_file):
        print(f'{params.gb_weights_file} Not Found. Training From Scratch')
        # # Classifier for each fold

        for fold in fold_stances:
            ids = list(range(len(folds)))
            del ids[fold]

            X_train = np.vstack(tuple([Xs[i] for i in ids]))
            y_train = np.hstack(tuple([ys[i] for i in ids]))

            X_test = Xs[fold]
            y_test = ys[fold]

            clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
            clf.fit(X_train, y_train)

            if params.run_2_class:
                predicted = [LABELS_RELATED[int(a)] for a in clf.predict(X_test)]
                actual = [LABELS_RELATED[int(a)] for a in y_test]

                fold_score = score_cal(actual, predicted)
                max_fold_score = score_cal(actual, actual)

            else:
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
    # Run on Holdout set and report the final score on the holdout set
    if params.run_2_class:
        predicted = [LABELS_RELATED[int(a)] for a in best_fold.predict(X_holdout)]
        actual = [LABELS_RELATED[int(a)] for a in y_holdout]
    else :
        predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
        actual = [LABELS[int(a)] for a in y_holdout]
        print("Scores on the dev set")
        report_score(actual,predicted)
        print("")
        print("")

    #Run on competition dataset
    predicted_combined = []
    if params.run_2_class:
        predicted = [LABELS_RELATED[int(a)] for a in best_fold.predict(X_competition)]
    else :
        predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]

    predicted_combined = [a if a == "unrelated" else aD for a,aD in zip(predicted, dl_model_pred)]
    actual = [LABELS[int(a)] for a in y_competition]
    print("Scores on the test set")
    report_score(actual, predicted_combined)

    final_result = {"Headline" : stances['Headline'], "Body ID" : stances['Body ID'], 'Stance' : predicted_combined}
    final_result = pd.DataFrame(final_result)
    final_result.to_csv(r'baseline-results/submission.csv', index=False, header=True)