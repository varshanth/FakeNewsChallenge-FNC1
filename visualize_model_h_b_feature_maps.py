import argparse
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import sys
import torch
from test_dl_model import get_predictions_from_FNC_1_Test

def parse_cmd_line_args():
    parser = argparse.ArgumentParser(description='Visualize H-Vec & B-Vec')
    parser.add_argument('-test_pkl', type=str, required=True,
            default=None, help = 'Path to Test PKL File')
    parser.add_argument('-weights_file', type=str, required=True,
            default=None, help = 'Path to Weights File')
    parser.add_argument('-plots_dir', type=str, required=True, default=None,
            help='Path to the Plots Directory to Output')
    parser.add_argument('-misclassified', action='store_true', default=False,
            help = 'Show plots only for misclassified points')
    parser.add_argument('-apply_pos_filter', action='store_true', default=False,
            help = 'Apply POS Filters')
    args = parser.parse_args()
    return args

def apply_pca_gen_plot(grouped_predictions, output_dir):
    fig = plt.figure()
    fig.suptitle('PCA Visualization for Headline & Body FT Vectors', fontsize=9)
    group_i = 1
    num_cols = len(grouped_predictions)
    set_ylabel_flag = False
    for label, vecs in grouped_predictions.items():
        print(f'Processing label {label.upper()}')
        X = np.vstack([vecs['h_vec'], vecs['b_vec']])
        print('Standardizing Data')
        X = StandardScaler().fit_transform(X)
        print('Applying PCA')
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X)
        print(f'Explained Variance: {pca.explained_variance_}')
        print('Plotting Datapoints')
        ax = fig.add_subplot(1, num_cols, group_i)
        ax.set_xlabel('Principal Component 1', fontsize=8)
        if not set_ylabel_flag:
            ax.set_ylabel('Principal Component 2', fontsize=8)
            set_ylabel_flag = True
        ax.set_title(f'{label.upper()}', fontsize=8)
        targets = ['Headline Features', 'Body Features']

        ax.scatter(
                principal_components[:len(vecs['h_vec']), 0],
                principal_components[:len(vecs['h_vec']), 1],
                c = 'r')

        ax.scatter(
                principal_components[len(vecs['h_vec']):, 0],
                principal_components[len(vecs['h_vec']):, 1],
                c = 'b')
        ax.legend(targets, loc='upper left')
        group_i += 1
    # plt.tight_layout()
    fig.subplots_adjust(wspace=0.5)
    plt.savefig(f'{output_dir}/h_b_ft_visualization.png', dpi = 500)
    # plt.show()


def get_cosine_sim_distribution_plot(grouped_predictions, output_dir):
    fig = plt.figure()
    fig.suptitle('Cosine Distance Distributions', fontsize=9)
    group_i = 1
    num_cols = len(grouped_predictions)
    set_ylabel_flag = False
    for label, vecs in grouped_predictions.items():
        print('Calculating Cosine Distances')
        dot_prod = np.sum(vecs['h_vec'] * vecs['b_vec'], axis = 1)
        h_vec_norm = np.linalg.norm(vecs['h_vec'], axis = 1)
        b_vec_norm = np.linalg.norm(vecs['b_vec'], axis = 1)
        norm_prod = h_vec_norm * b_vec_norm
        cos_sim = dot_prod / (0.000000001 + norm_prod)
        ax = fig.add_subplot(1, num_cols, group_i)
        n, bins, patches = ax.hist(cos_sim, facecolor='blue', alpha=0.5)
        ax.set_xlabel('Cosine Distance', fontsize=8)
        if not set_ylabel_flag:
            ax.set_ylabel('Frequency', fontsize=8)
            set_ylabel_flag = True
        ax.set_title(f'{label.upper()}', fontsize=8)
        group_i += 1
    # plt.tight_layout()
    fig.subplots_adjust(wspace=0.5)
    plt.savefig(f'{output_dir}/h_b_cos_dis_visualization.png', dpi = 500)
    # plt.show()

if __name__ == '__main__':
    args = parse_cmd_line_args()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('-------Start Visualize-------')
    print('Getting Predictions from Test Dataset')
    dl_model_pred, h_vec, b_vec = get_predictions_from_FNC_1_Test(
            args.weights_file, args.apply_pos_filter, DEVICE)

    print('Getting datapoints from PKL file')
    with open(args.test_pkl, 'rb') as test_pkl_fp:
        datapoints = pickle.load(test_pkl_fp)

    print('Organizing the data')
    dl_model_pred = np.array(dl_model_pred)
    gold_labels = np.array([datapoint['y'] for datapoint in datapoints])

    # Filter only related labels
    non_unrelated_indices = gold_labels != 'unrelated'
    gold_labels = gold_labels[non_unrelated_indices]
    dl_model_pred = dl_model_pred[non_unrelated_indices]
    h_vec = h_vec[non_unrelated_indices]
    b_vec = b_vec[non_unrelated_indices]

    labels = np.unique(gold_labels)

    if args.misclassified:
        misclassified_indices = gold_labels != dl_model_pred
        gold_labels = gold_labels[misclassified_indices]
        h_vec = h_vec[misclassified_indices]
        b_vec = b_vec[misclassified_indices]
        dl_model_pred = dl_model_pred[misclassified_indices]

    grouped_predictions = {
            label : {
                'h_vec' : h_vec[gold_labels == label],
                'b_vec' : b_vec[gold_labels == label]
                }
            for label in labels
            }

    apply_pca_gen_plot(grouped_predictions, args.plots_dir)
    get_cosine_sim_distribution_plot(grouped_predictions, args.plots_dir)

    print('--------End Visualize--------')

