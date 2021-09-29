# imports
import os
import argparse
import math
import torch
import pandas as pd
import numpy as np
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from utils_nlg import processors, InputExample

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import collections

import time

class InputConcat(InputExample):
    def __init__(self, input_obj, context_concat, length = 0, label = 0):
        InputExample.__init__(self, input_obj.example_id, input_obj.context, input_obj.endings, input_obj.label)
        self.context_concat = context_concat
        self.length = length
        self.label = label

# index for indices to answers
answer_index = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}

def load_examples(args):
    """
    Function for loading the examples.
    Inputs:
        args - Namespace object from the argument parser
    Outputs:
        examples - List of development set examples
    """

    # Load data features
    print("Creating features from dataset file at {}".format(args.data_dir))
    processor = processors["mutual"]()
    label_list = processor.get_labels()
    examples = processor.get_dev_examples(args.data_dir)
    print("Number of evaluation instances: {}".format(len(examples)))

    # Return the items
    return examples

def concat_data(eval_dataset):

    label_options = list(answer_index.values())
    eval_dataset_concat = []

    for example in eval_dataset:

        # print("Context: ", example.context)
        # print("Possible answers: ", example.endings)
        # print("Example ID: ", example.example_id)
        # print("True label: ", example.label)
        # print("True label: ", label_options.index(example.label))
        # print(example.endings[label_options.index(example.label)])
        # print(example.context + " " + example.endings[label_options.index(example.label)], '\n')

        context_concat = example.context + " " + example.endings[label_options.index(example.label)]
        new_example = InputConcat(example, context_concat)
        eval_dataset_concat.append(new_example)

    return eval_dataset_concat

def get_length(eval_dataset_concat):

    length = []
    for example in eval_dataset_concat:
        example.length = len(example.context_concat)   
        length.append(example.length)
    
    start = time.time()
    q_1, q_2, q_3 = np.quantile(length, 0.25), np.quantile(length, 0.50), np.quantile(length, 0.75)

    for example in eval_dataset_concat:
        if example.length <= q_1:
            example.label = 0
        elif example.length <= q_2:
            example.label = 1
        elif example.length <= q_3:
            example.label = 2
        else:
            example.label = 3

    print(time.time()-start)
    # Plot Histogram on length
    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

    plt.hist(length, bins=50)
    plt.gca().set(title='Frequency Histogram', xlabel='Length of Dialogue in characters', ylabel='Frequency')
    plt.show()
    
def get_tfidf(eval_dataset_concat):
    
    corpus = [example.context_concat for example in eval_dataset_concat]
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    
    print("WORDS: ", vectorizer.get_feature_names_out())   
    
    clusters = KMeans(n_clusters=4, n_init=128).fit_predict(X)
    print("LABELS: ", collections.Counter(clusters))
    
    for i, example in enumerate(eval_dataset_concat):
        example.label = clusters[i]
    
    # Plot clusters:
    pca = PCA(n_components=2)
    scatter_plot_points = pca.fit_transform(X.toarray())
    
    colors = ['r', 'b', 'c', 'y', 'm', 'g']

    x_axis = [o[0] for o in scatter_plot_points]
    y_axis = [o[1] for o in scatter_plot_points]

    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(x_axis, y_axis, c=[colors[d] for d in clusters])

    plt.show()

def data_partition(args, device, partition):
    """
    Function for evaluating the model.
    Inputs:
        args - Namespace object from the argument parser
        device - PyTorch device to place the model on
    Outputs:
        ?
    """
    
    print('Loading data..')
    eval_dataset = load_examples(args)
    eval_dataset_concat = concat_data(eval_dataset)
    
    print("here: ", eval_dataset_concat[885].example_id)
    print("here: ", eval_dataset_concat[885].context_concat, '\n')
    
    if partition == 'length':
        get_length(eval_dataset_concat)
    elif partition == 'tf-idf':
        get_tfidf(eval_dataset_concat)
    
def main(args):
    """
    Function for handling the arguments and starting data clustering
    Inputs:
        args - Namespace object from the argument parser
    """

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print the given parameters
    print('-----EVALUATION PARAMETERS-----')
    print('Device: {}'.format(device))
    print('Max sequence length: {}'.format(args.max_seq_length))
    print('Data directory: {}'.format(args.data_dir))
    print('Output directory: {}'.format(args.output_dir))
    print('-----------------------------')

    # Start evaluation

    data_partition(args, device, partition = 'tf-idf')


# Command line arguments parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Hyperparameters
    parser.add_argument('--model', default='gpt2', type=str,
                        help='Generator model type to use. Default is gpt2.',
                        choices=['gpt2'])
    parser.add_argument("--data_dir", default='data/mutual', type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task. Default is data/mutual.")
    parser.add_argument("--output_dir", default='experiment_outputs/', type=str,
                        help="The output directory for the .csv files. Default is experiment_outputs/.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded. Default is 128.")

    # Parse the arguments
    args = parser.parse_args()

    # Evaluate the model
    main(args)