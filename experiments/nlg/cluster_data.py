import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from utils_nlg import InputExample
from evaluate_nlg import answer_index, model_index, load_examples, learning_method_index


class InputConcat(InputExample):
    """Stores the concatenation of the context and its correct answer, the length of the result, its max tf-idf score and the cluster index of the datapoint"""    

    def __init__(self, input_obj, context_concat, length = 0, cluster = 0, tfidf = 0):

        InputExample.__init__(self, input_obj.example_id, input_obj.context, input_obj.endings, input_obj.label)
        self.context_concat = context_concat
        self.length = length
        self.tfidf = tfidf
        self.cluster = cluster

def plot_histogram(input, xlabel):
    """Plot a histogram given data

    Args:
        input (list): Input data that is plotted
        xlabel ([type]): Label for the OX axis
    """

    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

    plt.hist(input, bins=50)
    plt.gca().set(title='Histogram', xlabel=xlabel, ylabel='Frequency')
    plt.show()

def assign_clusters(eval_dataset_concat, attribute_values, attribute_name):
    """This method assigns a cluster to each datapoint given the data distribution and its length or max tfidf score

    Args:
        eval_dataset_concat (list): Concatenated evaluation dataset
        attribute_values (list): List of all lengths or tf-idf scores for faster computation of quartiles
        attribute_name (str): The current data paritioning relevant attribute
    """    

    q_1, q_2 = np.quantile(attribute_values, 0.33), np.quantile(attribute_values, 0.66)

    for example in eval_dataset_concat:
        attribute = getattr(example, attribute_name)
        example.cluster = 0 if attribute <= q_1 else 1 if attribute <= q_2 else 2

def concat_data(eval_dataset):
    """This method concatenates each dialogue with its correct continuation

    Args:
        eval_dataset (list): Evaluation dataset

    Returns:
        list: Concatenated evaluation dataset
    """

    label_options = list(answer_index.values())
    eval_dataset_concat = []

    for example in eval_dataset:

        context_concat = example.context + " " + example.endings[label_options.index(example.label)]
        new_example = InputConcat(example, context_concat)
        eval_dataset_concat.append(new_example)

    return eval_dataset_concat

def get_length(eval_dataset_concat):
    """Calculate the length of each datapoint

    Args:
        eval_dataset_concat (list): Concatenates evaluation dataset
    """    

    length = []

    for example in eval_dataset_concat:
        example.length = len(example.context_concat)   
        length.append(example.length)

    assign_clusters(eval_dataset_concat, length, 'length')
    plot_histogram(length, xlabel='Length of Dialogue in characters')

def get_token_length(eval_dataset_concat):
    """Compute average token length given tokenization processes for all models

    Args:
        eval_dataset_concat (list): Concatenated evaluation dataset
    """

    length_tokenizers = np.empty((len(model_index), len(eval_dataset_concat)))
    index_model = 0

    for model in model_index.values():

        print("Current model: ", model)

        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.pad_token = "[PAD]"

        length_tokenizers[index_model] = [len(tokenizer.tokenize(eval_dataset_concat[i].context_concat)) for i in range(len(eval_dataset_concat))]    
        index_model += 1

    mean_length_tokenizers = np.mean(length_tokenizers, axis = 0)

    j = 0
    for example in eval_dataset_concat:
        example.length = mean_length_tokenizers[j] 
        j += 1

    assign_clusters(eval_dataset_concat, mean_length_tokenizers, 'length')
    plot_histogram(mean_length_tokenizers, xlabel='Length of Tokenized Dialogue')

def get_tfidf(eval_dataset_concat):
    """Compute max tf-idf score for each example

    Args:
        eval_dataset_concat (list): Concatenated evaluation dataset
    """
    np.random.seed()
    corpus = [example.context_concat for example in eval_dataset_concat]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    array_tfidf = []

    for sentence_idx in range(X.shape[0]): # for each sentence get values and calculate max tf-idf

        max_tfidf = max([X[sentence_idx][(0, col)] for col in X[sentence_idx].nonzero()[1]])

        array_tfidf.append(max_tfidf)
        eval_dataset_concat[sentence_idx].tfidf = max_tfidf

    assign_clusters(eval_dataset_concat, array_tfidf, 'tfidf')

def data_partition(args,  eval_dataset):
    """This method selects the desired data partition

    Args:
        args (Namespace): Object from the argument parser
        eval_dataset (list): Evaluation dataset containing dialogue and all 4 possible answers
    """    

    # First concatenate the data
    eval_dataset_concat = concat_data(eval_dataset)

    if args.partition == 'length':
        get_length(eval_dataset_concat)
    elif args.partition == 'tf-idf':
        get_tfidf(eval_dataset_concat)
    elif args.partition == 'token_length':
        get_token_length(eval_dataset_concat)

    return eval_dataset_concat

def main(args):
    """Handles the arguments and starts data partitioning

    Args:
        args (Namespace): Object from the argument parser
    """    

    print('-----PARAMETERS-----')
    print('Data directory: {}'.format(args.data_dir))
    print('Data partitioning on: {}'.format(args.partition))
    print('-----------------------------')

    learning_method, _ = learning_method_index[args.learning_method]
    _, eval_dataset = load_examples(args, learning_method)
    eval_dataset_concat = data_partition(args, eval_dataset)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data_dir", default='data/mutual', type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task. Default is data/mutual.")
    parser.add_argument("--partition", default='length', type=str,
                        help="Types of data partition.",
                        choices=['length', 'token_length', 'tf-idf'])
    parser.add_argument('--model', default='gpt2', type=str,
                        help='Generator model type to use. Default is gpt2.',
                        choices=['gpt2', 'bart', 'gpt_neo', 'dialog_gpt', 'xlnet', 'xlprophetnet'])
    parser.add_argument('--learning_method', default='zero_shot', type=str,
                        help='Learning method to use. Default is zero_shot.',
                        choices=['zero_shot', '1_shot', '10_shot', '100_shot', '1000_shot', 'full_dataset'])

    args = parser.parse_args()

    main(args)
