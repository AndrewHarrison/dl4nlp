import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from utils_nlg import InputExample
from evaluate_nlg import answer_index, model_index, load_examples, calculate_eval_metrics


class InputConcat(InputExample):
    """Contains the concatenation of the context and its correct answer, the length of the result and the cluster inex of the datapoint"""    

    def __init__(self, input_obj, context_concat, length = 0, cluster = 0):

        InputExample.__init__(self, input_obj.example_id, input_obj.context, input_obj.endings, input_obj.label)
        self.context_concat = context_concat
        self.length = length
        self.cluster = cluster

def concat_data(eval_dataset):
    """This method concatenates the data: each dialogue with its correct continuation

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

def plot_histogram(input, xlabel):
    """Plot a histogram given data

    Args:
        input (list): Input data that is plotted
        xlabel ([type]): Label for the OX axis
    """

    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

    plt.hist(input, bins=50)
    plt.gca().set(title='Frequency Histogram', xlabel=xlabel, ylabel='Frequency')
    plt.show()

def assign_length_clusters(eval_dataset_concat, length):
    """This method assigns a cluster to each datapoint given its length and the data distribution

    Args:
        eval_dataset_concat (list): Concatenated evaluation dataset
        length (list): List of all lengths for faster computation of quartiles
    """    
    
    q_1, q_2, q_3 = np.quantile(length, 0.25), np.quantile(length, 0.50), np.quantile(length, 0.75)

    for example in eval_dataset_concat:
        example.cluster = 0 if example.length <= q_1 else 1 if example.length <= q_2 else 2 if example.length <= q_3 else 3
    
def get_length(eval_dataset_concat):
    """Calculate the length of each datapoint

    Args:
        eval_dataset_concat (list): Concatenates evaluation dataset
    """    

    length = []

    for example in eval_dataset_concat:
        example.length = len(example.context_concat)   
        length.append(example.length)
    
    assign_length_clusters(eval_dataset_concat, length)
    plot_histogram(length, xlabel='Length of Dialogue in characters')

def get_token_length(eval_dataset_concat):
    """Compute average token length given tokenization processes for all models

    Args:
        eval_dataset_concat (list): Concatenated evaluation dataset
    """    
    
    length_tokenizers = np.empty((len(model_index), len(eval_dataset_concat)))
    index_m = 0

    for model in model_index.values():
        
        print("Current model: ", model)

        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.pad_token = "[PAD]"
  
        length_tokenizers[index_m] = [len(tokenizer.tokenize(eval_dataset_concat[i].context_concat)) for i in range(len(eval_dataset_concat))]    
        index_m += 1
    
    mean_length_tokenizers = np.mean(length_tokenizers, axis = 0)
    
    j = 0
    for example in eval_dataset_concat:
        example.length = mean_length_tokenizers[j] 
        j += 1
        
    assign_length_clusters(eval_dataset_concat, mean_length_tokenizers)
    plot_histogram(mean_length_tokenizers, xlabel='Length of Tokenized Dialogue')
    
def get_tfidf(eval_dataset_concat):
    """Compute tf-idf scores for each datapoint and form clusters by applying KMeans on the data

    Args:
        eval_dataset_concat (list): Concatenated evaluation dataset
    """    
    corpus = [example.context_concat for example in eval_dataset_concat]
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    
    clusters = KMeans(n_clusters=4, n_init=128).fit_predict(X)
    
    for i, example in enumerate(eval_dataset_concat):
        example.cluster = clusters[i]
    
    # Plot clusters:
    pca = PCA(n_components=2)
    scatter_plot_points = pca.fit_transform(X.toarray())
    
    colors = ['r', 'b', 'c', 'y', 'm', 'g']

    x_axis = [o[0] for o in scatter_plot_points]
    y_axis = [o[1] for o in scatter_plot_points]

    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(x_axis, y_axis, c=[colors[d] for d in clusters])

    plt.show()

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

def eval_model(args, eval_dataset, model, tokenizer):
    """This method is used for evaluating the model

    Args:
        args (Namespace): Object from the argument parser
        eval_dataset (list): Current dataset used for evaluation
        model (transformer): Current model
        tokenizer (transformer): The tokenizer wrt the current model
    """    
    
    # Evaluate the model
    print('Starting evaluation..')
    example_ids = []
    predicted_labels = []
    real_labels = []
    full_predictions = []
    for example in eval_dataset:
        scored_options = []
        # Create all possible options
        for ending_index, ending in enumerate(example.endings):
            model.eval()
            with torch.no_grad():
                # Pass through the model
                inputs = tokenizer.encode_plus(
                    example.context,
                    ending,
                    add_special_tokens=True,
                    truncation=True,
                    return_tensors="pt",
                )
                outputs = model(**inputs, labels=inputs['input_ids'])
                perplexity = outputs.loss
                # Calculate the perplexity
                option_letter = answer_index[ending_index]
                scored_options.append((option_letter, perplexity))
        # Take the option with lowest perplexity
        predictions = sorted(scored_options, key=lambda x: x[1])
        print(predictions)
        predictions, _ = map(list, zip(*predictions))
        predicted = predictions[0]
        print(predicted)
        example_ids.append(example.example_id)
        predicted_labels.append(predicted)
        real_labels.append(example.label)
        full_predictions.append(predictions)
            
    # Calculate the evaluation metrics
    R4_1, R4_2, mrr = calculate_eval_metrics(full_predictions, real_labels)

    # Print the evaluation metrics
    print('R4_1: {}'.format(R4_1))
    print('R4_2: {}'.format(R4_2))
    print('MRR: {}'.format(mrr))
    print('Evaluation finished', '\n')
    
def eval_data_partitions(args, eval_dataset_concat):
    """This method divides the data into partitions according to their assigned cluster label and runs evaluation on each

    Args:
        args (Namespace): Object from the argument parser
        eval_dataset_concat (list): Concatenated evaluation dataset
    """    
    
    cluster_values = set(map(lambda x:x.cluster, eval_dataset_concat))
    # print(cluster_values)
    partitioned_dataset = [[example for example in eval_dataset_concat if example.cluster==x] for x in cluster_values]

    print('Loading model..')
    config = AutoConfig.from_pretrained(model_index[args.model])
    model = AutoModelForCausalLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(model_index[args.model])
    print('Model loaded')

    for cluster in cluster_values:
        print('Cluster ', cluster)
        eval_model(args, partitioned_dataset[cluster], model, tokenizer)
        

def main(args):
    """Handles the arguments and starts data partitioning

    Args:
        args (Namespace): Object from the argument parser
    """    

    # Print the given parameters
    print('-----PARAMETERS-----')
    print('Data directory: {}'.format(args.data_dir))
    print('Data partitioning on: {}'.format(args.partition))
    print('-----------------------------')

    # Load data and start data partitioning
    train_dataset, eval_dataset = load_examples(args)
    eval_dataset_concat = data_partition(args, eval_dataset)
        
    eval_data_partitions(args, eval_dataset_concat)

        
# Command line arguments parsing
if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Hyperparameters
    parser.add_argument("--data_dir", default='data/mutual', type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task. Default is data/mutual.")
    parser.add_argument("--partition", default='length', type=str,
                        help="Types of data partition. Can be one of the following: [length, token_length, tf-idf]/.")
    parser.add_argument('--model', default='gpt2', type=str,
                        help='Generator model type to use. Default is gpt2.',
                        choices=['gpt2', 'bart', 'gpt_neo', 'dialog_gpt', 'xlnet', 'xlprophetnet'])
    parser.add_argument('--learning_method', default='zero_shot', type=str,
                        help='Learning method to use. Default is zero_shot.',
                        choices=['zero_shot', '1_shot', '10_shot', '100_shot', '1000_shot', 'full_dataset'])
    # Parse the arguments
    args = parser.parse_args()

    # Evaluate the model
    main(args)
