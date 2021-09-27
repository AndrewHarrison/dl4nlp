# imports
import os
import argparse
import math
import torch
import pandas as pd
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from utils_nlg import processors

# index for indices to answers
answer_index = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}

# index for models to their pretrained directories
model_index = {
    'gpt2': 'gpt2',
    'bart': 'facebook/bart-large',
    'gpt_neo': 'EleutherAI/gpt-neo-1.3B',
    'bigbird_pegasus': 'google/bigbird-pegasus-large-arxiv',
    'roberta': 'roberta-base',
    'xlprophetnet': 'microsoft/xprophetnet-large-wiki100-cased',
    'rembert': 'rembert',
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


def calculate_perplexity(loss, sentence_length):
    """
    Function for calculating the perplexity of an option.
    Inputs:
        loss - Loss from the model for a certain option
        sentence_length - Length of the sentence
    Outputs:
        perplexity - Perplexity value for the option
    """

    # Calculate the perplexity
    perplexity = math.exp(loss / sentence_length)

    # Return the perplexity
    return perplexity


def calculate_eval_metrics(full_predictions, predicted_labels, real_labels):
    """
    Function for calculating the evaluation metrics.
    Inputs:
        full_predictions - List of ordered lists containing the predictions
        predicted_labels - List of the model label predictions
        real_labels - List of real labels
    Outputs:
        R4_1 - Recall at 1
        R4_2 - Recall at 2
        MRR - Mean Reciprocal Rank
    """

    mrr = 0
    p2 = 0
    p1 = 0
    for instance_index, full_prediction in enumerate(full_predictions):
        # Get the position of the real prediction in the predictions
        position = full_prediction.index(real_labels[instance_index])
        mrr += 1 / (position + 1)
        if position == 0:
            p1 += 1
        if position == 1:
            p2 += 1
    mrr = mrr / len(real_labels)
    R4_1 = p1 / len(real_labels)
    R4_2 = R4_1 + (p2 / len(real_labels))

    # Return the measures
    return R4_1, R4_2, mrr


def evaluate_model(args, device):
    """
    Function for evaluating the model.
    Inputs:
        args - Namespace object from the argument parser
        device - PyTorch device to place the model on
    """

    # Load the model
    print('Loading model..')
    config = AutoConfig.from_pretrained(model_index[args.model])
    model = AutoModelForCausalLM.from_config(config)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_index[args.model])
    print('Model loaded')

    # Load the data
    print('Loading data..')
    eval_dataset = load_examples(args)
    print('Data loaded')

    # Evaluate the model
    print('Starting evaluation..')
    example_ids = []
    predicted_labels = []
    real_labels = []
    full_predictions = []
    for example in eval_dataset:
        model.eval()
        with torch.no_grad():
            scored_options = []
            # Create all possible options
            for ending_index, ending in enumerate(example.endings):
                # Pass through the model
                inputs = tokenizer.encode_plus(
                    example.context,
                    ending,
                    add_special_tokens=True,
                    max_length=args.max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs.to(device)
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                # Calculate the perplexity
                option_letter = answer_index[ending_index]
                perplexity = calculate_perplexity(loss, inputs['input_ids'].size()[0])
                scored_options.append((option_letter, perplexity))
            # Take the option with lowest perplexity
            predictions = sorted(scored_options, key=lambda x: x[1])
            predictions, _ = map(list, zip(*predictions))
            predicted = predictions[0]
            example_ids.append(example.example_id)
            predicted_labels.append(predicted)
            real_labels.append(example.label)
            full_predictions.append(predictions)

    # Save the results in a .csv file
    df = pd.DataFrame(list(zip(example_ids, predicted_labels, real_labels)), columns=['example_id', 'predicted_label', 'real_label'])
    df['correct'] = np.where(df['predicted_label'] == df['real_label'], True, False)
    df.to_csv(args.output_dir + args.model + '_results.csv')

    # Calculate the evaluation metrics
    R4_1, R4_2, mrr = calculate_eval_metrics(full_predictions, predicted_labels, real_labels)

    # Print the evaluation metrics
    print('R4_1: {}'.format(R4_1))
    print('R4_2: {}'.format(R4_2))
    print('MRR: {}'.format(mrr))
    print('Evaluation finished')


def main(args):
    """
    Function for handling the arguments and starting the evaluation.
    Inputs:
        args - Namespace object from the argument parser
    """

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print the given parameters
    print('-----EVALUATION PARAMETERS-----')
    print('Device: {}'.format(device))
    print('Model: {}'.format(args.model))
    print('Max sequence length: {}'.format(args.max_seq_length))
    print('Data directory: {}'.format(args.data_dir))
    print('Output directory: {}'.format(args.output_dir))
    print('-----------------------------')

    # Start evaluation
    evaluate_model(args, device)


# Command line arguments parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Hyperparameters
    parser.add_argument('--model', default='gpt2', type=str,
                        help='Generator model type to use. Default is gpt2.',
                        choices=['gpt2', 'bart', 'gpt_neo', 'bigbird_pegasus', 'roberta', 'xlprophetnet', 'rembert'])
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