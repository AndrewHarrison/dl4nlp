# Imports
import os
import argparse
import math
import torch
import pandas as pd
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AdamW
from utils_nlg import processors

# Index for indices to answers
answer_index = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}

# Index for answers to indices
reverse_answer_index = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
}

# Index for models to their pretrained directories
model_index = {
    'gpt2': 'gpt2',
    'bart': 'facebook/bart-large',
    'gpt_neo': 'EleutherAI/gpt-neo-1.3B',
    'dialog_gpt': 'microsoft/DialoGPT-large',
    'xlnet': 'xlnet-base-cased',
    'xlprophetnet': 'microsoft/xprophetnet-large-wiki100-cased',
    'blenderbot': 'facebook/blenderbot-3B',
}

# Index for learning methods
learning_method_index = {
    'zero_shot': 0,
    '1_shot': 1,
    '10_shot': 10,
    '100_shot': 100,
    '1000_shot': 1000,
    'full_dataset': -1,
}


def load_examples(args):
    """
    Function for loading the examples.
    Inputs:
        args - Namespace object from the argument parser
    Outputs:
        examples - List of development set examples
    """

    # Get the learning method
    learning_method = learning_method_index[args.learning_method]

    # Load data features
    print("Creating features from dataset file at {}".format(args.data_dir))
    processor = processors["mutual"]()
    label_list = processor.get_labels()
    dev_examples = processor.get_dev_examples(args.data_dir)
    train_examples = processor.get_train_examples(args.data_dir)

    # Limit the data if needed
    if learning_method == 0:
        train_examples = []
    elif learning_method == -1:
        pass
    else:
        train_examples = train_examples[:learning_method]

    # Print the number of examples
    print("Number of train instances: {}".format(len(train_examples)))
    print("Number of evaluation instances: {}".format(len(dev_examples)))

    # Return the items
    return train_examples, dev_examples


def batch_dataset(tokenizer, iterable, n=1):
    """
    Function for batching the data.
    Inputs:
        tokenizer - Huggingface Tokenizer instance
        iterable - Iterable object containing the data
        n - Batch size
    Outputs:
        batches - List of batches
    """
    
    batches = []

    l = len(iterable)
    for ndx in range(0, l, n):
        batch = iterable[ndx:min(ndx + n, l)]

        # Tokenize the batch
        contexts = [example.context for example in batch]
        endings = [example.endings[reverse_answer_index[example.label]] for example in batch]
        zipped_input = list(zip(contexts, endings))
        tokenized_batch = tokenizer.batch_encode_plus(
            zipped_input,
            add_special_tokens=True,
            truncation='only_first',
            padding=True,
            return_tensors="pt",
        )

        # Add the batch to the list
        batches.append(tokenized_batch)
    
    # Return the batches
    return batches


def calculate_eval_metrics(full_predictions, real_labels):
    """
    Function for calculating the evaluation metrics.
    Inputs:
        full_predictions - List of ordered lists containing the predictions
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
    # Freeze the model except the last layer
    for param in model.transformer.parameters():
        param.requires_grad = False
    print(model)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_index[args.model])
    tokenizer.pad_token = tokenizer.eos_token
    print('Model loaded')

    # Load the data
    print('Loading data..')
    train_dataset, eval_dataset = load_examples(args)
    print('Data loaded')

    # Train the model
    if len(train_dataset) == 0:
        print('Skipping training')
    else:
        print('Starting training..')
        model.train()
        optimizer = AdamW(model.parameters())
        # Batch the dataset
        batches = batch_dataset(tokenizer, train_dataset, args.batch_size)
        for batch in batches:
            model.zero_grad()

            # Pass through the model
            batch.to(device)
            outputs = model(**batch, labels=batch['input_ids'])
            loss = outputs.loss

            # Optimize
            loss.backward()
            optimizer.step()
        print('Training finished')

    # Evaluate the model
    print('Starting evaluation..')
    example_ids = []
    predicted_labels = []
    real_labels = []
    full_predictions = []
    model.eval()
    for example in eval_dataset:
        scored_options = []
        # Create all possible options
        for ending_index, ending in enumerate(example.endings):
            with torch.no_grad():
                # Pass through the model
                inputs = tokenizer.encode_plus(
                    example.context,
                    ending,
                    add_special_tokens=True,
                    truncation='only_first',
                    return_tensors="pt",
                )
                inputs.to(device)
                outputs = model(**inputs, labels=inputs['input_ids'])
                perplexity = outputs.loss
                option_letter = answer_index[ending_index]
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
    df = pd.DataFrame(list(zip(example_ids, predicted_labels, real_labels, full_predictions)), columns=['example_id', 'predicted_label', 'real_label', 'full_ordering'])
    df['correct'] = np.where(df['predicted_label'] == df['real_label'], True, False)
    df.to_csv(args.output_dir + args.model + '_results.csv')

    # Calculate the evaluation metrics
    R4_1, R4_2, mrr = calculate_eval_metrics(full_predictions, real_labels)

    # Save the metrics in a .csv file
    df = pd.DataFrame([[R4_1, R4_2, mrr]], columns=['Recall@1', 'Recall@2', 'MRR'])
    df.to_csv(args.output_dir + args.model + '_metrics.csv')

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

    # Set a random seed
    torch.seed()

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print the given parameters
    print('-----EVALUATION PARAMETERS-----')
    print('Device: {}'.format(device))
    print('Model: {}'.format(args.model))
    print('Batch size: {}'.format(args.batch_size))
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
                        choices=['gpt2', 'bart', 'gpt_neo', 'dialog_gpt', 'xlnet', 'xlprophetnet', 'blenderbot'])
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size to use during training. Default is 8.')
    parser.add_argument("--data_dir", default='data/mutual', type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task. Default is data/mutual.")
    parser.add_argument("--output_dir", default='experiment_outputs/', type=str,
                        help="The output directory for the .csv files. Default is experiment_outputs/.")
    parser.add_argument('--learning_method', default='zero_shot', type=str,
                        help='Learning method to use. Default is zero_shot.',
                        choices=['zero_shot', '1_shot', '10_shot', '100_shot', '1000_shot', 'full_dataset'])

    # Parse the arguments
    args = parser.parse_args()

    # Evaluate the model
    main(args)