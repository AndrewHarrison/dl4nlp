import os
import argparse
import numpy as np
import pandas as pd
import re
from evaluate_nlg import load_examples, learning_method_index
from cluster_data import data_partition


cluster_dict = {
  0: "Short",
  1: "Medium",
  2: "Long"
}

def convert(string):
    """This method receives a string and returns a list of characters representing the dialogue answers returned in order by the model

    Args:
        string (str): String representing the chosen answers

    Returns:
        list: List of characters containing each answer separately
    """

    chars = re.sub('[^a-zA-Z]+', '', string)
    return list(chars)

def eval_data_partitions(args, eval_dataset_concat):
    """This method evaluates one architecture with multiple fine-tuning approaches on one specified data partition.
    It returns recall scores that are further analyzed to conclude wether the method tends to have better predictions
    for certain types of data.

    Args:
        args (Namespace): Object from the argument parser
        eval_dataset_concat (list): Concatenated evaluation dataset
    """

    cluster_values = set(map(lambda x:x.cluster, eval_dataset_concat))
    partitioned_dataset = [[example for example in eval_dataset_concat if example.cluster==x] for x in cluster_values]
    number_partitions = len(partitioned_dataset)

    for shot in args.few_shots:

        print(shot)
        total_mr1, total_mr2, total_mrr = np.zeros(number_partitions), np.zeros(number_partitions), np.zeros(number_partitions)

        for run in args.runs:

            run_mr1, run_mr2, run_mrr = [], [], []

            file_path = "experiment_outputs\\" + args.model + "_" + shot + run + args.model + "_results.csv"
            file_path = os.path.join(os.path.abspath(os.getcwd()), file_path)
            all_evaluation_results = pd.read_csv(file_path)

            for cluster in cluster_values:

                tp_1, tp_2, rr = 0, 0, 0
                len_cluster = len(partitioned_dataset[cluster])

                for example in partitioned_dataset[cluster]:

                    evaluation_results = all_evaluation_results.loc[all_evaluation_results['example_id'] == example.example_id.replace('\\', '/')]

                    real_label = str(evaluation_results.real_label.item())
                    predicted_labels = convert(evaluation_results.full_ordering.item())

                    if real_label == predicted_labels[0]:
                        tp_1 += 1

                    if real_label in predicted_labels[:2]:
                        tp_2 += 1

                    position = predicted_labels.index(real_label)
                    rr += 1/(position + 1)

                run_mr1.append(tp_1/len_cluster)
                run_mr2.append(tp_2/len_cluster)
                run_mrr.append(rr/len_cluster)

            total_mr1 += np.array(run_mr1)  
            total_mr2 += np.array(run_mr2)  
            total_mrr += np.array(run_mrr)  

        for cluster in cluster_values:
            print('Cluster', cluster, ":", cluster_dict[cluster])

        print("TOTAL MR1: ", np.round(total_mr1/3, decimals = 3))
        print("TOTAL MR2: ", np.round(total_mr2/3, decimals =3))
        print("TOTAL MRR: ", np.round(total_mrr/3, decimals =3), '\n')

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
    eval_data_partitions(args, eval_dataset_concat)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data_dir", default='data/mutual', type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task. Default is data/mutual.")
    parser.add_argument("--partition", default='tf-idf', type=str,
                        help="Types of data partition. Can be one of the following: [length, token_length, tf-idf]/.")
    parser.add_argument('--model', default='gpt2', type=str,
                        help='Generator model type to use. Default is gpt2.',
                        choices=['gpt2', 'bart', 'gpt_neo', 'dialog_gpt', 'xlnet', 'xlprophetnet', 'blenderbot'])
    parser.add_argument('--learning_method', default='zero_shot', type=str,
                        help='Learning method to use. Default is zero_shot.',
                        choices=['zero_shot', '1_shot', '10_shot', '100_shot', '1000_shot', 'full_dataset'])
    parser.add_argument("--few_shots", default= ['zero_shot', '1_shot', '10_shot', '1_epoch', '5_epoch', '10_epoch'], type=list,
                        help="Multiple fine-tuning approaches.")
    parser.add_argument("--runs", default=['\Run_1\\', '\Run_2\\', '\Run_3\\'], type=list,
                        help="Several runs for the same method.")

    args = parser.parse_args()

    main(args)
