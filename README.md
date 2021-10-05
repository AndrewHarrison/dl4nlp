# dl4nlp

Project for Deep Learning for Natural Language Processing (second year master AI @ UvA)

This repository contains research on the ability of Natural Language Generation (NLG) models to capture conversational reasoning capabilities in multi-turn dialogue on the MuTual dataset. The goal of the research is to examine the zero-shot performance of state-of-the-art NLG models on the MuTual dataset.

## Content

This repository consists of the following key scripts and folders:

- **evaluate_nlg.py**: this is the core script of the research. It zero-shot evaluates the NLG models on the MuTual dataset.
- **utils_nlg.py**: copied utils script from the original [MuTual Github](https://github.com/Nealcly/MuTual) with some functionality removed and altered for the NLG models.
- **cluster_data.py**: TODO, add description of functionality.
- **data**: folder containing all the data. This is a direct copy from the original [MuTual Github](https://github.com/Nealcly/MuTual).
- **experiment_outputs**: folder containing all the results from our experiment runs.

The accompanying short report of this project can be found in this repository as **Research_Report.pdf**.

## Prerequisites

- Anaconda. Available at: https://www.anaconda.com/distribution/

## Getting Started

1. Open Anaconda prompt and clone this repository (or download and unpack zip):

```bash
git clone https://github.com/AndrewHarrison/dl4nlp.git
```

2. Create the environment:

```bash
conda env create -f environments/environment.yml
```

Or use the Lisa environment when running on the SurfSara Lisa cluster:

```bash
conda env create -f environments/environment_lisa.yml
```

3. Activate the environment:

```bash
conda activate dl4nlp
```

4. Run the NLG evaluation script for GPT2:

```bash
python evaluate_nlg.py
```

Or provide the name of another model:

```bash
python evaluate_nlg.py --model MODEL
```

## Dataset

All data is available in the **data** folder of this repository. It is a direct copy from the original [MuTual Github](https://github.com/Nealcly/MuTual).

## Arguments

The NLG models can be evaluated with the following command line arguments:

```bash
usage: evaluate_nlg.py [-h] [--model MODEL] [--data_dir DATA_dir] [--output_dir OUTPUT_DIR]

optional arguments:
  -h, --help            			Show help message and exit.
  --model MODEL			          What model to use. Options: ['gpt2', 'bart', 'gpt_neo', 'dialog_gpt', 'xlnet', 'xlprophetnet', 'blenderbot']. Default is 'gpt2'.
  --data_dir DATA_DIR				  Directory where the data is stored. Default is 'data/mutual'.
  --output_dir OUTPUT_DIR			Directory where the evaluation results are stored as csv files. Default is 'experiment_outputs/'.
```

## Authors

- Luuk Kaandorp - luuk.kaandorp@student.uva.nl
- Andrew Harrison - andrew.harrison@student.uva.nl
- Roxana Petcu - roxana.petcu@student.uva.nl

## Acknowledgements

- Data and experimental code have been copied and adapted from the original [MuTual Github](https://github.com/Nealcly/MuTual).
