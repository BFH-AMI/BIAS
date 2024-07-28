# The BIAS Project - Measuring Real-World Biases in Word Embeddings and Language Models in European Languages 

This repository contains partial outcomes of the Horizon Europe Project BIAS https://www.biasproject.eu/.

*Important Note:* This repository will be updated soon with additional results and examples to explore the full functionality. 

**Reference:** When using this repository, please cite the following pre-print: 
Alexandre Puttick, Leander Rankwiler, Catherine Ikae and Mascha Kurpicz-Briki (2024). The BIAS Detection Framework: Bias Detection in Word Embeddings and Language Models for European Languages. Arxiv Pre-Print.
**TODO: Add pre-print link.**

Other papers based on this repository:

| Paper | Language(s) | Subfolder |
|---|---|---|
| Rankwiler, L. and Kurpicz-Briki, M. (2024). Evaluating Labor Market Biases Reflected in German Word Embeddings. Accepted for publication at SwissText 2024. | German  | BiasDetection/datasets/de |

More resources will be added soon.

# Funding 
This work is part of the Europe Horizon project BIAS funded by the European Commission, and
has received funding from the Swiss State Secretariat for Education, Research and Innovation
(SERI).

# Installation Guide 

Install system packages 
```
sudo apt install python3 python3-pip
sudo apt install virtualenv
```

Checkout the repository 

cd into the main folder of the repository

Create and active virtualenv
```
python3 -m virtualenv venv
source venv/bin/activate
```

Install dependencies
```
pip install -r requirements.tx`
```

If you do not already have the BERT model saved in your models folder, if will be downloaded automatically. To handle BERT, you will need the libraries `transformers` and `torch`.

If you want to run experiments with fasttext, GloVe or word2vec, follow the following instructions:

### fasttext
1. Download the zipped binary file from: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz (6GB)
2. Unzip the file
    * For Linux and macOS: Use `gunzip filename.gz` in the terminal.
    * For Windows: Unzip with software like 7-Zip or Winrar.
    * For Windows: You can also use Git CMD shell or WindowsSubsystemLinux to use gunzip (`gunzip filename.gz`) as with Linux  
3. Make sure the file is named `cc.en.300.bin`, or change `config.py` accordingly

To handle fasttext, you will need the libary `fasttext`. For conda you can get the library with `conda install -c esri fasttext`

# Getting Started Guide

**Step 1:** The experiments are selected in Experiments.py.
Here you can choose a subfolder of the experiments/ folder, where your experiments are located and a name for the run. 

(extract from Experiments.py)
```
### Select here your experiments configuration folder, see an example in folder experiments/example
folder_path = 'experiments/example/'
```

In Experiments.py, you can also select the following values for the experiment: 
```
### Experiment Configurations
pval = True # compute p-value along with the effect size
iterations = 100 # number of iterations (e.g., 100 for testing, 10K for experiments)
fullpermut = False # do not use, currently not implemented 
``` 

**Step 2:** Give your experiment a name, this will be used for the folder where the results are stored
experiment_name = "test1"

**Step 3:** In the subfolder of experiments, you can create several .txt files as experiments. For example, consider the example file located in experiments/example/experimentConfigExample.txt

```
###
# You can add all the experiments that you want to execute in such a .txt file.
# All lines starting with # are ignored.
# Each line should have the following format: datasetname,testname,embeddings,language,modelname
# The modelname is required for BERT experiments and can be set to None for the other experiments.
SEAT_WEAT_7,bert,en,google-bert/bert-base-uncased
CW1,WEAT,fasttext,de,None
```

The following table gives an overview of the options for the different values mentioned above: 

(*Note*: Examples and/or code for some configurations are missing in the current release, will be updated soon!)

| Name | Possible Values |
|----------|----------|
| dataset | Available datasets for the corresponding language can be found in datasets/<language>. For example, available WEAT tests for English are stored in datasets/en/WEAT/*.txt. You would use here the filename without the .txt |
| testname | WEAT, SEAT, LPBS, CROWS (refer to paper (**LINK**) for details about the different tests) |
| embeddings | fasttext, word2vec, glove, mBERT, bert, bert_pooling, bert_first (refer to paper (**LINK**) for details about the different embeddings) |
| languages | Available languages depending on the datasets and model selection (de=German, en=English, no=Norwegian, is=Icelandic, it=Italian, ne=Dutch, tr=Turkish) |
| modelname | Set to None for fasttext, word2vec, glove, mBERT. For BERT, select the corresponding model name, e.g., for English bert-base-uncased, or for Norwegian ltg/norbert3-base **TODO: Integrate OpenAI and VoyageAPI Documentation** |

**Step 4:** Running the experiments 
```
./venv/bin/python Experiments.py
```

In case you want to keep running the experiments in background:
```
nohup ./venv/bin/python Experiments.py &
``` 

After execution, you find the results in a corresponding subfolder of the results/ folder, including execution logs, code for Latex Tables on the results, and a plot for each experiment.
```
ls results/test1/
```
