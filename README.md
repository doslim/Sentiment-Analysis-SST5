# Sentiment Analysis SST-5

This repository contains a LSTM model implemented by PyTorch to perform sentiment classification on the [Stanford Sentiment Treebank (SST-5)](https://nlp.stanford.edu/sentiment/) dataset. We train the model with/without pretrained embeddings and conduct several experiments on different hyperparameters. Since the SST-5 dataset contains sentiment labels for each token in sentences, we develop a modified model to utilze this information.

We use the [300-dim GloVe embeddings from 6B tokens](https://nlp.stanford.edu/projects/glove/) and provide a report to introduce the implementation details and evaluation results.

## Project Structure and Environments
The required environments are as follows.
- torch-1.11.0
- torchtext-0.12.0
- pytreebank-0.2.7: used to load the datasets.
  
The structure of our projects is as follows.
- sentiment analysis: basic version.
    - /codes: contain all codes.
        - test.py: the entrance of our project.
        - train.py: define the class to train the model.
        - model.py: define the model.
        - utils.py: load the data for training and evaluations.
        - config.yaml: store the configurations for model trianing.
    - /weight: the directory to save models, training logs and results.
    - /data: the directory of the datasets and pretrained embeddings.
- improved: improved version.
    - /codes: contain all codes.
        - test\_improved.py: the entrance of the improved model.
        - train\_improved.py: define the class to train the model.
        - model\_improved.py: define the improved model.
        - utils\_improved.py: load the data for training and evaluations.
        - config.yaml: store the configurations for model trianing.
    - /weight: the directory to save models and training logs.
    - /data: the directory of corpus used in training and evaluation.
- report.pdf: a brief introduction of our implementation details, the improved model and evaluation results.

## Usage
First download the [Stanford Sentiment Treebank (SST-5)](https://nlp.stanford.edu/sentiment/) dataset and the [pretrained embeddings](https://nlp.stanford.edu/projects/glove/) into the /data directory. Then unzip them.

After that, you can use the following command to run our codes in the /codes directory.
```
python test.py --config=config.yaml
```
The meaning of each configuration can be found in our report.
