# utils.py
# Define functions to load the data

import pytreebank
import csv
import numpy as np
import pandas as pd
import torch
import os
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset


def get_english_tokenizer():
    """
    Documentation:
    https://pytorch.org/text/stable/_modules/torchtext/data/utils.html#get_tokenizer
    """
    tokenizer = get_tokenizer("basic_english", language="en")
    return tokenizer


def data_transform(dataset):
    '''
    Parse the labels and sentences from treebank dataset.
    We also

    parameters:
    - dataset: a list of pytreebank.labeled_trees.LabeledTree.

    return:
    - data: a list of sentences after word segmentation.
    - labels: a list of sentiment labels.
    '''

    tokenizer = get_english_tokenizer()
    labels = []
    data = []
    for sample in dataset:
        sample = sample.to_labeled_lines()[0]
        labels.append(sample[0])
        data.append(tokenizer(sample[1]))

    return data, labels


def build_vocab(data, min_freq=1):
    vocab = build_vocab_from_iterator(
        data,
        specials=["<unk>"],
        min_freq=min_freq
    )
    vocab.set_default_index(vocab["<unk>"])

    return vocab


def load_glove(path="glove.6B.300d.txt"):
    '''
    Load the pretrained glove embedding

    parameters:
    - path: the local path to load Glove embeddings.

    return:
    - words: vocabulary that contains all words.
    - embedding: corresponding word embeddings.
    '''

    words = pd.read_table(path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

    embedding = words.values
    words = words.index.to_numpy()

    print("The embedding size is {}, and the number of words is {}".format(embedding.shape[1], embedding.shape[0]))
    return words, embedding


def embedding_transform(vocab, words, embedding):
    '''
    Get a subset of pre-trained embeddings according to the small dataset.

    parameters:
    - vocab: torchtext.vocab.vocab, is the vocabulary of the small dataset.
    - words: vocabulary of the pre-trained embeddings.
    - embedding: embeddings corresponding to words.

    return:
    - transformed_embedding: embedding corresponding to vocab.
    - unknown_words: words that are out of the pre-trained vocabulary.
    '''

    word_to_idx = vocab.get_stoi()
    num_words = vocab.__len__()
    embedding_size = embedding.shape[1]
    transformed_embedding = np.zeros((num_words, embedding_size))
    unk_id = np.argwhere(words == '<unk>').item()

    unknown_words = []
    for token in word_to_idx.keys():
        idx = word_to_idx[token]
        if token in words:
            token_id = np.argwhere(words == token).item()
            transformed_embedding[idx,] = embedding[token_id,]
        else:
            transformed_embedding[idx,] = embedding[unk_id,]
            unknown_words.append(token)

    return transformed_embedding, unknown_words


def sentence_to_idx(data, vocab, MAX_SEQUENCE_LENGTH=25):
    '''
    Transform the words into index

    parameters:
    - data: a list of sentences after word segmentation.
    - vocab: vocabulary corresponding to data.

    return:
    - transformed_data: a list of sentences of which words are represented by ids.
    '''

    pad_id = vocab.__len__() - 1
    data_size = len(data)
    transformed_data = np.zeros((data_size, MAX_SEQUENCE_LENGTH))

    for i in range(data_size):
        sentence = data[i]
        len_sen = len(sentence)
        sentence = sentence[:MAX_SEQUENCE_LENGTH]
        word_ids = [vocab[i] for i in sentence]

        if len_sen < MAX_SEQUENCE_LENGTH:
            padding_length = MAX_SEQUENCE_LENGTH - len_sen
            word_ids.extend([pad_id] * padding_length)

        transformed_data[i,] = np.array(word_ids, dtype=np.int64)

    return transformed_data


def add_pad_embedding(embedding, seed=21):
    '''
    Add a random embedding for <PAD>

    parameters:
    - embedding: a 2D numpy array with size of NUM_WORDS * EMBEDDING_SIZE that contains all word embeddings.

    return:
    - new embedding.
    '''

    embedding_dim = embedding.shape[1]
    np.random.seed(seed)
    pad_embedding = np.random.rand(1, embedding_dim)

    return np.concatenate((embedding, pad_embedding), axis=0)


class MyDataset(Dataset):

    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


def search_file(file_name, search_path, pathsep = os.pathsep):

    for path in search_path.split(pathsep):
        candidate = os.path.join(path, file_name)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        return None


def get_dataloader_vocab_embedding(batch_size=64, dataset_path='./trees', embedding_path='./glove.6B.300d.txt'):

    # Load dataset
    dataset = pytreebank.load_sst(dataset_path)
    train = dataset['train']
    val = dataset['dev']
    test = dataset['test']

    # data transformation
    train_data, train_labels = data_transform(train)
    val_data, val_labels = data_transform(val)
    test_data, test_labels = data_transform(test)

    # label transformation
    train_labels = torch.LongTensor(train_labels)
    val_labels = torch.LongTensor(val_labels)
    test_labels = torch.LongTensor(test_labels)

    if search_file('vocab.pt', dataset_path):
        vocab = torch.load(os.path.join(dataset_path, "vocab.pt"))
        print("Load vocab directly.")
    else:
        # build vocabulary
        vocab = build_vocab(train_data)
        # vocab add <pad>
        vocab.insert_token("<PAD>", vocab.__len__())
        print("Build vocab.")
    print("The vocabulary contains {} words.".format(vocab.__len__()))

    if search_file("transformed_embedding.pt", dataset_path):
        transformed_embedding = torch.load(os.path.join(dataset_path, "transformed_embedding.pt"))
        print("Load transformed embeddings directly.")
    else:
        # load pre-trained embedding
        words, embedding = load_glove(embedding_path)

        # get the embedding of our vocabulary
        transformed_embedding, unknown_words = embedding_transform(vocab, words, embedding)
        print("There are {} words out of the pre-trained vocabulary.".format(len(unknown_words)))
        # add <pad> embedding
        transformed_embedding = add_pad_embedding(transformed_embedding)
        transformed_embedding = torch.tensor(transformed_embedding, dtype=torch.float)

    # transform the words into ids
    train_data = sentence_to_idx(train_data, vocab)
    val_data = sentence_to_idx(val_data, vocab)
    test_data = sentence_to_idx(test_data, vocab)

    # build datasets
    train_dataset = MyDataset(torch.tensor(train_data, dtype=torch.long), train_labels)
    val_dataset = MyDataset(torch.tensor(val_data, dtype=torch.long), val_labels)
    test_dataset = MyDataset(torch.tensor(test_data, dtype=torch.long), test_labels)

    # build dataloaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

    return vocab, transformed_embedding, train_loader, val_loader, test_loader











