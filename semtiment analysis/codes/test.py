import argparse
import yaml
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from utils import get_dataloader_vocab_embedding
from train import Trainer
from model import LSTM_net


def train_and_test(config):
    '''
    Function to control the training process
    '''

    batch_size = config['batch_size']
    dataset_path = config['dataset_path']
    embedding_path = config['embedding_path']
    vocab, transformed_embedding, train_loader, val_loader, test_loader = get_dataloader_vocab_embedding(batch_size,
                                                                                                         dataset_path,
                                                                                                         embedding_path)
    torch.save(transformed_embedding, os.path.join(dataset_path, "transformed_embedding.pt"))
    torch.save(vocab, os.path.join(dataset_path, "vocab.pt"))

    embed_size = transformed_embedding.shape[1]
    vocab_size = vocab.__len__()
    hidden_dim = config['hidden_dim']
    num_layers = config['num_layers']
    dropout = config['dropout']
    model_name = config['model_name']
    model_dir = config['model_dir']

    torch.random.manual_seed(21)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain = config['pretrain']
    if pretrain:
        model = LSTM_net(transformed_embedding,
                         embed_size,
                         vocab_size,
                         hidden_dim,
                         num_layers,
                         dropout)
    else:
        model = LSTM_net(None,
                         embed_size,
                         vocab_size,
                         hidden_dim,
                         num_layers,
                         dropout)

    criterion = nn.CrossEntropyLoss()
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    weight_decay = float(config['weight_decay'])
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.5, verbose=True)
    lr_decay = config['lr_decay']

    trainer = Trainer(model=model,
                      epochs=epochs,
                      train_dataloader=train_loader,
                      val_dataloader=val_loader,
                      test_dataloader=test_loader,
                      criterion=criterion,
                      optimizer=optimizer,
                      lr_decay=lr_decay,
                      lr_scheduler=lr_scheduler,
                      device=device,
                      model_dir=model_dir,
                      model_name=model_name)

    trainer.train()
    trainer.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the config')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    train_and_test(config)
