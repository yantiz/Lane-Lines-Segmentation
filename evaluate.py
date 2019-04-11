"""Evaluates the model"""

import configparser
import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import utils
import model.net as net
import model.data_loader as data_loader

config = configparser.RawConfigParser()
config.read_file(open(r'./configuration.txt'))

parser = argparse.ArgumentParser()
model_dir = config.get('model paths', 'model_dir')
parser.add_argument('--model_dir', default=model_dir, help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()

        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch tensor, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
        
    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    data_dir = config.get('data paths', 'data_dir')
    dataloaders = data_loader.fetch_dataloader(['test'], data_dir, params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    input_channels = int(config.get('settings', 'input_channels'))
    output_channels = int(config.get('settings', 'output_channels'))
    model = net.Net(input_channels, output_channels).cuda() if params.cuda else net.Net(input_channels, output_channels)
    
    loss_fn = nn.CrossEntropyLoss()
    metrics = net.metrics
    
    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
