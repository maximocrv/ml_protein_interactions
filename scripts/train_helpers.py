"""
This script contains the functions used for training in the different models.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def train(_model: torch.nn.Module, _criterion: torch.nn.MSELoss, dataset_train: DataLoader, dataset_test: DataLoader,
          _optimizer: torch.optim.Optimizer, n_epochs: int, device: torch.device, test_metrics: dict, log):
    """
    Trains the given model according to a certain criterion and logging
    the given test metrics.

    :param _model: Model to be trained
    :param _criterion: Criterion to use for training (e.g. RMSE)
    :param dataset_train: Dataset used for training
    :param dataset_test: Dataset used for testing/validation
    :param _optimizer:  Optimizer to use for training (e.g. Adam)
    :param n_epochs: Number of epochs
    :param device: Device to use for training (e.g. CPU or CUDA)
    :param test_metrics: Python dictionary with keys equal to the name of the test metric
                            to use and values equal functions of the form:
                                            fun(y_real, y_pred) -> float
    :param log: Handler to log information to an output file
    :return:  Final train loss and test scores/losses
    """
    # output
    print("Starting training")
    log.write("\tStarting training\n")
    log.write('\n')
    log.write("epoch\ttrain_loss\t{}\n".format('\t'.join(test_metrics.keys())))

    for epoch in range(n_epochs):
        # Train an epoch
        _model.train()
        train_losses = []
        for batch_x, batch_y in dataset_train:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            logits = _model.forward(batch_x)
            loss = _criterion(logits, batch_y)
            train_losses.append(loss.item())

            # Compute the gradient
            _optimizer.zero_grad()
            loss.backward()

            # Update the parameters of the model with a gradient step
            _optimizer.step()

        train_loss = np.mean(train_losses)

        # Evaluate the quality of the model and store the scores/losses
        _model.eval()
        eval_scores = {key: [] for key in test_metrics}
        n = 0
        for batch_x, batch_y in dataset_test:
            # To device
            batch_x = batch_x.to(device)

            # Evaluate the network (forward pass)
            prediction = _model(batch_x)

            # Back to host
            prediction = prediction.cpu()

            # Fix tensors to be used with test metrics
            prediction = prediction.detach().numpy().ravel()
            batch_y = batch_y.detach().numpy().ravel()

            # Weighted sum of scores according to batch_y length
            for key, fun in test_metrics.items():
                eval_scores[key].append(
                    len(batch_y) * fun(batch_y, prediction)
                )

            n += len(batch_y)

        eval_scores = {key: sum(eval_scores[key]) / n for key in test_metrics}

        # Output epoch information
        eval_scores_str = \
            ' '.join([f'{k}={v:12.5g}' for k, v in eval_scores.items()])
        print("Epoch {} | Train loss: {:12.5g} Validation scores: {}".format(
            epoch, train_loss, eval_scores_str))
        log.write("{}\t{:12.5g}\t{}\n".format(
            epoch, train_loss,
            '\t'.join([f'{v:12.5g}' for v in eval_scores.values()])
        ))

    return train_loss, eval_scores


def train_adaptive(_model: torch.nn.Module, _criterion: torch.nn.MSELoss, dataset_train: DataLoader,
                   dataset_test: DataLoader, _optimizer: torch.optim.Optimizer, device: torch.device,
                   test_metrics: dict, log, best_heuristic=lambda x, y: True, n_epochs: int = 10):
    """
    Trains the given model according to a certain criterion and logging
    the given test metrics. This training is done so that it stops only after
    `n_epochs` consecutive epochs that did not give a better score according
    to the heuristic of `best_heuristic`.

    ! ATTENTION: this function does NOT give the best trained model. Rather,
    the last epoch's trained model.

    :param _model: Model to be trained
    :param _criterion: Criterion to use for training (e.g. RMSE)
    :param dataset_train: Dataset used for training
    :param dataset_test: Dataset used for testing/validation
    :param _optimizer:  Optimizer to use for training (e.g. Adam)
    :param device: Device to use for training (e.g. CPU or CUDA)
    :param test_metrics: Python dictionary with keys equal to the name of the test metric to use and values equal
           functions of the form: fun(y_real, y_pred) -> float
    :param log: Handler to log information to an output file
    :param best_heuristic: lambda function, optional lambda function of the form: lambda x, y -> boolean. The inputs are
           the overall best evaluation scores and the current epoch's evaluation scores.
           If the output is `True`, the current evaluation score replaces the overall one. Otherwise, no change is
           performed. By default, this function always returns `True`.
    :param n_epochs: Max number of consecutive epochs allowed without achieving a better evaluation score. Defaults to
    10.
    :return: Dictionaries containing best train loss across all epochs, and the best evaluation scores/losses across all
             epochs.
    """
    # Initial output
    print("Starting training")
    log.write("\tStarting training\n")
    log.write('\n')
    log.write("epoch\ttrain_loss\t{}\n".format('\t'.join(test_metrics.keys())))

    best_eval = None
    best_epoch = 0
    best_train = None

    best_counter = 0
    epoch = -1
    while best_counter < n_epochs:
        best_counter += 1
        epoch += 1

        # Train an epoch
        _model.train()
        train_losses = []
        for batch_x, batch_y in dataset_train:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            logits = _model.forward(batch_x)
            loss = _criterion(logits, batch_y)
            train_losses.append(loss.item())

            # Compute the gradient
            _optimizer.zero_grad()
            loss.backward()

            # Update the parameters of the model with a gradient step
            _optimizer.step()

        train_loss = np.mean(train_losses)

        # Evaluate the quality of the model and store the scores/losses
        _model.eval()
        eval_scores = {key: [] for key in test_metrics}
        n = 0
        for batch_x, batch_y in dataset_test:
            # To device
            batch_x = batch_x.to(device)

            # Evaluate the network (forward pass)
            prediction = _model(batch_x)

            # Back to host
            prediction = prediction.cpu()

            # Fix tensors to be used with test metrics
            prediction = prediction.detach().numpy().ravel()
            batch_y = batch_y.detach().numpy().ravel()

            # Weighted sum of scores according to batch_y length
            for key, fun in test_metrics.items():
                eval_scores[key].append(
                    len(batch_y) * fun(batch_y, prediction)
                )

            n += len(batch_y)

        eval_scores = {key: sum(eval_scores[key]) / n for key in test_metrics}

        if best_eval is None or best_heuristic(best_eval, eval_scores):
            best_eval = eval_scores
            best_epoch = epoch
            best_train = train_loss
            best_counter = 0

        # Output epoch information
        if epoch % 5 == 0:
            eval_scores_str = \
                ' '.join([f'{k}={v:12.5g}' for k, v in eval_scores.items()])
            print("Epoch {} | Train loss: {:12.5g} Validation scores: {}".format(
                epoch, train_loss, eval_scores_str))
        log.write("{}\t{:12.5g}\t{}\n".format(
            epoch, train_loss,
            '\t'.join([f'{v:12.5g}' for v in eval_scores.values()])
        ))

    # Output best scores
    eval_scores_str = \
        ' '.join([f'{k}={v:12.5g}' for k, v in best_eval.items()])
    print("BEST - Epoch {} | Train loss: {:12.5g} Validation scores: {}".format(
        best_epoch, best_train, eval_scores_str))
    log.write("BEST - {}\t{:12.5g}\t{}\n".format(
        best_epoch, best_train,
        '\t'.join([f'{v:12.5g}' for v in best_eval.values()])
    ))

    return best_train, best_eval


def gen_loaders(x, y, batch_size):
    """
    Generates the batch laoders to use in the train loop.

    :param x: Input features
    :param y: Target values
    :param batch_size: Batch size
    :return: DataLoader object which is used as an iterator during training.
    """
    if type(x) == np.ndarray:
        x_tensor = torch.from_numpy(x)
    else:
        x_tensor = x
    if type(y) == np.ndarray:
        y_tensor = torch.from_numpy(y)
    else:
        y_tensor = y

    data = TensorDataset(x_tensor, y_tensor)

    loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False)

    return loader
