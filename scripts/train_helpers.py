"""
This script contains the functions used for training in the different models
"""

import numpy as np


def train(_model, _criterion, dataset_train, dataset_test, _optimizer,
          n_epochs, device, test_metrics, log):
    r"""Trains the given model according to a certain criterion and logging
    the given test metrics.

    Parameters
    ----------
    _model : torch.nn.Module
        model to be trained
    _criterion : torch.nn._Loss
        criterion to use for training (for ex.: RMSE)
    dataset_train : DataLoader
        dataset used for training
    dataset_test : DataLoader
        dataset used for testing/validation
    _optimizer : torch.optim.Optimizer
        optimizer to use for training (for ex.: Adam)
    n_epochs : int
        number of epochs
    device : torch.device
        device to use for training (for ex.: CPU or CUDA)
    test_metrics : dict
        python dictionary with keys equal to the name of the test metric
        to use and values equal functions of the form:
            fun(y_real, y_pred)
        returning a float value
    log : _io.TextIOWrapper
        handler to log information to an output file.

    Returns
    -------
    train_loss : float
        final train loss
    eval_scores : dict
        final test scores/losses
    """
    print("Starting training")
    log.write("\tStarting training\n")
    log.write('\n')
    log.write("epoch\ttrain_loss\t{}".format('\t'.join(test_metrics.keys())))
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

        # Test the quality on the test set
        _model.eval()
        eval_scores = {key: [] for key in test_metrics}
        n = 0
        for batch_x, batch_y in dataset_test:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            # TODO: solve allocation error in GPU
            prediction = _model(batch_x)

            # weighted sum of scores according to batch_y length
            for key, fun in test_metrics.items():
                eval_scores[key].append(
                    len(batch_y)*fun(batch_y, prediction)
                )

            n += len(batch_y)

        eval_scores = {key: sum(eval_scores[key])/n for key in test_metrics}
        eval_scores_str = \
            ' '.join([f'{k}={v:12.5g}' for k, v in eval_scores.items()])
        print("Epoch {} | Train loss: {:12.5g} Test scores: {:}".format(
            epoch, train_loss, eval_scores_str))
        log.write("{}\t{:12.5g}\t{}\n".format(
            epoch, train_loss,
            '\t'.join([f'{v:12.5g}' for v in eval_scores.values()])
        ))

    return train_loss, eval_scores
