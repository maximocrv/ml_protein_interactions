"""
This script contains the functions used for training in the different models
"""


def train(_model, _criterion, dataset_train, dataset_test, _optimizer,
          n_epochs, device, test_metrics):
    print("Starting training")
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
        mse_test = []
        for batch_x, batch_y in dataset_test:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            # TODO: solve allocation error in GPU
            prediction = _model(batch_x)
            mse_test.append(_criterion(prediction, batch_y).item())
            # ^^ could be source of memory leak
        print("Epoch {} | Train loss: {:.5f} Test loss: {:.5f}".format(
            epoch, train_loss, sum(mse_test)/len(mse_test)))
