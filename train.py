import os
from datetime import datetime
import json
from itertools import cycle

import torch

# PyTorch TensorBoard support
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from dataset import _create_dataloader
from model.lenet import LeNet5

def train_one_epoch(model, optimizer, training_loader, loss_fn, epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def _compute_loss_and_accuracy(logits: torch.Tensor, labels: torch.Tensor, loss_fn: torch.nn.modules.loss._Loss):
    loss = loss_fn(logits, labels)
    outputs = torch.argmax(logits, dim=1).double()
    accuracy = (outputs == labels).float().sum() * 100 / len(labels)

    return loss, accuracy

def train_one_iteration(data, optimizer, scheduler, model, loss_fn):
    start_time = datetime.now()
    inputs, labels = data
    optimizer.zero_grad()
    logits = model(inputs)
    loss, accuracy = _compute_loss_and_accuracy(logits, labels, loss_fn)
    loss.backward()
    optimizer.step()
    scheduler.step()
    loss = loss.item()

    return loss, accuracy, (datetime.now() - start_time).total_seconds()

def validate_one_iteration(vdata, model, loss_fn):
    with torch.no_grad():
        vinputs, vlabels = vdata
        vlogits = model(vinputs)
        vloss, vaccuracy = _compute_loss_and_accuracy(vlogits, vlabels, loss_fn)

    return vloss, vaccuracy

def _create_data_generator(dataloader):
    dataloader = cycle(dataloader)
    while True:
        for data in dataloader:
            yield data

def _create_result_dir():
    i = 1
    while True:
        path = f"./runs/run_{i}"
        if not os.path.isdir(path):
            os.makedirs(path)
            break
        i += 1
    return path

def _write_result_to_csv(result_dir, iteration, training_time, training_accuracy, validation_accuracy, training_loss, validation_loss):
    with open(os.path.join(result_dir, "results.csv"),'a') as f:
        f.write(f"{iteration},{training_time},{training_accuracy},{validation_accuracy},{training_loss},{validation_loss}")
        f.write('\n')

def main():
    result_dir = _create_result_dir()
    _write_result_to_csv(result_dir, "Iteration", "Training Time", "Training Accuracy", "Validation Accuracy", "Training Loss", "Validation Loss")

    BATCH_SIZE = 30
    ITERATIONS = 500

    cls_map, training_loader, validation_loader = _create_dataloader("./data/earth/source", BATCH_SIZE)
    model = LeNet5(len(cls_map), [5,5,5,5,5,5])
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    # optimizer =torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, 500, 0.99)

    with open(os.path.join(result_dir, "classes.json"), 'w') as f: 
        f.write(json.dumps(cls_map))

    training_data_generator = _create_data_generator(training_loader)
    validation_data_generator = _create_data_generator(validation_loader)

    accumulated_time = 0
    best_validation_loss = 1000000

    for iteration in range(1, ITERATIONS + 1):
        print(f"Iteration {iteration}:")

        model.train()
        training_loss, training_accuracy, iteration_time = train_one_iteration(next(training_data_generator), optimizer, scheduler, model, loss_fn)
        accumulated_time += iteration_time

        model.eval()
        validation_loss, validation_accuracy = validate_one_iteration(next(validation_data_generator), model, loss_fn)

        print('Learning Rate: {}'.format(optimizer.param_groups[0]["lr"]))
        print('Loss: train {} valid {}'.format(training_loss, validation_loss))
        print('Accuracy: train {}% valid {}%'.format(training_accuracy, validation_accuracy))

        _write_result_to_csv(result_dir, iteration, accumulated_time, training_accuracy, validation_accuracy, training_loss, validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), os.path.join(result_dir, f"best.pt"))

    return

if __name__ == "__main__":
    main()
