import torch
import torch.nn as nn
import time

from model import DNN
from dataset import (vocab, train_dataloader, test_dataloader)
from config import (model, optimizer, criterion)


import torch.optim as optim
from torch.nn import CrossEntropyLoss


torch.manual_seed(42)
torch.backends.cudnn.deterministic = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_params(model))
"""


def model_accuracy(pred, y_true):
    predict = pred.argmax(dim=1, keepdim=True)
    correct = predict.eq(y_true.view_as(predict)).sum()
    acc = correct.float() / y_true.shape[0]
    return acc



def train(model, criterion, optimizer, dataloader):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for label, data, text_lengths in dataloader:
        optimizer.zero_grad()

        predictions = model(data, text_lengths).squeeze(1)
        loss = criterion(predictions, label)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        acc = model_accuracy(predictions, label)

       
        loss.backward()
        optimizer.step()
        # scheduler.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


def test(model, criterion, dataloader):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for label, data, text_lengths in dataloader:
            predictions = model(data, text_lengths).squeeze(1)
            loss = criterion(predictions, label)
            acc = model_accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_epochs = 10
best_valid_loss = float('inf')

for epoch in range(N_epochs):
    start_time = time.time()

    train_loss, train_acc = train(model, criterion, optimizer, train_dataloader)
    valid_loss, valid_acc = test(model, criterion, test_dataloader)
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'nepali-model.pt')

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')  
    print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}%')  

