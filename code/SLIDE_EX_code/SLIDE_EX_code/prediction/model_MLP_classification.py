# !/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score

from utils import *


# check available device
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

##================================================================================================
class MLP_classification(nn.Module):
    def __init__(self, n_inputs, n_hiddens, n_outputs, dropout, bias_init=None):
        super(MLP_classification, self).__init__()

        self.layer0 = nn.Sequential(
            nn.Linear(n_inputs, n_hiddens),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.layer1 = nn.Linear(n_hiddens, n_outputs)
        if bias_init is not None:  # set bias of the last layer
            self.layer1.bias = bias_init

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = torch.mean(x, dim=0)  # sum over tiles
        return x


##================================================================================================
def training_epoch(model, optimizer, train_set, batch_size):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()

    n_slides_train = len(train_set)

    # shuffle training set
    idx_list = np.arange(n_slides_train)
    np.random.shuffle(idx_list)

    loss_list = []
    total_count = 0
    correct_count = 0
    
    for i_batch in range(0, n_slides_train, batch_size):    
        n_slides_batch = min(batch_size, n_slides_train - i_batch)

        # for each batch
        loss = 0
        for k in range(n_slides_batch):
            idx = idx_list[i_batch + k]
            x, y = train_set[idx]
            output = model(x.float().to(device)).squeeze()
            loss += loss_fn(output, torch.tensor(y, dtype=torch.float).to(device))
            predicted = (torch.sigmoid(output) > 0.5).float()

            total_count += 1
            correct_count += (predicted == y).sum().item()

        loss /= n_slides_batch
        loss_list += [loss.detach().cpu().numpy()]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(loss_list), correct_count / total_count


##================================================================================================
def predict(model, valid_set):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    
    labels = []
    preds = []
    confs = []
    loss_list = []
    total_count = 0
    correct_count = 0

    with torch.no_grad():
        for x, y in valid_set:
            output = model(x.float().to(device))

            loss = loss_fn(output.squeeze(), torch.tensor(y, dtype=torch.float).to(device))
            loss_list += [loss.detach().cpu().numpy()]
            
            sigmoid_output = torch.sigmoid(output).squeeze()
            predicted = (sigmoid_output > 0.5).float()
            conf = sigmoid_output.item()

            total_count += 1
            correct_count += (predicted == y).sum().item()
            
            labels.append(y)
            preds.append(predicted.detach().cpu().item())
            confs.append(conf)
    
    labels = np.array(labels)
    preds = np.array(preds)
    confs = np.array(confs)
    
    return np.mean(loss_list), labels, preds, confs, correct_count / total_count


##================================================================================================
def fit(model, optimizer, train_set, valid_set, max_epochs, patience, batch_size):
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []

    epoch_since_best = 0
    best_val_acc = 0.0
    
    for e in range(max_epochs):
        epoch_since_best += 1

        # train
        train_loss, train_acc = training_epoch(model, optimizer, train_set, batch_size)

        # predict
        valid_loss, valid_labels, valid_preds, valid_confs, valid_acc = predict(model, valid_set)

        print(f"{e}, train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}")

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)

        if valid_acc > best_val_acc:
            epoch_since_best = 0
            best_val_acc = valid_acc

        if epoch_since_best == patience:
            print('Early stopping at epoch {}'.format(e + 1))
            break

    return model, train_loss_list, train_acc_list,\
        valid_loss_list, valid_acc_list, valid_labels, valid_preds, valid_confs


##================================================================================================
def analyze_result(result_dir, model, train_loss, train_acc,\
               valid_loss, valid_acc, valid_labels, valid_preds, valid_confs, test_set):

    # save trained model
    torch.save(model.state_dict(), "%s/model_trained.pth"%(result_dir))

    train_valid_loss = np.array((train_loss, valid_loss, train_acc, valid_acc)).T
    np.savetxt(f"{result_dir}/train_valid_loss.txt", train_valid_loss, fmt="%.6f")

    # predict test
    test_loss, test_labels, test_preds, test_confs, test_acc = predict(model, test_set)
    
    valid_precision = precision_score(valid_labels, valid_preds, average='macro')
    valid_recall = recall_score(valid_labels, valid_preds, average='macro')
    valid_f1 = f1_score(valid_labels, valid_preds, average='macro')
    valid_accuracy = accuracy_score(valid_labels, valid_preds)
    valid_balanced_acc = balanced_accuracy_score(valid_labels, valid_preds)
    valid_metrics = np.array([valid_precision, valid_recall, valid_f1, valid_accuracy, valid_balanced_acc]).reshape(1, 5)
    valid_metrics_header = 'valid_precision valid_recall valid_f1 valid_accuracy valid_balanced_acc'
    
    test_precision = precision_score(test_labels, test_preds, average='macro')
    test_recall = recall_score(test_labels, test_preds, average='macro')
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_balanced_acc = balanced_accuracy_score(test_labels, test_preds)
    test_metrics = np.array([test_precision, test_recall, test_f1, test_accuracy, test_balanced_acc]).reshape(1, 5)
    test_metrics_header = 'test_precision test_recall test_f1 test_accuracy test_balanced_acc'

    np.savetxt(f"{result_dir}/valid_labels.txt", valid_labels, fmt="%.8f")
    np.savetxt(f"{result_dir}/valid_preds.txt", valid_preds, fmt="%.8f")
    np.savetxt(f"{result_dir}/valid_confs.txt", valid_confs, fmt="%.8f")
    np.savetxt(f"{result_dir}/valid_metrics.txt", valid_metrics, header=valid_metrics_header, fmt="%.8f")
    np.savetxt(f"{result_dir}/test_labels.txt", test_labels, fmt="%.8f")
    np.savetxt(f"{result_dir}/test_preds.txt", test_preds, fmt="%.8f")
    np.savetxt(f"{result_dir}/test_confs.txt", test_confs, fmt="%.8f")
    np.savetxt(f"{result_dir}/test_metrics.txt", test_metrics, header=test_metrics_header, fmt="%.8f")

    
    fig, ax = plt.subplots(figsize=(8,6))

    # 1st col:
    ax.plot(train_loss, 'k--', label="train")
    ax.plot(valid_loss, 'b-', label="valid")

    ax.set_xlabel("n_epochs")
    ax.legend()
    ax.set_ylabel("loss")

    plt.tight_layout(h_pad=1, w_pad= 0.5)
    plt.savefig(f"{result_dir}/loss.pdf", format='pdf', dpi=50)
