import joblib
import torch
from sklearn import metrics
from torch import cuda

from dataset import get_data_loader
from metrics import Metrics
from utils import get_train_data, random_over_sampling, save_model

my_metrics = Metrics()

device = 'cuda' if cuda.is_available() else 'cpu'


def train_ml_models(model, train_data, model_name, pre_processing_fun, pre_processing):
    accuracy = []
    precision = []
    f1_score = []
    recall = []

    for fold in range(5):
        X_train, X_valid, y_train, y_valid = get_train_data(train_data, fold)
        X_train, y_train = random_over_sampling(X_train, y_train)
        if pre_processing == "tfidf":
            X_train, y_train = pre_processing_fun.transform(X_train, y_train)
            X_valid, y_valid = pre_processing_fun.transform(X_valid, y_valid)
        elif pre_processing == "embedding":
            X_train, y_train = pre_processing_fun(X_train, y_train)
            X_valid, y_valid = pre_processing_fun(X_valid,y_valid)
        else:
            raise ValueError("Given type not supported")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        accuracy.append(my_metrics("accuracy", y_valid, y_pred))
        precision.append(my_metrics("precision", y_valid, y_pred))
        f1_score.append(my_metrics("f1", y_valid, y_pred))
        recall.append(my_metrics("recall", y_valid, y_pred ))
    print(f"Mean Accuracy validation set: {sum(accuracy)/len(accuracy)}")
    print(f"Mean Precision validation set: {sum(precision)/len(precision)}")
    print(f"Mean Recall val set: {sum(recall)/len(recall)}")
    print(f"Mean F1 validation set: {sum(f1_score)/len(f1_score)}")
    print("Classification Report: \n", metrics.classification_report(y_valid, y_pred))
    print("Confusion matrix: \n", metrics.confusion_matrix(y_valid, y_pred))
    # save model and predictions
    joblib.dump(model, f"../models/{model_name}_{pre_processing}.pkl")


def _calcuate_accuracy(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct


def train_epoch(model, criterion, optimizer, train_loader, epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    y_preds = []
    y_true = []

    for _, data in enumerate(train_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, mask)
        loss = criterion(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += _calcuate_accuracy(big_idx, targets)
        y_preds.extend(list(big_idx.cpu().detach().numpy()))
        y_true.extend(list(targets.cpu().detach().numpy()))
        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()
    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")
    #print("Classification Report: \n", metrics.classification_report(y_true, y_preds))
    #print("Confusion matrix: \n", metrics.confusion_matrix(y_true, y_preds))


def validate(model, val_loader):
    y_preds = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(val_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            outputs = model(ids, mask)
            big_val, big_idx = torch.max(outputs.data, dim=1)
            y_preds.extend(list(big_idx.cpu().detach().numpy()))
            y_true.extend(list(targets.cpu().detach().numpy()))
        print(f"Validation accuracy: {my_metrics('accuracy', y_true, y_preds)}")
        print(f"Validation Precision: {my_metrics('precision', y_true, y_preds)}")
        print(f"Validation recall: {my_metrics('recall', y_true, y_preds)}")


def train_transformer(model, train_data, tokenizer, learning_rate, epochs, model_name):
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    for fold in range(5):
        X_train, X_valid, y_train, y_valid = get_train_data(train_data, fold)
        train_loader, valid_loader = get_data_loader(X_train, X_valid, y_train, y_valid, tokenizer)
        for epoch in range(epochs):
            train_epoch(model, criterion, optimizer, train_loader, epoch)
            validate(model, valid_loader)
    save_model(model, optimizer, model_name)
