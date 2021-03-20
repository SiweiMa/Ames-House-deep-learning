import torch
from torch import nn
import pandas as pd
from torch.utils import data
from sklearn.model_selection import KFold
from helper import *

train_data = pd.read_csv('train_pytorch_01.csv', index_col=0)
test_data = pd.read_csv('test_pytorch_01.csv', index_col=0)

X_train = torch.tensor(train_data.iloc[:, (train_data.columns != 'SalePrice')].values, dtype=torch.float32)
y_train = torch.tensor(train_data.iloc[:, ~(train_data.columns != 'SalePrice')].values, dtype=torch.float32)
X_test = torch.tensor(test_data.values, dtype=torch.float32)


def load_array(data_arrays, batch_size, is_training=True):
    """
    construct a pytorch data iterator
    data_arrays is a tuple with (X_train, y_train)
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_training)


def get_k_fold_data(X, y, k, is_shuffled=True):
    """
    define k-folder by using KFold in sklearn
    return X_train, X_valid, y_train, y_valid
    """
    kf = KFold(n_splits=k, shuffle=is_shuffled)
    for train_index, valid_index in kf.split(X):
        yield X[train_index], X[valid_index], y[train_index], y[valid_index]


def get_net(d, h1, dropout1, q):
    """
    define the model with one hiden layer
    """
    net = nn.Sequential(
        nn.Linear(d, h1),
        nn.ReLU(),
        nn.Dropout(dropout1),
        nn.Linear(h1, q)
    )
    return net


def init_weights(m):
    """
    initialize the parameters
    m is the module in nn.Sequential
    """
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        # nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, val=0)


def log_rmse(net, loss, X, y):
    """
    define validation metrics as log rmse and return log rmse in the format of a single python number
    """
    # set the model to evaluation model, thus  effectively layers like dropout, batchnorm etc. can behave accordingly.
    if isinstance(net, torch.nn.Module):
        net.eval()

    clipped_preds = torch.clamp(net(X), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(y)))
    return rmse.item()


def optimize(net, lr, wd):
    """
    define the optimization with Adam optimizer
    """
    return torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)


def train_epoch(net, loss, optimizer, train_iter):
    """
    define the training process for an epoch.
    get net, loss, optimizer from k_fold_train function
    update parameter in the model for each minibatch
    """
    # set the model to training model, thus  effectively layers like dropout, batchnorm etc. can behave accordingly.
    if isinstance(net, torch.nn.Module):
        net.train()

    for X, y in train_iter:
        with torch.enable_grad():
            optimizer.zero_grad()
            l = loss(net(X), y)
        l.backward()
        optimizer.step()


def train(net, optimizer, loss, X_train, X_valid, y_train, y_valid, num_epochs, batch_size):
    """
    define the training process for a k-fold split
    get net, loss, optimizer from k_fold_train function
    get X_train, X_valid, y_train, y_valid from k_fold_train function

    create a train iterator by using load_array with (X_train, y_train) and hyperparameter batch_size
    return evaluation metrics for a k-fold split
    """
    eval_metrics_train, eval_metrics_valid = [], []

    train_iter = load_array((X_train, y_train), batch_size, is_training=True)

    for epoch in range(num_epochs):
        train_epoch(net, loss, optimizer, train_iter)

        eval_metrics_train.append(log_rmse(net, loss, X_train, y_train))
        if y_valid is not None:
            eval_metrics_valid.append(log_rmse(net, loss, X_valid, y_valid))

    return eval_metrics_train, eval_metrics_valid


def k_fold_train(model_hps, optimizer_hps, train_hps, k, X_train, y_train):
    """
    define the k-fold training process
    model_hps: the hyperparameters control model, net
    optimizer_hps: the hyperparameters control Adam optimizer
    train_hps: the hyperparameters, num_epochs, batch_size, control the general training process
    print out the average train and validation evalution metrics
    """
    eval_metrics_train_sum, eval_metrics_valid_sum = 0, 0
    for k_fold_data in get_k_fold_data(X_train, y_train, k):  # X_train, X_valid, y_train, y_valid
        net = get_net(*model_hps)
        net.apply(init_weights)
        optimizer = optimize(net, *optimizer_hps)
        loss = nn.MSELoss()  # define the loss function

        eval_metrics_train, eval_metrics_valid = train(net, optimizer, loss, *k_fold_data, *train_hps)

        eval_metrics_train_sum += eval_metrics_train[-1]
        eval_metrics_valid_sum += eval_metrics_valid[-1]

    plot(list(range(1, train_hps[0] + 1)), [eval_metrics_train, eval_metrics_valid],
         xlabel='epoch', ylabel='rmse', xlim=[1, train_hps[0]],
         legend=['train', 'valid'], yscale='linear')

    print(f'{k}-fold validation: average train log rmse: {eval_metrics_train_sum / k}, average validation log rmse: {eval_metrics_valid_sum / k}')


def finaltrain_and_predict(model_hps, optimizer_hps, train_hps, X_train, y_train, X_test):
    """
    final train with all data in train set
    predict sale price by using test set and save it in submission file
    """
    net = get_net(*model_hps)
    net.apply(init_weights)
    optimizer = optimize(net, *optimizer_hps)
    loss = nn.MSELoss()  # define the loss function
    eval_metrics_train, _ = train(net, optimizer, loss, X_train, None, y_train, None, *train_hps)

    print(f'train log rmse: {eval_metrics_train[-1]}')
    plot(list(range(1, train_hps[0] + 1)), [eval_metrics_train],
         xlabel='epoch', ylabel='rmse', xlim=[1, train_hps[0]],
         legend=['train', 'valid'], yscale='linear')

    preds = net(X_test).detach().numpy()
    # Reformat it to export to Kaggle
    test = pd.read_csv('test.csv')
    test['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test['Id'], test['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    d = X_train.shape[1]
    h1 = 32
    dropout1 = 0.2
    q = 1
    model_hps = [d, h1, dropout1, q]

    lr = 0.5
    wd = 0
    optimizer_hps = [lr, wd]

    num_epochs = 20
    batch_size = 64
    train_hps = [num_epochs, batch_size]

    k = 5

    k_fold_train(model_hps, optimizer_hps, train_hps, k, X_train, y_train)
    finaltrain_and_predict(model_hps, optimizer_hps, train_hps, X_train, y_train, X_test)