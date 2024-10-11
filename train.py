import torch
import numpy as np
import time
import datetime
import os
from utils import evaluate_sharpe, get_crossval_dataloaders
from models import DeepNetwork

def train_one_epoch(model, train_loader, mask, optimizer, criterion, config):
    epoch_loss = torch.tensor(0.0)
    y_train = []
    y_pred_train = []

    for i, (X, y) in enumerate(train_loader):
        X, y = X.float().to(config['device']), y.float().to(config['device'])
        optimizer.zero_grad()
        y_pred = model(X)

        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().cpu().item()

        y_train.append(y.detach().cpu())
        y_pred_train.append(y_pred.detach().cpu())

    epoch_loss /= i + 1
    y_train = torch.cat(y_train, dim=0)
    y_pred_train = torch.cat(y_pred_train, dim=0)
    epoch_sharpe = evaluate_sharpe(y_pred_train, y_train, mask)

    return epoch_loss, epoch_sharpe


def validate(model, val_loader, mask, criterion, config):
    # Validation
    with torch.no_grad():
        epoch_val_loss = torch.tensor(0.0)
        y_val = []
        y_pred_val = []
        for i, (X, y) in enumerate(val_loader):
            X, y = X.float().to(config['device']), y.float().to(config['device'])
            y_pred = model(X)

            loss = criterion(y_pred, y)
            epoch_val_loss += loss.detach().cpu().item()
            y_val.append(y.detach().cpu())
            y_pred_val.append(y_pred.detach().cpu())

    epoch_val_loss /= i + 1
    y_val = torch.cat(y_val, dim=0)
    y_pred_val = torch.cat(y_pred_val, dim=0)
    epoch_val_sharpe = evaluate_sharpe(y_pred_val, y_val, mask)

    return epoch_val_loss, epoch_val_sharpe


def training(model, train_loader, val_loader, train_mask, val_mask, config, fold, run):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['reg_l2'])
    criterion = torch.nn.MSELoss()
    best_model_state = model.state_dict()
    best_val_loss = torch.tensor(float('inf'))
    best_val_sharpe = torch.tensor(-float('inf'))
    best_epoch = 0

    sharpe_train = []
    sharpe_val = []
    loss_train = []
    loss_val = []

    time_start = time.time()
    for epoch in range(config['epochs']):
        epoch_loss, epoch_sharpe = train_one_epoch(model=model,
                                                   train_loader=train_loader,
                                                   mask=train_mask,
                                                   optimizer=optimizer,
                                                   criterion=criterion,
                                                   config=config)
        loss_train.append(epoch_loss)
        sharpe_train.append(epoch_sharpe)

        # Validation
        epoch_val_loss, epoch_val_sharpe = validate(model=model,
                                                    val_loader=val_loader,
                                                    mask=val_mask,
                                                    criterion=criterion,
                                                    config=config)
        loss_val.append(epoch_val_loss)
        sharpe_val.append(epoch_val_sharpe)

        if epoch <= 50 or epoch % 50 == 0 or epoch > (config['epochs'] - 50):
            print(
                'Epoch {} - Training Loss: {:.8f}, Val Loss: {:.8f}, Train Sharpe: {:.8f}, Validation Sharpe: {:.8f}'.format(
                    epoch + 1,
                    epoch_loss,
                    epoch_val_loss,
                    epoch_sharpe,
                    epoch_val_sharpe))

        if config['criteria'] == 'Factor_sharpe':
            if epoch_val_sharpe > best_val_sharpe:
                best_val_sharpe = epoch_val_sharpe
                best_model_state = model.state_dict()
                best_epoch = epoch
                print("Best model updated at epoch {}".format(epoch + 1))
                if (50 < epoch < (config['epochs'] - 50)) and epoch % 50 != 0:
                    print(
                        'Epoch {} - Training Loss: {:.8f}, Val Loss: {:.8f}, Train Sharpe: {:.8f}, Validation Sharpe: {:.8f}'.format(
                            epoch + 1, epoch_loss, epoch_val_loss, epoch_sharpe, epoch_val_sharpe))

        elif epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch
            print("Best model updated at epoch {}".format(epoch + 1))
            if (50 < epoch < (config['epochs'] - 50)) and epoch % 50 != 0:
                print(
                    'Epoch {} - Training Loss: {:.8f}, Val Loss: {:.8f}, Train Sharpe: {:.8f}, Validation Sharpe: {:.8f}'.format(
                        epoch + 1, epoch_loss, epoch_val_loss, epoch_sharpe, epoch_val_sharpe))

    exp_path = './Experiments/'
    exp_subset_path = os.path.join(exp_path, config['subset'])
    if not os.path.exists(exp_subset_path):
        os.makedirs(exp_subset_path)

    exp_subset_fold_path = os.path.join(exp_subset_path, 'fold' + str(fold + 1))
    if not os.path.exists(exp_subset_fold_path):
        os.makedirs(exp_subset_fold_path)

    model.load_state_dict(best_model_state)
    time_stamp = datetime.datetime.today().strftime('%Y%m%d_%H%M')
    model_save_path = os.path.join(exp_subset_fold_path,
                                   'fold' + str(fold + 1) + '_' + 'model' + str(run + 1) + '_' + config[
                                       'subset'] + '_' + time_stamp + '.pth')
    torch.save(model, model_save_path)

    duration = time.time() - time_start
    print('Training completed in {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))
    return model, loss_train, loss_val, sharpe_train, sharpe_val, best_epoch


def train_ensembles(config, crossval_loaders, masks, fold, ensemble_members=8):
    train_loader, val_loader, test_loader = crossval_loaders[fold]['dataloaders']
    train_mask, val_mask, test_mask = masks[fold]

    avg_loss_ens = {'train': 0.0, 'val': 0.0, 'test': 0.0}
    avg_sharpe_ens = {'train': 0.0, 'val': 0.0, 'test': 0.0}

    for run in range(ensemble_members):
        print('\nTRAINING ENSEMBLE MEMBER {}'.format(run + 1))
        model = DeepNetwork(config).to(device=config['device'])
        model, loss_train, loss_val, sharpe_train, sharpe_val, best_epoch = training(model, train_loader, val_loader,
                                                                                     train_mask, val_mask, config, fold,
                                                                                     run)
        test_loss, test_sharpe = validate(model, test_loader, test_mask, torch.nn.MSELoss(), config)

        avg_loss_ens['train'] += loss_train[best_epoch]
        avg_loss_ens['val'] += loss_val[best_epoch]
        avg_loss_ens['test'] += test_loss

        avg_sharpe_ens['train'] += sharpe_train[best_epoch]
        avg_sharpe_ens['val'] += sharpe_val[best_epoch]
        avg_sharpe_ens['test'] += test_sharpe

    avg_loss_ens = {key: value / ensemble_members for key, value in avg_loss_ens.items()}
    avg_sharpe_ens = {key: value / ensemble_members for key, value in avg_sharpe_ens.items()}

    return avg_loss_ens, avg_sharpe_ens


def run_one_subset(config):
    crossval_loaders, masks = get_crossval_dataloaders(config['data_path'], config['split_lists'], config['subset'],
                                                       batch_size=config['batch_size'])

    avg_loss_fold = {'train': 0.0, 'val': 0.0, 'test': 0.0}
    avg_sharpe_fold = {'train': 0.0, 'val': 0.0, 'test': 0.0}

    for fold in range(len(crossval_loaders)):
        print('\nRUNNING FOLD NO. {}'.format(fold + 1))
        avg_loss_ens, avg_sharpe_ens = train_ensembles(config, crossval_loaders, masks, fold,
                                                       config['ensemble_members'])

        avg_loss_fold['train'] += avg_loss_ens['train']
        avg_loss_fold['val'] += avg_loss_ens['val']
        avg_loss_fold['test'] += avg_loss_ens['test']

        avg_sharpe_fold['train'] += avg_sharpe_ens['train']
        avg_sharpe_fold['val'] += avg_sharpe_ens['val']
        avg_sharpe_fold['test'] += avg_sharpe_ens['test']

    avg_loss_fold = {key: value / (len(crossval_loaders)) for key, value in avg_loss_fold.items()}
    avg_sharpe_fold = {key: value / (len(crossval_loaders)) for key, value in avg_sharpe_fold.items()}
    return avg_loss_fold, avg_sharpe_fold


def run_all_subsets(config):
    subset2col = {
        'flow+fund_mom+sentiment': list(range(56, 60)) + [47],
        'fund_ex_mom_flow': [59] + [x for x in range(46, 58) if x not in (list(range(54, 58)) + [47])],
        'stock': range(46),
        'fund': range(46, 59),
        'fund+sentiment': range(46, 60),
        'stock+fund': range(59),
        'F_r12_2+sentiment': [58, 59],
        'stock+sentiment': [59] + list(range(0, 46)),
        'stock+fund+sentiment': range(60),
        'F_r12_2+flow+sentiment': [47, 58, 59]
    }
    losses = {}
    sharpes = {}
    for subset in subset2col.keys():
        print('\nTRAINING FOR SUBSET: {}'.format(subset))
        config['subset'] = subset
        config['input_dim'] = len(subset2col[config['subset']])
        losses[subset], sharpes[subset] = run_one_subset(config)
    return losses, sharpes
