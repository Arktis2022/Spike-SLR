import json
import warnings

import pytorch_lightning as pl
import torch
import tqdm
from spikingjelly.clock_driven import functional
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import training_utils
from data_generation import Event_DataModule
from snn_models.SNN_Conv_plus import EvNetBackbone, CLFBlock

warnings.filterwarnings("ignore")
accuracy = pl.metrics.Accuracy()

import matplotlib.pyplot as plt


def plot_training_results(train_losses, train_accs, val_losses, val_accs, save_path):
    plt.figure(figsize=(20, 10))

    # training loss
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # training accuracy
    plt.subplot(2, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.title('Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # validation loss
    plt.subplot(2, 2, 3)
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # validation accuracy
    plt.subplot(2, 2, 4)
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # save figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

class SimpleLogger:
    def __init__(self, file_path):
        self.file_path = file_path

    def log(self, message):
        with open(self.file_path, 'a') as file:
            file.write(str(message) + '\n')

    def log_and_print(self, message):
        print(message)
        self.log(message)


class EvNetModel(nn.Module):
    def __init__(self, backbone_params, clf_params, optim_params, loss_weights=None):
        super(EvNetModel, self).__init__()
        self.backbone_params = backbone_params
        self.clf_params = clf_params
        self.optim_params = optim_params

        self.backbone = EvNetBackbone(**backbone_params)
        # Initialize classifier
        self.clf_params['ipt_dim'] = self.backbone_params['embed_dims']
        self.models_clf = nn.ModuleDict([[str(0), CLFBlock(**self.clf_params)]])

        self.loss_weights = torch.tensor(loss_weights) if loss_weights is not None else None
        self.criterion = nn.NLLLoss(weight=self.loss_weights)

    def forward(self, x):
        embs = self.backbone(x)
        clf_logits = torch.stack([self.models_clf[str(0)](embs)]).mean(axis=0)

        return embs, clf_logits


def train_one_epoch(epoch, model, train_loader, optimizer, device, batch_size, iters, scheduler):
    model.train()
    total_loss = 0.0
    total_acc = 0.0  

    pbar = tqdm.tqdm(total=iters, desc=f'Epoch {epoch}, Loss: {total_loss:.4f}, Acc: {total_acc:.4f}')
    for iter, (input_spikes, y) in enumerate(train_loader, start=1):
        input_spikes, y = input_spikes.to(device), y.to(device)
        optimizer.zero_grad()
        _, clf_logits = model(input_spikes)
        loss = model.criterion(clf_logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step() 
        total_loss += loss.item()

        # update
        acc = accuracy(clf_logits.cpu(), y.cpu())
        total_acc += acc

        current_lr = optimizer.param_groups[0]['lr']

        pbar.set_description(
            f'Epoch {epoch}, Loss: {total_loss / iter:.4f}, Acc: {total_acc / iter:.4f}, LR: {current_lr}')
        pbar.update(1)  

        if iter >= iters:
            break

        functional.reset_net(model)  
    pbar.close() 
    return total_loss / iters, total_acc / iters


def val_one_epoch(model, val_loader, device):
    model.eval() 
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    with torch.no_grad():  
        for input_spikes, y in val_loader:
            input_spikes, y = input_spikes.to(device), y.to(device)
            _, clf_logits = model(input_spikes)
            loss = model.criterion(clf_logits, y)

            acc = accuracy(clf_logits.cpu(), y.cpu()).item()

            total_loss += loss.item() * y.size(0)
            total_acc += acc * y.size(0)
            total_samples += y.size(0)

            functional.reset_net(model)  
    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    return avg_loss, avg_acc


def train(folder_name, path_results, data_params, backbone_params, clf_params,
          training_params, optim_params, callback_params, logger_params):
    
    path_model = training_utils.create_model_folder(path_results, folder_name)

    # logging.basicConfig(filename='./'+path_model+'training.log',
    # level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = SimpleLogger('./' + path_model + 'training.txt')

    logger.log_and_print('------------------------------')

    logger.log_and_print('load dataset....')
    dm = Event_DataModule(**data_params)
    batch_size = dm.batch_size
    logger.log_and_print('batch size : ' + str(dm.batch_size))
    logger.log_and_print('train data num : ' + str(len(dm.train_dataloader()) * dm.batch_size))
    logger.log_and_print('val datal num : ' + str(len(dm.val_dataloader()) * (dm.batch_size // 2 + 1)))

    logger.log_and_print('------------------------------')
    logger.log_and_print('dataloader info...') 
    for input_spikes, label in dm.train_dataloader():
        logger.log_and_print('input_spikes shape:' + str(input_spikes.shape))
        logger.log_and_print('label shape:' + str(label.shape))
        logger.log_and_print('------------------------------')
        break

    
    clf_params['opt_classes'] = dm.num_classes

    
    if optim_params['scheduler']['name'] == 'one_cycle_lr':
        optim_params['scheduler']['params']['steps_per_epoch'] = 1
    clf_params['ipt_dim'] = backbone_params['embed_dims']

    
    json.dump({'data_params': data_params, 'backbone_params': backbone_params, 'clf_params': clf_params,
               'training_params': training_params,
               'optim_params': optim_params, 'callbacks_params': callback_params, 'logger_params': logger_params},
              open(path_model + 'all_params.json', 'w'))

    
    model = EvNetModel(backbone_params=backbone_params,
                       clf_params=clf_params,
                       optim_params=optim_params,
                       loss_weights=None if not data_params[
                           'balance'] else dm.train_dataloader().dataset.get_class_weights())

    print(count_parameters(model))

    
    functional.reset_net(model)

    logger.log_and_print('model info...')
    logger.log_and_print(model.backbone)
    logger.log_and_print(model.models_clf)
    logger.log_and_print('------------------------------')

    
    optimizer = AdamW(model.parameters(), **optim_params['optim_params'])
    epochs = optim_params['scheduler']['params']['epochs']
    log_every_n_steps = training_params['log_every_n_steps']
    gpu_id = training_params['gpus']

    scheduler = OneCycleLR(optimizer, max_lr=0.001,
                           steps_per_epoch=log_every_n_steps, epochs=epochs,
                           anneal_strategy='linear')  

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    best_val_acc = 0

    for epoch in range(epochs):
        functional.reset_net(model)
        train_loss, train_acc = train_one_epoch(epoch, model, dm.train_dataloader(), optimizer, device, batch_size,
                                                iters=log_every_n_steps, scheduler=scheduler)
        functional.reset_net(model)
        print()
        logger.log_and_print(
            'epoch: ' + str(epoch) + ' train_loss: ' + str(train_loss) + ' train acc: ' + str(train_acc.item()))
        print()
        functional.reset_net(model)
        val_loss, val_acc = val_one_epoch(model, dm.val_dataloader(), device)
        functional.reset_net(model)
        logger.log_and_print(f'Epoch: {epoch}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print()
        
        model_path = './' + path_model + '/weights/' + f'epoch_{epoch}_val_loss_{val_loss:.4f}_val_acc_{val_acc:.4f}.pth'
        torch.save(model.state_dict(), model_path)
        logger.log_and_print(f'Model saved: {model_path}')
        print()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

       
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = './' + path_model + '/weights/' + f'best_acc={best_val_acc:.4f}.pth'
            torch.save(model.state_dict(), best_model_path)

    plot_training_results(train_losses, train_accs, val_losses, val_accs, save_path='./' + path_model + '/loss_and_acc.png')
