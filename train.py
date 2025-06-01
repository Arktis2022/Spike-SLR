import argparse
import json

from trainer_snn import train

parser = argparse.ArgumentParser(description='Training script with JSON file path')
parser.add_argument('json_path', type=str, help='Path to the JSON file with training parameters')
args = parser.parse_args()

path_results = 'pretrained_model'

# load training params
train_params = json.load(open(args.json_path, 'r'))
train_params['logger_params']['csv']['save_dir'] = '{}'
for k, v in train_params['callbacks_params']:
    if k != 'model_chck':
        continue
    v['dirpath'] = '{}/weights/'
    v['filename'] = '{epoch}-{val_loss_total:.5f}-{val_loss_clf:.5f}-{val_acc:.5f}'

path_model = train('/tests', path_results,
                   data_params=train_params['data_params'],
                   backbone_params=train_params['backbone_params'],
                   clf_params=train_params['clf_params'],
                   training_params=train_params['training_params'],
                   optim_params=train_params['optim_params'],
                   callback_params=train_params['callbacks_params'],
                   logger_params=train_params['logger_params'])
