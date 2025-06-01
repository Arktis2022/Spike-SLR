import os

os.chdir('..')
import numpy as np
import sparse
import pickle

import aermanager
from aermanager.aerparser import load_events_from_file
from tqdm import tqdm

np.random.seed(0)

chunk_len_ms = 12
chunk_len_us = chunk_len_ms * 1000
height, width = 128, 128

mapping = {'climb':0, 'stand':1, 'jump':2, 'fall':3, 'sit':4, 'get up':5,
    'walk':6, 'run':7, 'lift':8, 'lie':9, 'bend':10, 'pick':11}

# Source data folder
path_dataset = './datasets/DailyActionDVS/'

root_train = path_dataset
files_all = []
files_train = []
files_test = []
for now_file, label_train_one, files_train_one in os.walk(root_train):
    i = 0
    for now_data_file in files_train_one:
        if i < 120 * 0.8:
            files_train.append(now_file + '/' + now_data_file)
        else:
            files_test.append(now_file + '/' + now_data_file)
        files_all.append(now_file + '/' + now_data_file)
        i += 1

parser = aermanager.parsers.parse_dvs_128

# Target data folder
if not os.path.isdir(path_dataset + 'clean_datasets'):
    os.mkdir(path_dataset + 'clean_datasets')
    os.makedirs(path_dataset + 'clean_datasets/' + f'dataset_{chunk_len_us}/train')
    os.makedirs(path_dataset + 'clean_datasets/' + f'dataset_{chunk_len_us}/validation')

for events_file in tqdm(files_all):
    shape, events = load_events_from_file(events_file, parser=parser)
    new_event_ending = events_file.split('/')[-1].replace('.aedat', '.pckl')
    filename_dst = path_dataset + 'clean_datasets/' + f'dataset_{chunk_len_us}/{{}}/' + \
                        f'{{}}_{new_event_ending}'

    total_events = np.array([events['x'], events['y'], events['t'], events['p']]).transpose()

    total_chunks = []
    while total_events.shape[0] > 0:
        start_time = total_events[0][2]
        end_time = total_events[-1][2]
        if start_time < end_time:
            end_t = end_time
            chunk_inds = np.where(total_events[:, 2] >= end_t - chunk_len_us)[0]
            if len(chunk_inds) <= 4:
                pass
            else:
                total_chunks.append(total_events[chunk_inds])
            total_events = total_events[:max(1, chunk_inds.min()) - 1]

        else:
            start_t = end_time
            chunk_inds = np.where(total_events[:, 2] <= start_t + chunk_len_us)[0]
            if len(chunk_inds) <= 4:
                pass
            else:
                total_chunks.append(total_events[chunk_inds])
            total_events = total_events[:max(1, chunk_inds.min()) - 1]

            
    if len(total_chunks) == 0:
            continue
    total_chunks = total_chunks[::-1]

    total_frames = []
    for chunk in total_chunks:
        frame = sparse.COO(chunk[:, [0, 1, 3]].transpose().astype('int32'), np.ones(chunk.shape[0]).astype('int32'), (height, width, 2))  # .to_dense()
        total_frames.append(frame)
    total_frames = sparse.stack(total_frames)

    total_frames = np.clip(total_frames, a_min=0, a_max=255)
    total_frames = total_frames.astype('uint8')
    label = mapping[events_file.split('/')[-2]]


    if events_file in files_train:
        pickle.dump(total_frames, open(filename_dst.format('train', label), 'wb'))
    if events_file in files_test:
        pickle.dump(total_frames, open(filename_dst.format('validation', label), 'wb'))
