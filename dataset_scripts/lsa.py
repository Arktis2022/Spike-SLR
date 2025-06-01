import os

os.chdir('..')
import numpy as np
import sparse
import pickle

import aermanager
from aermanager.aerparser import load_events_from_file
from sklearn.model_selection import train_test_split
from tqdm import tqdm

np.random.seed(0)

chunk_len_ms = 2
chunk_len_us = chunk_len_ms * 1000
height, width = 128, 128
train_test_size = 0.25
# train_validation_size = 0.25

# Source data folder
path_dataset = './datasets/LSA_DVS/'
files = os.listdir(path_dataset + 'LSAData/')
parser = aermanager.parsers.parse_dvs_128

# Target data folder
if not os.path.isdir(path_dataset + 'LSA_DVS_splits'):
    os.mkdir(path_dataset + 'LSA_DVS_splits')
    os.makedirs(path_dataset + 'LSA_DVS_splits/' + f'dataset_{chunk_len_us}/train')
    # os.makedirs(path_dataset + 'LSA_DVS_splits/' + f'dataset_{chunk_len_us}/validation')
    os.makedirs(path_dataset + 'LSA_DVS_splits/' + f'dataset_{chunk_len_us}/test')

train_samples_4sets, test_samples_4sets = train_test_split(files, test_size=train_test_size, random_state=0, stratify=[f[:-10] for f in files])
# train_samples_4sets, validation_samples_4sets = train_test_split(train_samples_4sets, test_size=train_validation_size, random_state=0, stratify=[f[:-10] for f in train_samples_4sets])

for events_file in tqdm(files):
    shape, events = load_events_from_file(path_dataset + 'LSAData/' + events_file, parser=parser)
    filename_dst = path_dataset + 'LSA_DVS_splits/' + f'dataset_{chunk_len_us}/{{}}/' + \
                        events_file.replace('.aedat', '.pckl')

    total_events = np.array(
        [events['x'], events['y'], events['t'], events['p']]).transpose()

    total_chunks = []
    while total_events.shape[0] > 0:
        end_t = total_events[-1][2]
        chunk_inds = np.where(total_events[:, 2] >= end_t - chunk_len_us)[0]
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

    class_lsa = events_file.split('_')[1]

    if events_file in train_samples_4sets:
        pickle.dump(total_frames, open(filename_dst.format('train'), 'wb'))
    # if events_file in validation_samples_4sets:
    #     pickle.dump(total_frames, open(filename_dst.format('validation'), 'wb'))
    if events_file in test_samples_4sets:
        pickle.dump(total_frames, open(filename_dst.format('test'), 'wb'))
