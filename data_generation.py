import os
import pickle

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from timm.models.layers import to_2tuple
import copy
from scipy import ndimage


DVS128_class_mapping = {0: 'background', 1: 'hand_clapping', 2: 'right_hand_wave', 
                 3: 'left_hand_wave', 4: 'right_arm_clockwise', 
                 5: 'right_arm_counter_clockwise', 6: 'left_arm_clockwise', 
                 7: 'left_arm_counter_clockwise', 8: 'arm_roll', 
                 9: 'air_drums', 10: 'air_guitar', 11: 'other_gestures'}

class EventDataset(Dataset):
    def __init__(self, samples_folder,   
                 time_step,             
                 time_durations,        
                 validation,            
                 augmentation_params,  
                 preproc_polarity,     
                 crop_size,             
                 dataset_name,         
                 height, width,         
                 classes_to_exclude=[]  
                 ):

        self.samples_folder = samples_folder
        self.time_step = time_step
        self.time_durations = time_durations
        self.validation = validation
        self.augmentation_params = augmentation_params
        
        self.chunk_len_us = self.time_durations * 1000

       
        self.sparse_frame_len_us = int(self.samples_folder.split('/')[-3].split('_')[-1])
        self.sparse_frame_len_ms = self.sparse_frame_len_us / 1000
        assert self.chunk_len_us % self.sparse_frame_len_us == 0, 'Chunk length should be multiple of sparse frame length'
        
        self.chunk_size = self.chunk_len_us // self.sparse_frame_len_us

        self.height = height
        self.width = width

        
        if augmentation_params is not None and len(augmentation_params) != 0:
            
            assert self.time_durations % 2 == 0, 'Time duration should be even'
            self.num_sparse_frames = (self.time_step * self.time_durations) // self.sparse_frame_len_ms
            
            if 'random_frame_size' in augmentation_params and augmentation_params['random_frame_size'] is not None:
                self.x_lims = to_2tuple(int(width * augmentation_params['random_frame_size']))
                self.y_lims = to_2tuple(int(height * augmentation_params['random_frame_size']))
            
            if 'drop_token' in augmentation_params and augmentation_params['drop_token'][0] != 0.0:
                self.drop_perc, self.drop_mode = augmentation_params['drop_token'] 
            self.h_flip = augmentation_params.get('h_flip', False)

        self.preproc_polarity = preproc_polarity
        self.crop_size = crop_size

        self.samples = os.listdir(samples_folder)
        if dataset_name in ['SLAnimals_4s', 'SLAnimals_3s']:
            self.labels = [s.split('_')[-1][:-5] for s in self.samples]
            self.unique_labels = {l: i for i, l in enumerate(sorted(set(self.labels)))}
            self.labels = [self.unique_labels[l] for l in self.labels]
            self.num_classes = len(self.unique_labels)
        elif dataset_name in ['LSA_DVS', 'LSA64_DVS_Right']:
            self.labels = [s.split('_')[-2] for s in self.samples]
            self.unique_labels = {l: i for i, l in enumerate(sorted(set(self.labels)))}
            self.labels = [self.unique_labels[l] for l in self.labels]
            self.num_classes = len(self.unique_labels)
        elif dataset_name == 'DVS128':
            for l in classes_to_exclude:
                self.samples = [ s for s in self.samples if '_label{:02}'.format(l) not in s ]
            self.labels = np.array([ int(t[5:7]) for s in self.samples for t in s.split('_') if 'label' in t ]).astype('int8')
            self.unique_labels = { l:i for i,l in enumerate(sorted(set(self.labels))) }
            self.labels = [ self.unique_labels[l] for l in self.labels ]
            self.num_classes = len(self.unique_labels)
        elif dataset_name == 'DailyAction':
            self.labels = [s.split('_')[0] for s in self.samples]
            self.unique_labels = {l: i for i, l in enumerate(sorted(set(self.labels)))}
            self.labels = [self.unique_labels[l] for l in self.labels]
            self.num_classes = len(self.unique_labels)
        else: raise ValueError(f'Dataset {dataset_name} not recognized')

   
    def get_class_weights(self):
        label_dict = self.get_label_dict()
        label_dict = {k: label_dict[k] for k in sorted(label_dict)}
        num_samples = sum([len(v) for v in label_dict.values()])
        class_weights = [num_samples / (len(label_dict) * len(v)) for k, v in label_dict.items()]
        return torch.tensor(class_weights)

    def get_label_dict(self):
        label_dict = { c:[] for c in set(self.labels) }
        for i,l in enumerate(self.labels): label_dict[l].append(i)
        for k in label_dict: label_dict[k] = torch.IntTensor(label_dict[k])
        return label_dict

    
    def crop_in_time(self, total_events):
        if len(total_events) > self.num_sparse_frames:
            if not self.validation:
                init = np.random.randint(len(total_events) - self.num_sparse_frames)
                end = init + self.num_sparse_frames
                total_events = total_events[init:end]
            else:
                init = (len(total_events) - self.num_sparse_frames) // 2
                end = init + self.num_sparse_frames
                total_events = total_events[init:end]
        return total_events

   
    def crop_in_space(self, total_events):
        _, y_size, x_size, _ = total_events.shape
        new_x_size = 128
        new_y_size = 128

        x1 = int(round((x_size - new_x_size))/2.)
        y1 = int(round((y_size - new_y_size))/2.)

        total_events = total_events[:, y1:y1 + new_y_size, x1:x1 + new_x_size, :]
        assert total_events.shape[1] == new_y_size and total_events.shape[2] == new_x_size, print(total_events.shape,
                                                                                                  new_y_size,
                                                                                                  new_x_size)
        return total_events

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx, return_sparse_array=False):
        filename = self.samples[idx]
        label = self.labels[idx]

       
        total_events = pickle.load(open(self.samples_folder + filename, 'rb'))
        
        total_events = self.crop_in_time(total_events)
        
        if 'random_frame_size' in self.augmentation_params and self.augmentation_params['random_frame_size'] is not None:
            total_events = self.crop_in_space(total_events)
        if not self.validation and self.h_flip and np.random.rand() > 0.5: total_events = total_events[:, :, ::-1, :]

        
        current_chunk = None
        all_events = []

       
        sf_num = len(total_events) - 1
        while sf_num > 0:

            if current_chunk is None:
                current_chunk = total_events[max(0, sf_num - self.chunk_size):sf_num][::-1]
                current_chunk = current_chunk.todense()
                sf_num -= self.chunk_size
                if '1' in self.preproc_polarity: current_chunk = current_chunk.sum(-1, keepdims=True)

            current_chunk = current_chunk.sum(0)

            # if 'log' in self.preproc_polarity: current_chunk = np.log(current_chunk)
            # else: raise ValueError(f'Preprocessing polarity {self.preproc_polarity} not recognized')

            all_events.append(torch.tensor(current_chunk.astype(np.float32)))
            current_chunk = None
        input_spikes = torch.stack(all_events)   # TxHxWxC
 
        if input_spikes.shape[0] != self.time_step:
            padding = torch.zeros(self.time_step - input_spikes.shape[0], *input_spikes.shape[1:])
            input_spikes = torch.cat([input_spikes, padding], dim=0)
        input_spikes = input_spikes.permute(0, 3, 1, 2)  # TxCxHxW

        return input_spikes, label


class CustomBatchSampler:

    def __init__(self, batch_size, label_dict, sample_repetitions=1, drop_last=False):

        assert batch_size % sample_repetitions == 0
        self.batch_size = batch_size
        self.label_dict = label_dict
        self.sample_repetitions = sample_repetitions
        self.drop_last = drop_last
        self.generator = torch.Generator()
        self.generator.manual_seed(0)
        self.num_classes = len(self.label_dict)
        self.unique_labels = list(self.label_dict.keys())

    def __len__(self):
        epoch_length = sum([len(v) for v in self.label_dict.values()]) * self.sample_repetitions // self.batch_size
        return epoch_length

    def __iter__(self):

        total_labels = []
        while True:
            inds = []
            for b in range(self.batch_size // self.sample_repetitions):
                if len(total_labels) == 0: total_labels = self.unique_labels.copy()
                k = np.random.randint(0, len(total_labels), size=1)[0]
                k = total_labels.pop(k)
                num_k_samples = len(self.label_dict[k])
                ind = np.random.randint(0, num_k_samples, size=1)[0]
                ind = self.label_dict[k][ind]
                for _ in range(self.sample_repetitions):  inds.append(ind)

            yield inds


class Event_DataModule(LightningDataModule):
    def __init__(self, batch_size, time_step, time_durations,
                 crop_size,
                 augmentation_params,
                 dataset_name,
                 skip_last_event=False,
                 sample_repetitions=1,
                 preproc_polarity=None,
                 custom_sampler=True,
                 workers=8,
                 pin_memory=False,
                 classes_to_exclude=[],
                 balance = None):
        super().__init__()
        self.batch_size = batch_size
        self.time_step = time_step
        self.time_durations = time_durations

        self.augmentation_params = augmentation_params
        self.dataset_name = dataset_name
        self.workers = workers
        self.sample_repetitions = sample_repetitions
        self.preproc_polarity = preproc_polarity
        self.skip_last_event = skip_last_event
        self.pin_memory = pin_memory
        self.crop_size = crop_size
        self.classes_to_exclude = classes_to_exclude

        self.pre_padding = True
        self.custom_sampler = custom_sampler
        self.dataset_name = dataset_name
        if dataset_name == 'SLAnimals_3s':
            self.data_folder = './datasets/SL_Animals/SL_animal_splits/dataset_3sets_2000/'
            self.width, self.height = 128, 128
            self.num_classes = 19
            self.class_mapping = {i: l for i, l in enumerate(range(self.num_classes))}
        elif dataset_name == 'SLAnimals_4s':
            self.data_folder = './datasets/SL_Animals/SL_animal_splits/dataset_4sets_2000/'
            self.width, self.height = 128, 128
            self.num_classes = 19
            self.class_mapping = {i: l for i, l in enumerate(range(self.num_classes))}
        elif dataset_name == 'LSA_DVS':
            self.data_folder = './datasets/LSA_DVS/LSA_DVS_splits/dataset_2000/'
            self.width, self.height = 128, 128
            self.num_classes = 64
            self.class_mapping = {i: l for i, l in enumerate(range(self.num_classes))}
        elif dataset_name == 'LSA64_DVS_Right':
            self.data_folder = './datasets/LSA64_DVS_Right/LSA64_DVS_Right_splits/dataset_Right_2000/'
            self.width, self.height = 128, 128
            self.num_classes = 42
            self.class_mapping = {i: l for i, l in enumerate(range(self.num_classes))}
        elif dataset_name == 'DVS128':
            self.data_folder = './datasets/DvsGesture/clean_dataset_frames_2000/'
            self.width, self.height = 128, 128
            self.num_classes = 12 - len(classes_to_exclude)
            self.class_mapping = copy.deepcopy(DVS128_class_mapping)
            for c in classes_to_exclude: del self.class_mapping[c]
            self.class_mapping = { i:l[1] for i,l in enumerate(sorted(self.class_mapping.items(), key=lambda x:x[0])) }
        elif dataset_name == 'DailyAction':
            self.data_folder = './datasets/DailyActionDVS/clean_datasets/dataset_12000/'
            self.width, self.height = 128, 128
            self.num_classes = 12
            self.class_mapping = {i: l for i, l in enumerate(range(self.num_classes))}

    def train_dataloader(self):
        dt = EventDataset(self.data_folder + 'train/', time_step=self.time_step,
                          time_durations=self.time_durations,
                          validation=False,
                          preproc_polarity=self.preproc_polarity, crop_size=self.crop_size,
                          dataset_name=self.dataset_name, height=self.height, width=self.width,
                          augmentation_params=self.augmentation_params,
                          classes_to_exclude=self.classes_to_exclude)
        if self.custom_sampler:
            sampler = CustomBatchSampler(batch_size=self.batch_size, label_dict=dt.get_label_dict(),
                                         sample_repetitions=self.sample_repetitions)
            dl = DataLoader(dt, batch_sampler=sampler, num_workers=self.workers,
                            pin_memory=self.pin_memory)
        else:
            dl = DataLoader(dt, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.workers, pin_memory=self.pin_memory)
        return dl

    def val_dataloader(self):
        dt = EventDataset(self.data_folder + 'test/', time_step=self.time_step,
                          time_durations=self.time_durations,
                          validation=True,
                          preproc_polarity=self.preproc_polarity, crop_size=self.crop_size,
                          dataset_name=self.dataset_name, height=self.height, width=self.width,
                          augmentation_params=self.augmentation_params,
                          classes_to_exclude=self.classes_to_exclude)
        dl = DataLoader(dt, batch_size=(self.batch_size // 2) + 1, shuffle=False,
                        num_workers=self.workers, pin_memory=self.pin_memory)
        return dl

