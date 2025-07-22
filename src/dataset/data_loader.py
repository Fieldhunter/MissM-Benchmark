import torch
from torch.utils.data import DataLoader
import pickle
from copy import deepcopy
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import random
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path


missing_type_index = {'language': 1, 'video': 2, 'audio': 3, 'image': 4}
index_missing_type = {1: 'language', 2: 'video', 3: 'audio', 4: 'image'}



class MMDataset_sims_mosi(torch.utils.data.Dataset):
    def __init__(self, df, data_path, tokenizer, modality_transform, labels, mode='train', missing=False, missing_index=None, retrieval=False, training_loader=None):
        key_list = ['language', 'video', 'audio', 'reg_label', 'label_T', 'label_A', 'label_V', 'annotation', 'label']

        self.data = {}
        self.data['language'] = list(df['text'])
        self.data['video'] = list(data_path + '/data/' + df['video_id'] + '/' + df['clip_id'] + '.mp4')
        self.data['audio'] = list(data_path + '/wav/' + df['video_id'] + '/' + df['clip_id'] + '.wav')
        self.data['reg_label'] = list(df['label'])
        self.data['label_T'] = list(df['label_T'])
        self.data['label_A'] = list(df['label_A'])
        self.data['label_V'] = list(df['label_V'])
        self.data['annotation'] = list(df['annotation'])
        self.data['mode'] = list(df['mode'])

        self.data['label'] = labels

        self.mode = mode
        self.retrieval = retrieval
        if retrieval:
            if self.mode == 'test':
                self.training_loader = training_loader
            else:
                self.label2indices = {}
                for idx, label in enumerate(self.data['label']):
                    if label not in self.label2indices:
                        self.label2indices[label] = []
                    self.label2indices[label].append(idx)

        self.missing = missing
        self.missing_index = missing_index if missing and missing_index else [0 for _ in range(len(self.data['label']))]

        self.tokenizer = tokenizer
        self.modality_transform = modality_transform

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, index):
        if self.mode == 'train' and self.missing:
            missing_index = random.choice([0, 1, 2, 3])
        else:
            missing_index = self.missing_index[index]

        data = {
            'language': self.data['language'][index],
            'video': self.data['video'][index],
            'audio': self.data['audio'][index]
        }

        if self.retrieval and missing_index != 0:
            if self.mode == 'test':
                data[index_missing_type[missing_index]] = self.training_loader.get_retrieval_data(index, self.data['label'][index], missing_index)
            else:
                data[index_missing_type[missing_index]] = self.get_retrieval_data(index, self.data['label'][index], missing_index)
            missing_index = 0

        for k, v in data.items():
            if k == 'language':
                data[k] = self.tokenizer(v, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
            else:
                data[k] = self.modality_transform[k](v)
        
        label = {
            'label': self.data['label'][index],
            'label_T': self.data['label_T'][index],
            'label_A': self.data['label_A'][index],
            'label_V': self.data['label_V'][index],
            'annotation': self.data['annotation'][index]
        }
    
        return data, label, missing_index
    
    def get_retrieval_data(self, current_index, label, missing_index):
        complete_index = random.choice(self.label2indices[label])
        while complete_index == current_index:
            complete_index = random.choice(self.label2indices[label])
        
        return self.data[index_missing_type[missing_index]][complete_index]


class MMDataset_eNTERFACE(torch.utils.data.Dataset):
    def __init__(self, df, data_path, tokenizer, modality_transform, labels, mode='train', missing=False, missing_index=None, retrieval=False, training_loader=None):
        key_list = ['video', 'audio', 'label']

        self.data = {}
        self.data['video'] = list(df['avi_path'])
        self.data['audio'] = list(df['avi_path'].str.replace('.avi', '.wav', regex=False).str.replace('/data/', '/wav/', regex=False))
        self.data['mode'] = list(df['mode'])

        self.data['label'] = labels

        self.mode = mode
        self.retrieval = retrieval
        if retrieval:
            if self.mode == 'test':
                self.training_loader = training_loader
            else:
                self.label2indices = {}
                for idx, label in enumerate(self.data['label']):
                    if label not in self.label2indices:
                        self.label2indices[label] = []
                    self.label2indices[label].append(idx)

        self.missing = missing
        self.missing_index = missing_index if missing and missing_index else [0 for _ in range(len(self.data['label']))]

        self.tokenizer = tokenizer
        self.modality_transform = modality_transform

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, index):
        if self.mode == 'train' and self.missing:
            missing_index = random.choice([0, 2, 3])
        else:
            missing_index = self.missing_index[index]
        
        data = {
            'video': self.data['video'][index],
            'audio': self.data['audio'][index]
        }
        
        if self.retrieval and missing_index != 0:
            if self.mode == 'test':
                data[index_missing_type[missing_index]] = self.training_loader.get_retrieval_data(index, self.data['label'][index], missing_index)
            else:
                data[index_missing_type[missing_index]] = self.get_retrieval_data(index, self.data['label'][index], missing_index)
            missing_index = 0
    
        for k, v in data.items():
            data[k] = self.modality_transform[k](v)
        
        label = {'label': self.data['label'][index]}        
    
        return data, label, missing_index

    def get_retrieval_data(self, current_index, label, missing_index):
        complete_index = random.choice(self.label2indices[label])
        while complete_index == current_index:
            complete_index = random.choice(self.label2indices[label])
        
        return self.data[index_missing_type[missing_index]][complete_index]


class MMDataset_AVE(torch.utils.data.Dataset):
    def __init__(self, df, data_path, tokenizer, modality_transform, labels, mode='train', missing=False, missing_index=None, retrieval=False, training_loader=None):
        key_list = ['path', 'label']

        self.data = {}
        self.data['video'] = list(df['path'])
        self.data['audio'] = list(df['path'].str.replace('.mp4', '.wav', regex=False).str.replace('_split/', '_split_wav/', regex=False))
        self.data['mode'] = list(df['mode'])

        self.data['label'] = labels

        self.mode = mode
        self.retrieval = retrieval
        if retrieval:
            if self.mode == 'test':
                self.training_loader = training_loader
            else:
                self.label2indices = {}
                for idx, label in enumerate(self.data['label']):
                    if label not in self.label2indices:
                        self.label2indices[label] = []
                    self.label2indices[label].append(idx)

        self.missing = missing
        self.missing_index = missing_index if missing and missing_index else [0 for _ in range(len(self.data['label']))]

        self.tokenizer = tokenizer
        self.modality_transform = modality_transform

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, index):
        if self.mode == 'train' and self.missing:
            missing_index = random.choice([0, 2, 3])
        else:
            missing_index = self.missing_index[index]
        
        data = {
            'video': self.data['video'][index],
            'audio': self.data['audio'][index]
        }
        
        if self.retrieval and missing_index != 0:
            if self.mode == 'test':
                data[index_missing_type[missing_index]] = self.training_loader.get_retrieval_data(index, self.data['label'][index], missing_index)
            else:
                data[index_missing_type[missing_index]] = self.get_retrieval_data(index, self.data['label'][index], missing_index)
            missing_index = 0
        
        for k, v in data.items():
            data[k] = self.modality_transform[k](v)
        
        label = {'label': self.data['label'][index]}
    
        return data, label, missing_index

    def get_retrieval_data(self, current_index, label, missing_index):
        complete_index = random.choice(self.label2indices[label])
        while complete_index == current_index:
            complete_index = random.choice(self.label2indices[label])
        
        return self.data[index_missing_type[missing_index]][complete_index]


class MMDataset_mvsa(torch.utils.data.Dataset):
    def __init__(self, df, data_path, tokenizer, modality_transform, labels, mode='train', missing=False, missing_index=None, retrieval=False, training_loader=None):
        key_list = ['ID']

        self.data = {}
        self.data['language'] = list(df['language'])
        self.data['image'] = list(data_path + '/data/' + df['ID'].apply(str) + '.jpg')
        self.data['mode'] = list(df['mode'])

        self.data['label'] = labels

        self.mode = mode
        self.retrieval = retrieval
        if retrieval:
            if self.mode == 'test':
                self.training_loader = training_loader
            else:
                self.label2indices = {}
                for idx, label in enumerate(self.data['label']):
                    if label not in self.label2indices:
                        self.label2indices[label] = []
                    self.label2indices[label].append(idx)

        self.missing = missing
        self.missing_index = missing_index if missing and missing_index else [0 for _ in range(len(self.data['label']))]

        self.tokenizer = tokenizer
        self.modality_transform = modality_transform

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, index):
        if self.mode == 'train' and self.missing:
            missing_index = random.choice([0, 1, 4])
        else:
            missing_index = self.missing_index[index]
        
        data = {
            'language': self.data['language'][index],
            'image': self.data['image'][index]
        }
        
        if self.retrieval and missing_index != 0:
            if self.mode == 'test':
                data[index_missing_type[missing_index]] = self.training_loader.get_retrieval_data(index, self.data['label'][index], missing_index)
            else:
                data[index_missing_type[missing_index]] = self.get_retrieval_data(index, self.data['label'][index], missing_index)
            missing_index = 0
    
        for k, v in data.items():
            if k == 'language':
                data[k] = self.tokenizer(v, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
            else:
                data[k] = self.modality_transform[k](v)
        
        label = {'label': self.data['label'][index]}        
    
        return data, label, missing_index


def training_loader(args, csv_path, tokenizer, modality_transform):
    if args.datasetName == 'sims':
        dataset = MMDataset_sims_mosi
    elif args.datasetName == 'mosi':
        dataset = MMDataset_sims_mosi
    elif args.datasetName == 'eNTERFACE':
        dataset = MMDataset_eNTERFACE
    elif args.datasetName == 'AVE':
        dataset = MMDataset_AVE
    elif args.datasetName == 'mvsa':
        dataset = MMDataset_mvsa

    data_path = "/".join(csv_path.split('/')[:-1])
    df = pd.read_csv(csv_path, converters={'clip_id': str})
    train_df = df[df['mode'] == 'train']
    valid_df = df[df['mode'] == 'valid']

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(list(df['annotation']))

    train_data = dataset(train_df, data_path, tokenizer, modality_transform, labels[df['mode'] == 'train'], 'train', args.train_missing, retrieval=args.fusion_type=='retrieval')
    val_data = dataset(valid_df, data_path, tokenizer, modality_transform, labels[df['mode'] == 'valid'], 'val', False)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, sampler=DistributedSampler(train_data))
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, sampler=DistributedSampler(val_data))

    return train_loader, val_loader, len(label_encoder.classes_)


def testing_loader(args, csv_path, tokenizer, modality_transform):
    if args.datasetName == 'sims':
        dataset = MMDataset_sims_mosi
        missing_path = '/big-data/person/yuanjiang/MLMM_datasets/CH-SIMS/missing_index.pkl'
    elif args.datasetName == 'mosi':
        dataset = MMDataset_sims_mosi
        missing_path = '/big-data/person/yuanjiang/MLMM_datasets/CMU_MOSI/missing_index.pkl'
    elif args.datasetName == 'eNTERFACE':
        dataset = MMDataset_eNTERFACE
        missing_path = '/big-data/person/yuanjiang/MLMM_datasets/eNTERFACE/missing_index.pkl'
    elif args.datasetName == 'AVE':
        dataset = MMDataset_AVE
        missing_path = '/big-data/person/yuanjiang/MLMM_datasets/AVE_Dataset/missing_index.pkl'
    elif args.datasetName == 'mvsa':
        dataset = MMDataset_mvsa
        missing_path = '/big-data/person/yuanjiang/MLMM_datasets/mvsa_multiple/missing_index.pkl'

    data_path = "/".join(csv_path.split('/')[:-1])
    df = pd.read_csv(csv_path, converters={'clip_id': str})
    train_df = df[df['mode'] == 'train']
    test_df = df[df['mode'] == 'test']

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(list(df['annotation']))

    with open(missing_path, 'rb') as f:
        file = pickle.load(f)

    test_missing_index = {}
    for modal in args.test_missing_type:
        test_missing_index[modal] = {i: file['test'][modal][i] for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}

    train_data = dataset(train_df, data_path, tokenizer, modality_transform, labels[df['mode'] == 'train'], 'train', False, retrieval=args.fusion_type=='retrieval')
    test_data = {}
    for modal in args.test_missing_type:
        test_data[modal] = {i: dataset(test_df, data_path, tokenizer, modality_transform, labels[df['mode'] == 'test'], 'test', True, test_missing_index[modal][i], args.fusion_type=='retrieval', train_data) for i in test_missing_index[modal].keys()}
        test_data[modal][0.0] = dataset(test_df, data_path, tokenizer, modality_transform, labels[df['mode'] == 'test'], 'test', False)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    test_loader = {}
    for modal in args.test_missing_type:
        test_loader[modal] = {i: DataLoader(test_data[modal][i], batch_size=args.batch_size, shuffle=False) for i in test_data[modal].keys()}

    return train_loader, test_loader, len(label_encoder.classes_)


if __name__ == '__main__':
    import argparse
    from languagebind import LanguageBind, transform_dict, to_device, LanguageBindImageTokenizer
    from tqdm import tqdm
    import cv2
    import wave
    import contextlib


    def parse_args():
        parser = argparse.ArgumentParser()
        # 数据集相关参数
        parser.add_argument('--train_mode', type=str, default="classification", help='regression or classification')
        parser.add_argument('--datasetName', type=str, default='AVE', help='support mosi/sims/eNTERFACE/AVE')
        parser.add_argument('--csv_path', type=str, default='/big-data/person/yuanjiang/MLMM_datasets/AVE_Dataset/label.csv', help='')
        parser.add_argument('--modality_types', type=list, default=['video', 'audio'], help="['language', 'video', 'audio']/") # 严格遵守help顺序

        # 缺失相关参数
        parser.add_argument('--train_missing', type=bool, default=False)
        
        # 模型相关参数
        parser.add_argument('--feature_dims', type=int, default=768, help='the output dims of languagebind')
        parser.add_argument('--fusion_type', type=str, default='sum', help='sum/concat/regression/retrieval/intra_attention/inter_attention/graph_fusion/unified_graph/dedicated_dnn/[Distill_tea/MTD_stu/KL_stu]/self_distill')
        parser.add_argument('--fusion_dim', type=int, default=256)
        parser.add_argument('--dropout_prob', type=float, default=0.1)

        # 训练相关参数
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--num_epochs', type=int, default=50)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--weight_decay', type=float, default=0)
        parser.add_argument('--patience', type=int, default=8)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--save_path', type=str, default='checkpoints')
        parser.add_argument('--log_dir', type=str, default='logs')

        return parser.parse_args()
    

    args = parse_args()

    # 初始化Encoder
    clip_type = {}
    tokenizer = None
    if 'language' in args.modality_types:
        tokenizer = LanguageBindImageTokenizer.from_pretrained(f'lb203/LanguageBind_Image',
                                                                cache_dir='./cache_dir/tokenizer_cache_dir')
    if 'video' in args.modality_types:
        clip_type['video'] = 'LanguageBind_Video'
    if 'audio' in args.modality_types:
        clip_type['audio'] = 'LanguageBind_Audio'
    encoder_model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
    modality_transform = {c: transform_dict[c](encoder_model.modality_config[c]) for c in clip_type.keys()}

    # 加载数据集
    print("Loading data...")
    train_loader, _, _ = training_loader(args, args.csv_path, tokenizer, modality_transform)

    for data, label, missing_index in tqdm(train_loader):
        pass
