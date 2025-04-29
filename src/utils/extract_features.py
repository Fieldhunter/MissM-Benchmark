import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append('.')
import os

import numpy as np
from src.model.encoder import Encoder


def calculate_statistics(data_list, type):
    """计算列表数据的均值和中位数"""
    data_array = np.array(data_list)

    if type == 'mean':
        return np.mean(data_array, axis=0).tolist()
    else:
        return np.median(data_array, axis=0).tolist()


def sims_mosi_extract(csv_path, modal=['t', 'v', 'a']):
    key_list = ['language', 'video', 'audio', 'label', 'label_T', 'label_A', 'label_V', 'annotation']
    data_path = "/".join(csv_path.split('/')[:-1])

    df = pd.read_csv(csv_path, converters={'clip_id': str})
    data = {}
    data['language'] = list(df['text'])
    data['video'] = list(data_path + '/data/' + df['video_id'] + '/' + df['clip_id'] + '.mp4')
    data['audio'] = list(data_path + '/wav/' + df['video_id'] + '/' + df['clip_id'] + '.wav')
    data['label'] = list(df['label'])
    data['label_T'] = list(df['label_T'])
    data['label_A'] = list(df['label_A'])
    data['label_V'] = list(df['label_V'])
    data['annotation'] = list(df['annotation'])
    data['mode'] = list(df['mode'])

    encoder = Encoder(modal=modal)
    result = {m: {name: [] for name in key_list} for m in ['train', 'valid', 'test']}
    result['statistics'] = {}

    for index in tqdm(range(len(data['label']))):
        inputs = {
            'language': data['language'][index],
            'video': data['video'][index],
            'audio': data['audio'][index]
        }
        embedding = encoder.extract(encoder.transform(inputs))

        for key in key_list:
            if key in ('language', 'video', 'audio'):
                result[data['mode'][index]][key].append(embedding[key].squeeze(0).cpu().tolist())
            else:
                result[data['mode'][index]][key].append(data[key][index])

    # 计算每个模态的统计信息
    for type in ['mean', 'median']:
        result['statistics'][type] = {}
        for modal_key in ['language', 'video', 'audio']:
            result['statistics'][type][modal_key] = calculate_statistics(result['train'][modal_key], type)

    return result


def eNTERFACE_extract(csv_path, modal=['v', 'a']):
    key_list = ['video', 'audio', 'label']

    df = pd.read_csv(csv_path)
    data = {}
    data['video'] = list(df['avi_path'])
    data['audio'] = list(df['avi_path'].str.replace('.avi', '.wav', regex=False).str.replace('/data/', '/wav/', regex=False))
    data['mode'] = list(df['mode'])

    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(list(df['annotation']))

    encoder = Encoder(modal=modal)
    result = {m: {name: [] for name in key_list} for m in ['train', 'valid', 'test']}
    result['class_index'] = label_encoder.classes_
    result['statistics'] = {}

    for index in tqdm(range(len(data['label']))):
        inputs = {
            'video': data['video'][index],
            'audio': data['audio'][index]
        }
        embedding = encoder.extract(encoder.transform(inputs))

        for key in key_list:
            if key in ('video', 'audio'):
                result[data['mode'][index]][key].append(embedding[key].squeeze(0).cpu().tolist())
            else:
                result[data['mode'][index]][key].append(data[key][index])

        # 计算每个模态的统计信息
        for type in ['mean', 'median']:
            result['statistics'][type] = {}
            for modal_key in ['video', 'audio']:
                result['statistics'][type][modal_key] = calculate_statistics(result['train'][modal_key], type)

    return result


if __name__ == '__main__':
    dataset = 'eNTERFACE'
    csv_path = '/big-data/person/yuanjiang/MLMM_datasets/eNTERFACE/label.csv'

    if dataset == 'sims' or dataset == 'mosi':
        result = sims_mosi_extract(csv_path)
    elif dataset == 'eNTERFACE':
        result = eNTERFACE_extract(csv_path)

    with open("/".join(csv_path.split('/')[:-1]) + '/' + 'embedding.pkl', 'wb') as f:
        pickle.dump(result, f)
