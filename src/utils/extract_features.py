import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append('.')
import os

import numpy as np
from src.model.encoder import Encoder

def calculate_statistics(data_list):
    """计算列表数据的均值和中位数"""
    data_array = np.array(data_list)
    return {
        'mean': np.mean(data_array, axis=0).tolist(),
        'median': np.median(data_array, axis=0).tolist()
    }


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
    result['statistics'] = {m: {} for m in ['train', 'valid', 'test']}



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
    for mode in ['train', 'valid', 'test']:
        for modal_key in ['language','video', 'audio']:
            if result[mode][modal_key]:  # 确保列表不为空
                result['statistics'][mode][modal_key] = calculate_statistics(result[mode][modal_key])


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
    result['statistics'] = {m: {} for m in ['train', 'valid', 'test']}
    # 为每个模态创建统计信息字典


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
    for mode in ['train', 'valid', 'test']:
        for modal_key in ['video', 'audio']:
            if result[mode][modal_key]:  # 确保列表不为空
                result['statistics'][mode][modal_key] = calculate_statistics(result[mode][modal_key])

    # 将统计信息添加到结果中
    return result


if __name__ == '__main__':
    dataset = 'eNTERFACE'
    modal = ['v', 'a']
    csv_path = '/big-data/person/yuanjiang/MLMM_datasets/eNTERFACE/label.csv'

    if dataset == 'sims' or dataset == 'mosi':
        result = sims_mosi_extract(csv_path, modal)
    elif dataset == 'eNTERFACE':
        result = eNTERFACE_extract(csv_path, modal)

    with open("/".join(csv_path.split('/')[:-1]) + '/' + 'embedding.pkl', 'wb') as f:
        pickle.dump(result, f)
