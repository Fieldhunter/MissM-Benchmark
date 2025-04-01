import pickle
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append('.')

from src.model.encoder import Encoder


def sims_extract(csv_path):
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

    encoder = Encoder(modal=['t', 'v', 'a'])
    result = {m: {name: [] for name in key_list} for m in ['train', 'valid', 'test']}
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

    return result


def save_pkl(csv_path, result):
    with open("/".join(csv_path.split('/')[:-1])+'/'+'embedding.pkl', 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    dataset = 'sims'
    csv_path = '/big-data/person/yuanjiang/MLMM_datasets/CH-SIMS/label.csv'

    if dataset == 'sims':
        result = sims_extract(csv_path)

    save_pkl(csv_path, result)
