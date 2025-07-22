import pickle
import numpy as np
import random
from tqdm import tqdm
import pandas as pd


def simulate_missing_modality(n_samples,
                              missing_type, 
                              missing_ratio,
                              modal,
                              seed=2025):
    """
    Simulate multimodal missing data
    :param n_samples: the number of training data
    :param missing_type: missing type ("text", "image", "mixed")
    :param missing_ratio: missing ratio (decimal between 0-1)
    :return: index list of missing data
    """

    # 0: avaliable, 1: text missing, 2: video missing, 3: audio missing
    missing_type_index = {'language': 1, 'video': 2, 'audio': 3, 'image': 4}
    missing_count = int(n_samples * missing_ratio)
    missing_index_list = [0 for _ in range(n_samples)]

    random.seed(seed)
    np.random.seed(seed)

    # Select sample indexes for missing
    missing_indices = random.sample(range(n_samples), missing_count)
    # Generate missing details.
    if missing_type == 'mixed':
        modals_index = [missing_type_index[i] for i in modal[:-1]]
        for idx in tqdm(missing_indices, desc=f'missing type {missing_type}, ratio {missing_ratio}'):
            missing_index_list[idx] = random.choice(modals_index)
    else:
        for idx in tqdm(missing_indices, desc=f'missing type {missing_type}, ratio {missing_ratio}'):
            missing_index_list[idx] = missing_type_index[missing_type]
    
    return missing_index_list


if __name__ == '__main__':
    file_path = '/big-data/person/yuanjiang/MLMM_datasets/mvsa_multiple/label.csv'
    seed = 2025
    missing_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    modal = ['language', 'image']  # ['language', 'video', 'audio', 'image']
    modal.append('mixed')

    df = pd.read_csv(file_path, converters={'clip_id': str})

    missing_list = {}
    for dataset in ['train', 'valid', 'test']:
        n_samples = len(df[df['mode'] == dataset]['annotation'])

        missing_list[dataset] = {}
        for missing_type in modal:
            missing_list[dataset][missing_type] = {}
            for ratio in missing_ratio:
                missing_list[dataset][missing_type][ratio] = simulate_missing_modality(n_samples, missing_type, ratio, modal, seed)

            # different ratio shared same seed.
            seed += 1

    save_path = "/".join(file_path.split("/")[:-1]) + "/" + "missing_index.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(missing_list, f)
