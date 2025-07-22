import pandas as pd
from pathlib import Path
import random
import os


def eNTERFACE(data_dir):
    data = {name: [] for name in ['avi_path', 'annotation']}

    for file_path in Path(data_dir).rglob("*"):  # `*` 匹配所有文件和子目录
        if file_path.is_file() and not any(
            part.startswith('.') for part in file_path.parts) and file_path.suffix == '.avi':
            data['avi_path'].append(str(file_path))
            data['annotation'].append(str(file_path).split('/')[-3])

    train_num = int(len(data['annotation']) * 0.8)
    val_num = int(len(data['annotation']) * 0.1)
    test_num = len(data['annotation']) - train_num - val_num
    mode = ['train'] * train_num + ['valid'] * val_num + ['test'] * test_num
    random.shuffle(mode)
    data['mode'] = mode

    return data


def AVE(data_dir):
    all_df = []
    for mode in ['train', 'valid', 'test']:
        paths = []
        labels = []
        with open(os.path.join(data_dir, f'{mode}Set_split.txt'), 'r') as f:
            for line in f:
                line = line.strip()
                if not line: 
                    continue

                parts = line.split()
                if len(parts) < 2: 
                    continue
                paths.append(parts[0])
                labels.append(''.join(parts[1:]))
        all_df.append(pd.DataFrame({'path': paths, 'annotation': labels, 'mode': mode}))
    
    return pd.concat(all_df, ignore_index=True)


def mvsa(data_dir):
    data = {name: [] for name in ['ID', 'language', 'annotation']}
    with open(os.path.join(data_dir, 'labelResultAll_vote.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过第一行标题
            parts = line.strip().split()

            with open(os.path.join(data_dir, 'data', f'{parts[0]}.txt'), 'r') as t:
                data['language'].append(t.readlines()[0].strip())
            data['ID'].append(parts[0])         # 第一项是ID
            data['annotation'].append(parts[-1])
    
    train_num = int(len(data['annotation']) * 0.8)
    val_num = int(len(data['annotation']) * 0.1)
    test_num = len(data['annotation']) - train_num - val_num
    mode = ['train'] * train_num + ['valid'] * val_num + ['test'] * test_num
    random.shuffle(mode)
    data['mode'] = mode

    return data


if __name__ == '__main__':
    dataset = 'mvsa'
    data_dir = '/big-data/person/yuanjiang/MLMM_datasets/mvsa_multiple'
    seed = 2025
    random.seed(seed)

    if dataset == 'eNTERFACE':
        data = eNTERFACE(data_dir)
        save_path = data_dir.replace('/data', '/label.csv')
    elif dataset == 'AVE':
        data = AVE(data_dir)
        save_path = os.path.join(data_dir, 'label.csv')
    elif dataset == 'mvsa':
        data = mvsa(data_dir)
        save_path = os.path.join(data_dir, 'label.csv')
        
    pd.DataFrame(data).to_csv(save_path, index=False)
