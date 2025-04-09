import pandas as pd
from pathlib import Path
import random


def eNTERFACE(data_dir):
    data = {name: [] for name in ['avi_path', 'annotation']}

    for file_path in Path(data_dir).rglob("*"):  # `*` 匹配所有文件和子目录
        if file_path.is_file() and not any(
            part.startswith('.') for part in file_path.parts) and file_path.suffix == '.avi':
            data['avi_path'].append(str(file_path))
            data['annotation'].append(str(file_path).split('/')[-3])
    data['mode'] = random.choices(['train', 'valid', 'test'], weights=[8, 1, 1], k=len(data['annotation']))

    return data


if __name__ == '__main__':
    dataset = 'eNTERFACE'
    data_dir = '/big-data/person/yuanjiang/MLMM_datasets/eNTERFACE/data'

    if dataset == 'eNTERFACE':
        data = eNTERFACE(data_dir)

    save_path = data_dir.replace('/data', '/label.csv')
    pd.DataFrame(data).to_csv(save_path, index=False)
