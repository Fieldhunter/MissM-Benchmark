import torch
import cv2
import librosa

from .basic import BasicDataset


class CocoClassification(BasicDataset):
    def __init__(self, 
                 root,
                 metafiles, 
                 mm_transforms=None,
                 seed=2025,
                 isMissing=False,
                 missing_config: str=None,
                 keep_raw=False,
                 merge_metadata=False,
                 *args, **kwargs):
        super(CocoClassification, self).__init__(root, metafiles, mm_transforms, seed, isMissing, missing_config, keep_raw, merge_metadata, *args, **kwargs)
        
    def load_data(self, *args, **kwargs):
        """load data from a file or a database"""
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError


"""
目前假设的数据集组织形式为：每个样本的三个模态都是单个文件
"""


def read_raw_text(file_path='text.txt'):
    with open(file_path, "r") as file:
        content = file.read()

    return content


def read_raw_video(file_path='video.mp4', frame_interval=10):
    cap = cv2.VideoCapture(file_path)

    frame_list = []
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            frame_list.append(frame)
        frame_index += 1

    return frame_list  # numpy list


def read_raw_audio(file_path='audio.mp4'):
    signal, sample_rate = librosa.load(file_path)

    return signal, sample_rate  # signal shape: (n,)
