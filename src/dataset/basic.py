import torch
from pathlib import Path
from torch.utils.data import Dataset


class BasicDataset(Dataset):
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
        """
        Args:
            root (str): root directory of the dataset
            metafiles (dict): dictionary of metafiles.
            mm_transforms (callable, optional): multimodal transforms.
            seed (int): random seed.
            isMissing (bool): whether the dataset is missing.
            missing_config (str): path to the missing configuration file.
            keep_raw (bool): whether to keep the raw data.
            merge_metadata (bool): whether to merge metadata.
        """
        self.root = Path(root)
        self.data = self.load_data(metafiles, *args, **kwargs)
        self.seed = seed
        self.mm_transforms = mm_transforms
        self.isMissing = isMissing
        self.missing_config = missing_config
        self.keep_raw = keep_raw
        self.merge_metadata = merge_metadata
        

    def __len__(self):
        return len(self.data)
    
    def load_data(self, *args, **kwargs):
        """load data from a file or a database"""
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
    
    def transform(self, data):
        """transform data"""
        raise NotImplementedError
    