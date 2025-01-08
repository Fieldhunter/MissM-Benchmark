import torch

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