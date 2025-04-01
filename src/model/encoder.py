from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
import torch


class Encoder():
    def __init__(self, modal=['t', 'v', 'a']):
        self.modal = modal
        self.device = 'cuda'

        clip_type = {}
        if 't' in self.modal:
            self.tokenizer = LanguageBindImageTokenizer.from_pretrained(f'lb203/LanguageBind_Image',
                                                                   cache_dir='./cache_dir/tokenizer_cache_dir')
        if 'v' in self.modal:
            clip_type['video'] = 'LanguageBind_Video_FT'
        if 'a' in self.modal:
            clip_type['audio'] = 'LanguageBind_Audio_FT'

        self.encoder_model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
        self.encoder_model = self.encoder_model.to(self.device)
        self.encoder_model.eval()

        self.modality_transform = {c: transform_dict[c](self.encoder_model.modality_config[c]) for c in clip_type.keys()}

    def transform(self, data):
        for k, v in data.items():
            if k == 'language':
                data[k] = self.tokenizer(v, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
            else:
                data[k] = self.modality_transform[k](v)

        return data

    def extract(self, data):
        for k, v in data.items():
            data[k] = to_device(v, self.device)

        with torch.no_grad():
            embedding = self.encoder_model(data)

        return embedding
