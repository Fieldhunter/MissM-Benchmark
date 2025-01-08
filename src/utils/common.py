import omegaconf
import os

def load_config(root: str = 'src/configs', config_name: str = '') -> omegaconf.DictConfig:
    # load basic configuration
    basic_path = os.path.join(root, 'basic-config.yaml')
    if not os.path.exists(basic_path):
        raise FileNotFoundError(f'File not found: {basic_path}')
    basic_config = omegaconf.OmegaConf.load(basic_path)
    
    # load specific configuration
    specific_path = os.path.join(root, config_name)
    if not os.path.exists(specific_path):
        raise FileNotFoundError(f'File not found: {specific_path}')
    specific_config = omegaconf.OmegaConf.load(specific_path)
    
    # merge configurations
    config = omegaconf.OmegaConf.merge(basic_config, specific_config)
    return config
    
    
if __name__ == '__main__':
    config = load_config(config_name='coco-classification.yaml')
    print(config)
    