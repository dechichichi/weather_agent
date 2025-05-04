import yaml

def load_api_config():
    with open('api.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def load_model_config():
    with open('model_params.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config