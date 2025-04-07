import yaml
from types import SimpleNamespace

def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return _dict_to_namespace(cfg)

def _dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    return d
