import numpy as np
import pandas as pd
import json

def flatten_record_named(record: dict, array_key_names: dict) -> dict:
    """キーに応じて名前をつけて ndarray を展開"""
    flat = {}
    for key, val in record.items():
        if isinstance(val, np.ndarray):
            name_list = array_key_names.get(key)
            if name_list and len(name_list) == len(val):
                flat.update({f"{key}_{name}": v for name, v in zip(name_list, val)})
            else:
                flat.update({f"{key}_{i}": v for i, v in enumerate(val)})
        else:
            flat[key] = val
    return flat