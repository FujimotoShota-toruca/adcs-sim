import yaml
import numpy as np
from datetime import datetime
from pathlib import Path
from dacite import from_dict, Config as DaciteConfig
from . import datatype_input

def parse_datetime(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")

def list_to_ndarray(lst) -> np.ndarray:
    return np.array(lst)

def load_config(path: str) -> datatype_input.Config:
    with open(path, "r", encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Timeセクションの変換（duration前提）
    time_data = data["time"]
    data["time"]["start"] = parse_datetime(time_data["start"])

    type_hooks = {
        datetime: lambda x: x,
        np.ndarray: list_to_ndarray
    }

    return from_dict(
        data_class=datatype_input.Config,
        data=data,
        config=DaciteConfig(type_hooks=type_hooks)
    )