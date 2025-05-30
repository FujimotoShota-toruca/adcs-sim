import yaml
import os

def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")

    with open(path, 'r', encoding='utf-8') as file:
        try:
            return yaml.safe_load(file) or {}
        except yaml.YAMLError as e:
            raise RuntimeError(f"YAMLの読み込みエラー: {e}")
