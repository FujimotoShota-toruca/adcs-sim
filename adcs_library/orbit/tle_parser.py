# tle_parser.py

from pathlib import Path
from typing import List, Tuple
from skyfield.api import EarthSatellite, load

def parse_tle(tle_file_path: str) -> List[Tuple[str, EarthSatellite]]:
    """
    TLEファイルを読み取り、衛星名とSkyfieldのEarthSatelliteオブジェクトを返す。

    Parameters:
        tle_file_path (str): TLEファイルのパス

    Returns:
        List[Tuple[str, EarthSatellite]]: (衛星名, Skyfieldオブジェクト) のリスト
    """
    tle_path = Path(tle_file_path)

    if not tle_path.exists():
        raise FileNotFoundError(f"TLEファイルが存在しません: {tle_file_path}")

    with tle_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    satellites = []
    ts = load.timescale()

    i = 0
    while i < len(lines) - 2:
        if lines[i][0].isdigit():  # 名前が省略されている
            name = f"UNKNOWN_{i}"
            tle1, tle2 = lines[i], lines[i+1]
            i += 2
        else:
            name, tle1, tle2 = lines[i], lines[i+1], lines[i+2]
            i += 3

        satellite = EarthSatellite(tle1, tle2, name, ts)
        satellites.append((name, satellite))

    return satellites
