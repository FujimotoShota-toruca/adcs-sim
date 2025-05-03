import os

# 作成したいディレクトリとファイルの構造を定義
structure = {
    "apps": ["main_simulation.py", "config.yaml", "interface.py"],
    "adcs_library": [
        "init.py"
    ],
    "adcs_library/datatypes": [
        "input_types.py", "output_types.py", "config_model.py"
    ],
    "adcs_library/orbit": [
        "propagate.py", "tle_parser.py"
    ],
    "adcs_library/attitude": [
        "dynamics.py", "transforms.py"
    ],
    "adcs_library/environment": [
        "magnetic_field.py", "air_drag.py", "gravity.py"
    ],
    "adcs_library/control": [
        "bdot.py", "quaternion_feedback.py"
    ],
    "adcs_library/utility": [
        "logger.py", "progress.py", "line_notify.py"
    ],
    "tests": [
        "test_attitude.py", "test_orbit.py", "test_environment.py"
    ],
    "docs": [
        "README.md", "architecture.md", "datatypes.md"
    ],
    "output/outputlog_YYYYMMDD_HHMMSS": [
        "attitude_log.csv", "orbit_log.csv", "config_dump.yaml", "summary.txt"
    ]
}

def create_structure(base_path="."):
    for folder, files in structure.items():
        dir_path = os.path.join(base_path, folder)
        os.makedirs(dir_path, exist_ok=True)
        for file in files:
            file_path = os.path.join(dir_path, file)
            if not os.path.exists(file_path):
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("# " + file + "\n")

    # ルートのその他ファイル
    for root_file in ["requirements.txt", ".gitignore"]:
        path = os.path.join(base_path, root_file)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("# " + root_file + "\n")

    print("プロジェクト構造を生成しました。")

if __name__ == "__main__":
    create_structure()
