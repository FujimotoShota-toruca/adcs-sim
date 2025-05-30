import socket
import json
import time
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List
from multiprocessing import Process

HOST = '127.0.0.1'
PORT = 50007

@dataclass
class SimState:
    timestamp: float
    attitude: List[float]
    angular_velocity: List[float]

    def to_json(self):
        return json.dumps(asdict(self)) + "\n"

    @staticmethod
    def from_json(data: str):
        return SimState(**json.loads(data))

@dataclass
class ControlCommand:
    torque: List[float]

    def to_json(self):
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(data: str):
        return ControlCommand(**json.loads(data))

class Logger:
    def __init__(self):
        self.records = []

    def flatten_dict(self, d: dict) -> dict:
        flat = {}
        for k, v in d.items():
            if isinstance(v, list):
                for i, item in enumerate(v):
                    flat[f"{k}_{'xyz'[i] if i < 3 else i}"] = item
            else:
                flat[k] = v
        return flat

    def log(self, obj, extra: dict = None):
        record = self.flatten_dict(asdict(obj))
        if extra:
            record.update(self.flatten_dict(extra))
        self.records.append(record)

    def save_csv(self, path: str):
        df = pd.DataFrame(self.records)
        df.to_csv(path, index=False)
        print(f"[LOG] 保存完了: {path}")

def run_simulator(sim_steps: int, init_attitude: List[float], init_ang_vel: List[float], log_file: str):
    logger = Logger()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"[SIM] 接続待機中 on {HOST}:{PORT}")
        conn, addr = s.accept()
        with conn:
            print(f"[SIM] 接続されました: {addr}")
            for step in range(sim_steps):
                sim_state = SimState(
                    timestamp=time.time(),
                    attitude=[init_attitude[0] + 0.1 * step, init_attitude[1], init_attitude[2]],
                    angular_velocity=init_ang_vel
                )

                conn.sendall(sim_state.to_json().encode())

                try:
                    data = conn.recv(1024).decode()
                    ctrl_cmd = ControlCommand.from_json(data)
                except (json.JSONDecodeError, ConnectionResetError):
                    print("[SIM] 受信エラー")
                    break

                logger.log(sim_state, extra={
                    "step": step,
                    "torque": ctrl_cmd.torque
                })

                print(f"[SIM] Step {step} トルク: {ctrl_cmd.torque}")
                time.sleep(0.01)

    logger.save_csv(log_file)

def run_controller(sim_steps: int):
    time.sleep(0.5)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        for _ in range(sim_steps):
            data = s.recv(1024).decode()
            sim_state = SimState.from_json(data)
            print(f"[CTRL] 姿勢: {sim_state.attitude}")

            torque = [-0.01 * sim_state.attitude[0], 0.0, 0.0]
            cmd = ControlCommand(torque=torque)
            s.sendall(cmd.to_json().encode())

def main():
    # === 設定 ===
    sim_steps = 100
    initial_attitude = [0.0, 0.0, 0.0]
    initial_angular_velocity = [0.01, 0.0, 0.0]
    log_file = 'log.csv'

    # === プロセス起動 ===
    sim_proc = Process(target=run_simulator, args=(sim_steps, initial_attitude, initial_angular_velocity, log_file))
    ctrl_proc = Process(target=run_controller, args=(sim_steps,))

    sim_proc.start()
    ctrl_proc.start()

    sim_proc.join()
    ctrl_proc.join()

if __name__ == "__main__":
    main()
