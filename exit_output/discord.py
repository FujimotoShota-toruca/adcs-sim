# 参考：Qiita@t_A_M_u「サービス修了するLINE Notifyの代わりにDiscordを使ってみた」(https://qiita.com/t_A_M_u/items/96b7cc73107984278fe1)

import requests
import time
from datetime import datetime, timedelta

class DiscordSendMessage:

    def __init__(self, token: str, channel_id: str):
        self.__token = token
        self.__channel_id = channel_id
        self.__last_sent_time = None  # 前回送信時刻の記録

    def send_message(self, message: str):
        url = f"https://discord.com/api/v10/channels/{self.__channel_id}/messages"
        headers = {
            "Authorization": f"Bot {self.__token}",
            "Content-Type": "application/json"
        }
        data = {
            "content": message
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=5)  # タイムアウト追加
            if response.status_code in [200, 204]:
                print(f"[{datetime.now()}] 送信完了: {message}")
            else:
                print(f"[{datetime.now()}] 送信失敗: {response.status_code}, {response.text}")
        except requests.exceptions.RequestException as e:
            # ネットワークエラーなどをキャッチ
            print(f"[{datetime.now()}] メッセージ送信エラー: {e}")

    def maybe_send_progress(self, message: str, interval_minutes: int = 10):
        """
        指定分数以上経過していればメッセージを送信。
        シミュレーションループの中で使う。
        """
        now = datetime.now()
        if (self.__last_sent_time is None or 
            now - self.__last_sent_time >= timedelta(minutes=interval_minutes)):
            self.send_message(message)
            self.__last_sent_time = now