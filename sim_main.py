# 軌道計算系
from skyfield.api import load
from skyfield.api import EarthSatellite
from skyfield.api import wgs84
from skyfield.api import Timescale
from skyfield.api import load, Topos
# 数学/データ解析
import pandas as pd
import numpy as np
# 日付関連
import datetime
import os
# adcsライブラリ
import adcs_lib.interface_input as input_if
import adcs_lib.interface_output as output_if
import adcs_lib.term_interface as term
import adcs_lib.math_rotation as rotation
import adcs_lib.math_utilities as math
import adcs_lib.physics_attitude as attitude
import adcs_lib.physics_environment as environment
import adcs_lib.adcs_function as adcs
# 外部システムとのインタフェース
import exit_output.discord as discord
import exit_output.loader as loader
import exit_output.influx_writer as influxDB

# 可視化のやつ
#influx = influxDB.InfluxWriter()

# 外部output用クラス(無くても動く:discord_bodの部分はコメントアウト推奨)
#ext_config = loader.load_config('.\\exit_output\\ext_config.yaml')
#discord_token = ext_config['discord']['token']
#discord_ch = ext_config['discord']['ch_id']
# sampleのところに入れてもいいし,直接トークンとチャンネルを書いてもいい
#discord_bot = discord.DiscordSendMessage(discord_token, discord_ch)

# シミュレーション条件読み込み
config = input_if.load_config('.\\input\\sample_config.yaml')

# -------------------初期値設定ここから-------------------
# 慣性テンソル系の定義
inertia = config.satellite_structure.inertia_tensor
inertia_inv = np.linalg.inv(inertia)

# TLEセット
tle_lins = config.tle
tle_lines = tle_lins.split('\n') # 改行コードで分割する

# TLE分析
ts = load.timescale()
satellite = EarthSatellite(tle_lines[1], tle_lines[2], tle_lines[0], ts)
#平均運動を抜き出して1週あたりにかかる時間を計算
mean_motion = float(tle_lines[2][52:63]) # 周回数/day
rad_sec = mean_motion * 2 * np.pi /24.0 /60.0 /60.0 # rad/sec

# configファイルからシミュレーション開始時刻を抽出
epoch_time = config.time.start.replace(tzinfo=datetime.timezone.utc)

# !シミュレーション時間を定義
dt = config.time.step
duration = config.time.duration
total_step = (int)(duration/dt) # 総量/幅をループ回数とする
print(total_step)

# !初期状態
angular_velocity = config.initial_attitude_parameters.angular_velocity # !初期角速度
attitude_quaternion = config.initial_attitude_parameters.quaternion # !初期クォータニオン

# カウント
count1 = 1.0
count2 = 2.0
# カウント制御
count1sec = 1.0
count2sec = 2.0
# 磁気トルカ駆動判定
MTQ_con = False
# 磁気トルカ駆動設定
MTQ_vector = np.array([0,0,0])
# センサ
sat_mag_sen = np.array([0,0,0])
sat_gyr_sen = np.array([0,0,0])

#制御変数
Quaternion_ref = np.array([0,0,0,1]) #基準クォータニオン
gyro_ref = np.array([0,0,0]) # 基準角速度
Quaternion_err = np.array([0,0,0,1]) #偏差クォータニオン

euler = rotation.dcm2euler(np.array([[1,0,0],[0,1,0],[0,0,1]]))
q_back = np.array([0,0,0,1])
Quaternion_err_buf = np.array([0,0,0,1])
qback = np.array([0,0,0,1])

d_gyro_err = np.array([0,0,0])

controal_torque = np.array([0, 0, 0,])

dd = np.array([0,0,0])
td = np.array([0,0,0])
tp = np.array([0,0,0])
ff = np.array([0,0,0])

sat_mag_sen_b = np.array([0, 0, 0])

# 発生磁気モーメント平滑化:(論文)
inc = satellite.model.inclo
kap = np.sqrt(1+3*(np.sin(inc)**2))
D_mat = np.diag([(4-kap)/3, 1-(np.cos(inc)**2)/kap, ((4/kap)-1)/3])
D_inv = np.linalg.inv(D_mat)
# 出力平滑化フィルタ
alpha = 0.5
t_filterd = np.array([0.0, 0.0, 0.0])

# 記録リスト
ARRAY_KEY_NAMES = {
    "err_euler": ["roll_err", "pitch_err", "yaw_err"],
    "err_vec": ["x", "y", "z"],
    "Quaternion_err": ["w", "x", "y", "z"],
    "angular_velocity": ["wx", "wy", "wz"],
    "mtq_output": ["x", "y", "z"],
    "gyro_ref": ["x", "y", "z"],
    "Quaternion_ref": ["w", "x", "y", "z"],
    "attitude_quaternion": ["w", "x", "y", "z"],
    "controal_torque": ["x", "y", "z"],
    "tp": ["x", "y", "z"],
    "td": ["x", "y", "z"],
    "ff": ["x", "y", "z"],
    "torque": ["x", "y", "z"],
    "air_torque": ["x", "y", "z"],
    "gra_trq": ["x", "y", "z"],
    "sat_mag_sen": ["x", "y", "z"],
    "B_enu": ["x", "y", "z"],
    "B_ecef": ["x", "y", "z"],
    "B_ECI": ["x", "y", "z"],
    "t": ["x", "y", "z"]
}

# ログリスト
raw_log = []

# -------------------初期値設定ここまで-------------------
# 軌道伝搬→環境外乱計算→姿勢伝搬
for elapsed_time in range(total_step):

    # UI通知処理
    # 進捗表示
    term.update_progress_bar(elapsed_time + 1, total_step)
    # discord 通知
    progress = elapsed_time / total_step * 100
    message = f"Step {elapsed_time}/{total_step}（{progress:.1f}%）完了"
    #discord_bot.maybe_send_progress(message, interval_minutes=10)

    # 時刻更新
    sim_time = config.time.start + datetime.timedelta(seconds=elapsed_time*dt)
    t = ts.from_datetime(sim_time.replace(tzinfo=datetime.timezone.utc))
    date = t.utc_datetime()

    # 軌道伝搬
    geocentric = satellite.at(t)
    # 情報抽出
    lat, lon = wgs84.latlon_of(geocentric)
    Alt = wgs84.height_of(geocentric)
    eci_pos = geocentric.position.m #メートル法で位置を算出
    eci_vel = geocentric.velocity.m_per_s #メートル法で速度を算出

    # 地磁気トルクの計算
    B_enu, B_ecef, B_ECI = environment.earth_magnetic_field_eci(sim_time, lon.degrees, lat.degrees, Alt.km)
    eci2body = (rotation.quaternion2dcm(attitude_quaternion))
    earth_mag = eci2body @ B_ECI
    controal_torque = np.cross(MTQ_vector, earth_mag)
    # 大気抵抗トルクの計算
    body_velocity = eci2body @ eci_vel # 速度ベクトル成分を座標変換
    air_torque = environment.atmospheric_torque_body(sim_time, lon.degrees, lat.degrees, Alt.km, 270, 270, 7.9, body_velocity, config.satellite_structure)
    # 重力傾斜モデル
    gra_trq = environment.gravity_gradient_torque_body(eci_pos, eci2body, inertia)
    
    # 外乱トルクを考慮
    torque = controal_torque + air_torque + gra_trq
    
    err_vec = adcs.quaternion_to_diff(Quaternion_err)
    err_euler = rotation.dcm2euler(rotation.quaternion2dcm(Quaternion_err))

    count1 = count1 + dt
    if count1 >= count1sec:
        count1 = dt
        record = {
            'elapsed_time[sec]': elapsed_time * dt,
            'elapsed_time[hour]': elapsed_time * dt / 3600,
            'err_euler': err_euler,
            'err_vec': err_vec,
            'quaternion_err': Quaternion_err,
            'angular_velocity': angular_velocity,
            'gyro_ref': gyro_ref,
            'mtq_output': MTQ_vector,
            'quaternion_ref': Quaternion_ref,
            'quaternion_i2b': attitude_quaternion,
            'control_torque': controal_torque,
            't': t_filterd,
            'tp': tp,
            'td': td,
            'ff': ff,
            'torque ': torque ,
            'air_torque': air_torque,
            'gravity_torque': gra_trq,
            'body_magnetic_field': sat_mag_sen,
            'enu_magnetic_field': B_enu,
            'ecef_magnetic_field': B_ecef,
            'eci_magnetic_field': B_ECI,
            'between_magnetic-bodyZaxis': np.rad2deg(np.acos(np.dot(np.array([0,0,-1]),sat_mag_sen/np.linalg.norm(sat_mag_sen))))
        }
        raw_log.append(record)

    # """
    # ２秒カウント countUp
    count2 = count2 + dt
    if count2 >= count2sec:
        count2 = dt # count reset
        if MTQ_con == True:
            #"""
            # 制御期間コンフィギュレーション設定
            ##print("C")
            # B-dot
            #static = 1e6 * np.cross(sat_mag_sen, angular_velocity - np.array([0, 0, 0*np.deg2rad(10)]))
            static = 1e6 * np.cross(sat_mag_sen, angular_velocity)
            follow = 1e5 * (sat_mag_sen-sat_mag_sen_b) # 1e6
            sat_mag_sen_b = sat_mag_sen
            
            MTQ_vector = 2*np.array([adcs.bang_bang_bdot(follow[0], 0.1), adcs.bang_bang_bdot(follow[1], 0.1), adcs.bang_bang_bdot(follow[2], 0.1)])
            #"""

            """Quaternion Feedback
            # !Quaternion
            x_axis, y_axis, z_axis = adcs.generate_axis_vector_LVLH(eci_pos, eci_vel)
            dcm_eci2lvlh = np.array([x_axis, y_axis, z_axis]).T
            Quaternion_ref = rotation.dcm2quaternion(dcm_eci2lvlh)
            if np.dot(q_back, Quaternion_ref) < 0 :
                Quaternion_ref = - Quaternion_ref
            q_back = Quaternion_ref
            Quaternion_err = rotation.Quaternion_product(rotation.conj_quat(Quaternion_ref), (attitude_quaternion))
            qe_vec = np.sign(Quaternion_err[3]) * Quaternion_err[0:3]
            # !角速度
            gyro_ref = -np.abs(rad_sec) * y_axis # Eq(16-17)_DOI: 10.2322/tastj.16.441
            gyro_err = gyro_ref - angular_velocity
            J = inertia_inv
            kp = 1.0e-6
            kd = 1.0e-4
            tp = kp * qe_vec
            td = kd * gyro_err
            ff = attitude.skew(gyro_ref) @ inertia @ gyro_ref
            t = tp + td #+ gra_trq#+ ff
            #t_filterd = adcs.lowpass_filter(t_filterd, t, alpha)
            t_filterd = t
            # 普通のクロスプロダクト
            MTQ_vector = np.cross(sat_mag_sen, t) / (np.linalg.norm(sat_mag_sen)**2)
            # Eq(44)_DOI: 10.2322/tastj.16.441
            #MTQ_vecror = attitude.skew(sat_mag_sen) @ D_inv @ t / (np.linalg.norm(sat_mag_sen)**2)
            #Quaternion Feedback"""

            # *磁気トルカの出力サチュレーション表現"""
            #MTQ_vector = adcs.discretize_and_limit_moment(MTQ_vector, 0.2, 1) # 理想の入力, 飽和値, 分割数
            print(follow, end =',')
            print(MTQ_vector , end =',')
            print(err_vec[2], end=' ')
            print(np.linalg.norm(angular_velocity), end=' ')

            # !制御パラメータ(次回off)
            MTQ_con = False
        else:
            # 計測期間コンフィギュレーション設定
            MTQ_vector = np.array([0,0,0])
            sat_mag_sen = earth_mag
            # !制御パラメータ(次回on)
            MTQ_con = True
    #
    #print(attitude_quaternion)
    angular_velocity, attitude_quaternion = attitude.runge_kutta_quaternion(torque, angular_velocity, attitude_quaternion, inertia, inertia_inv, dt)

#discord_bot.send_message(f"Step {total_step}/{total_step} (100%)完了")
# CSVファイルにデータを書き込む
# ファイル/フォルダ名等がコンフリクトしないように調整
now = datetime.datetime.now()
output_dir = '.\\output\\log_' + now.strftime('%Y%m%d_%H%M%S')
os.makedirs(output_dir, exist_ok=True)
f_name = output_dir + '\\sim_data' + now.strftime('%Y%m%d_%H%M%S') + '.csv'
# データ書き込み部
# 最後に一括 flatten（array_key_names を渡す）
flat_log = [output_if.flatten_record_named(r, ARRAY_KEY_NAMES) for r in raw_log]
# pandas で保存
df = pd.DataFrame(flat_log)
df.to_csv(f_name, index=False)
#discord_bot.send_message('CSVファイルに保存しました\n結果を確認してください！')
print('CSVファイルに保存しました\n結果を確認してください！')
