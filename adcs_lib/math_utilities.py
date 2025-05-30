import numpy as np

# SR-Invers Tohoku-Univ.
def SRInv(matrix):
    MATRIX = np.array([[1,0,0],[0,1,0],[0,0,1]])
    MATRIX = np.transpose(matrix) @ matrix
    MATRIX = MATRIX + np.array([[1,0,0],[0,1,0],[0,0,1]])
    MATRIX = MATRIX @ matrix
    return MATRIX

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        # ゼロベクトルの場合はそのまま返すか、ゼロベクトルを定義する
        normalized_vector = vector  # または normalized_vector = np.zeros_like(vector)
    else:
        normalized_vector = vector / norm
    return normalized_vector

# 移動平均フィルタ
def moving_average_3d(current_vector, vector_array):
    # 現在のベクトルを配列に追加
    vector_array = np.vstack((vector_array, current_vector))
    # 配列のサイズが5を超えた場合、古いベクトルを削除
    if len(vector_array) > 20:
        vector_array = vector_array[1:]
    # 移動平均を計算
    ma_vector = np.mean(vector_array, axis=0)
    return ma_vector, vector_array