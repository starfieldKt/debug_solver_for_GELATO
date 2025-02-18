import math
import numpy as np
import iric
import sys
import time  # 追加
from datetime import timedelta  # 追加

# 計算格子のサイズ
node_in = np.dtype([
    ("coordinate_x", "f8"),
    ("coordinate_y", "f8"),
    ("elevation", "f8"),
])

node_sol = np.dtype([
    ("depth", "f8"),
    ("water_level", "f8"),
    ("velocity_x", "f8"),
    ("velocity_y", "f8"),
])

start_time = time.time()  # 計算開始時間を記録
print("----------Start----------")

###############################################################################
# CGNSを開く
###############################################################################

# iRICで動かす時用
# =============================================================================
if len(sys.argv) < 2:
    print("Error: CGNS file name not specified.")
    exit()

cgns_name = sys.argv[1]

print("CGNS file name: " + cgns_name)

# CGNSをオープン
fid = iric.cg_iRIC_Open(cgns_name, iric.IRIC_MODE_MODIFY)

# コマンドラインで動かす時用
# =============================================================================

# CGNSをオープン
# fid = iric.cg_iRIC_Open("./project/Case1.cgn", iric.IRIC_MODE_MODIFY)

# 分割保存したい場合はこれを有効にする
# os.environ['IRIC_SEPARATE_OUTPUT'] = '1'

###############################################################################
# 古い計算結果を削除
###############################################################################

iric.cg_iRIC_Clear_Sol(fid)

###############################################################################
# 計算条件読み込み
###############################################################################

# 計算時間を読み込み
time_end = iric.cg_iRIC_Read_Integer(fid, "time_end")

# 流れの情報の配列の長さを読み込み
flow_info_arry_size = iric.cg_iRIC_Read_FunctionalSize(fid, "flow_info")
# 流れの情報の配列を読み込み
flow_info_time_arr = iric.cg_iRIC_Read_FunctionalWithName(fid, "flow_info", "time") 
water_level_arr = iric.cg_iRIC_Read_FunctionalWithName(fid, "flow_info", "water_level")
Velocity_xi_coefficient_arr = iric.cg_iRIC_Read_FunctionalWithName(fid, "flow_info", "Velocity_xi_coefficient")
Velocity_eta_coefficient_arr = iric.cg_iRIC_Read_FunctionalWithName(fid, "flow_info", "Velocity_eta_coefficient")

###############################################################################
# 格子の情報を読み込む
###############################################################################

# 格子サイズを読み込み
node_size_i, node_size_j = iric.cg_iRIC_Read_Grid2d_Str_Size(fid)
cell_size_i = node_size_i - 1
cell_size_j = node_size_j - 1
# 読み込んだ格子サイズをコンソールに出力
print("Grid size:")
print(f"    node_size_i= {node_size_i}")
print(f"    node_size_j= {node_size_j}")

node_in = np.zeros((node_size_i, node_size_j), dtype=node_in)
node_sol = np.zeros((node_size_i, node_size_j), dtype=node_sol)

# 格子の座標を読み込み
grid_x_arr_org, grid_y_arr_org = iric.cg_iRIC_Read_Grid2d_Coords(fid)
node_in["coordinate_x"] = grid_x_arr_org.reshape(node_size_j, node_size_i).T
node_in["coordinate_y"] = grid_y_arr_org.reshape(node_size_j, node_size_i).T

# 標高を読み込み
node_in["elevation"] = iric.cg_iRIC_Read_Grid_Real_Node(fid, "Elevation").reshape(node_size_j, node_size_i).T

print("----------mainloop start----------")

###############################################################################
# 一般座標系の流速を物理座標系の流速に変換するためのヤコビアン行列の要素を計算
###############################################################################
grid_interval_xi = 1/(cell_size_i)
grid_interval_eta = 1/(cell_size_j)

# ヤコビアン行列の要素を計算
dx_dxi = np.gradient(node_in["coordinate_x"], grid_interval_xi, axis=0, edge_order=1)
dx_deta = np.gradient(node_in["coordinate_x"], grid_interval_eta, axis=1, edge_order=1)
dy_dxi = np.gradient(node_in["coordinate_y"], grid_interval_xi, axis=0, edge_order=1)
dy_deta = np.gradient(node_in["coordinate_y"], grid_interval_eta, axis=1, edge_order=1)

# 逆ヤコビアン行列の要素を計算
# det_jacobian = dx_dxi * dy_deta - dx_deta * dy_dxi
# dxi_dx = dy_deta / det_jacobian
# dxi_dy = -dx_deta / det_jacobian
# deta_dx = -dy_dxi / det_jacobian
# deta_dy = dx_dxi / det_jacobian

###############################################################################
# 一般座標系の流速をセット
###############################################################################

velocity_xi = grid_interval_xi
velocity_eta = grid_interval_eta

# node_size_jが奇数でjが中央値の場合、dx_detaの中央値は0になる
# また、jが半分より小さい時は正、大きい時は負になる
node_index_j_center = node_size_j // 2

# velocity_eta_sol の符号を j 中心で反転
sign_eta = np.ones(node_size_j)

if node_size_j % 2 == 1:  # 奇数のとき
    sign_eta[node_index_j_center:] = -1  # 中心以降を反転
    sign_eta[node_index_j_center] = 0  # ちょうど中心は 0 にする
else:  # 偶数のとき
    sign_eta[node_index_j_center:] = -1  # 完全な反転 (0 は作らない)

###############################################################################
# メインループスタート
###############################################################################
for t in range(time_end + 1):

    ###########################################################################
    # この時間における流れの情報を計算
    ###########################################################################

    # 水位を計算
    water_level = np.interp(t, flow_info_time_arr, water_level_arr)
    node_sol["water_level"] = water_level

    # 水深を計算、水位が標高よりも低い場合は水深は0
    node_sol["depth"] = np.maximum(water_level - node_in["elevation"], 0.0)

    # 流速係数を計算
    Velocity_xi_coefficient = np.interp(t, flow_info_time_arr, Velocity_xi_coefficient_arr)
    Velocity_eta_coefficient = np.interp(t, flow_info_time_arr, Velocity_eta_coefficient_arr)

    # 一般座標系の流速に係数をかける
    velocity_xi_sol = velocity_xi * Velocity_xi_coefficient
    velocity_eta_sol = velocity_eta * Velocity_eta_coefficient

    # 流速変換 (Numpyブロードキャスト)
    sign_eta_velocity_eta_sol = sign_eta * velocity_eta_sol
    node_sol["velocity_x"] = velocity_xi_sol * dx_dxi + sign_eta_velocity_eta_sol * dx_deta
    node_sol["velocity_y"] = velocity_xi_sol * dy_dxi + sign_eta_velocity_eta_sol * dy_deta

    ###########################################################################
    # 結果の書き込みスタート
    ###########################################################################

    # 時間ごとの書き込み開始をGUIに伝える
    iric.cg_iRIC_Write_Sol_Start(fid)

    # 時刻を書き込み
    iric.cg_iRIC_Write_Sol_Time(fid, float(t))

    # 標高を書き込み
    iric.cg_iRIC_Write_Sol_Node_Real(fid, "Elevation", node_in["elevation"].T.flatten())

    # 流速を書き込み
    iric.cg_iRIC_Write_Sol_Node_Real(fid, "velocityX", node_sol["velocity_x"].T.flatten())
    iric.cg_iRIC_Write_Sol_Node_Real(fid, "velocityY", node_sol["velocity_y"].T.flatten())

    # 水位を書き込み
    iric.cg_iRIC_Write_Sol_Node_Real(fid, "waterLevel", node_sol["water_level"].T.flatten())

    # 水深を書き込み
    iric.cg_iRIC_Write_Sol_Node_Real(fid, "depth", node_sol["depth"].T.flatten())

    # CGNSへの書き込み終了をGUIに伝える
    iric.cg_iRIC_Write_Sol_End(fid)

    # コンソールに時間を出力
    print("t= " + str(t))

    # 計算結果の再読み込みが要求されていれば出力を行う
    iric.cg_iRIC_Check_Update(fid)

    # 計算のキャンセルが押されていればループを抜け出して出力を終了する。
    canceled = iric.iRIC_Check_Cancel()
    if canceled == 1:
        print("Cancel button was pressed. Calculation is finishing. . .")
        break

end_time = time.time()  # 計算終了時間を記録

# 計算時間をhh:mm:ss形式で出力
elapsed_time = end_time - start_time
formatted_time = str(timedelta(seconds=elapsed_time))
print("----------Finish----------")
print(f"計算時間: {formatted_time}")  # 計算時間を出力

###############################################################################
# 計算終了処理
###############################################################################
iric.cg_iRIC_Close(fid)
