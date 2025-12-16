import numpy as np
import Mechanics as mech
from scipy.integrate import quad
from scipy.optimize import brentq
import Visualize as viz
import Dynamics as Dy


#Process 1 - 统一使用米作为长度单位
L1 = 0.30 # 30cm = 0.30m
L2 = 0.40 # 40cm = 0.40m
theta_1 = 80
theta_2 = 30
distance = 0.45 # 45cm = 0.45m

force = 5 #N
m = 2 #kg
J_1 = 0.02 # 抽屉的转动惯量 (kg·m²) - 修正为合理数值
act_dis = 0.10 # 10cm = 0.10m
Lambda = L2/2 - act_dis
Wall_gap = (distance - L2) / 2

# 修正初始条件 - 抽屉应该从原点附近开始
x_mass_ini = 0.0  # 抽屉中心x位置从原点开始
y_mass_ini = 0.0  # 抽屉中心y位置从原点开始
theta_0 = 0.0
vx_ini = 0.0
vy_ini = 0.0
O_0 = 0.0


params = Dy.DrawerParams(
    length=L2,
    width=L1,
    handle_width=Lambda,
    mass=m,
    Izz=J_1,
    wall_gap= Wall_gap
)

drawer_system = Dy.DrawerDynamics2D(params=params)


initial_conditions = (x_mass_ini , y_mass_ini , theta_0, vx_ini , vy_ini, O_0)
# 缩短仿真时间，减少累积误差
simulation_result = drawer_system.simulate(t_span=(0,10), initial_conditions=initial_conditions)


print('算完啦！')

# 1. 获取时间轴 (t_eval 的值)
time_points = simulation_result.t
    
# 2. 获取状态向量的历史记录
# simulation_result.y 是一个 6xN 的数组，N是时间点的数量
# 每一列代表一个时间点的状态 [x, y, theta, x_dot, y_dot, theta_dot]
state_history = simulation_result.y
    
# 3. 更方便地提取每个自由度的历史数据
x_history = state_history[0, :]
y_pos_history = state_history[1, :]
theta_z_history = state_history[2, :]
    
x_dot_history = state_history[3, :]
y_dot_history = state_history[4, :]
theta_z_dot_history = state_history[5, :]

# --- 使用这些信息 ---
    
# 打印一些结果来验证
print("仿真成功完成！")
print(f"共计算了 {len(time_points)} 个时间点。")
    
# 打印最后一个时间点的状态
print("x位置历史数据:")
#print(x_history)
print("y位置历史数据:")
#print(y_pos_history)
print('角度历史数据')
print(theta_z_history)

print("\n最后一个时间点的状态:")
print(f"  时间: {time_points[-1]:.2f} s")
print(f"  x 位置: {x_history[-1]:.4f} cm")
print(f"  y 位置: {y_pos_history[-1]:.4f} cm")
print(f"  theta_z 角度: {np.rad2deg(theta_z_history[-1]):.4f} 度")




#viz.Visualize.animate_Process_total(L1, L2, x_masspoint = x_history, y_masspoint = y_pos_history, theta_arr = theta_z_history, x_history = x_history, time_list= time_points, handle_width = Lambda, deg = 30)
viz.animate_Process_total_with_heatmap(
    L1, 
    L2, 
    x_masspoint = x_history,          # 将你的 x_history 数据赋给 x_masspoint 参数
    y_masspoint = y_pos_history,      # 将你的 y_pos_history 数据赋给 y_masspoint 参数
    theta_arr = theta_z_history,      # 将你的 theta_z_history 数据赋给 theta_arr 参数
    time_list = time_points, 
    handle_width = Lambda, 
    deg = 30
    # 注意：函数还有默认参数 grid_x=10, grid_y=5，你可以根据需要覆盖它们
)

