import numpy as np
import Mechanics as mech
from scipy.integrate import quad
from scipy.optimize import brentq
import Visualize as viz

#Process 1
L1 = 30 #cm
L2 = 40
theta_1 = 80
theta_2 = 30
distance = 45

force = 5 #N
m = 2 #kg
J_1 = 200 # 抽屉的转动惯量
act_dis = 10


# Calculate valid sliding distance
valid_dis = mech.Mechanics.Sliding_dist(L1,L2,theta_1,theta_2)
print("Valid sliding distance:", valid_dis, "cm")

# 我们现在假设力是竖直向下的，而且是恒定的力
Lambda = L2/2 - act_dis
def d_Omega_d_t(T):
    return ( 1/2 * Lambda * L2 * force ) / J_1
    
sin_alpha_1 = distance / np.sqrt(L1*L1 + L2*L2)
alpha_1 = np.arcsin(sin_alpha_1)
cos_alpha_1 = np.cos(alpha_1)

alpha_2 = np.arctan(L2/L1)
sin_alpha_2 = np.sin(alpha_2)
cos_alpha_2 = np.cos(alpha_2)

theta_stuck_rad = alpha_1 - alpha_2
theta_stuck = np.degrees(theta_stuck_rad)

print("Stuck angle:", theta_stuck)

#下面利用数值方法求解卡住所需要的时间：
def theta(T):
    # 先对 d_Omega_d_t 积分得到 Omega
    def omega_func(t):
        return quad(d_Omega_d_t, 0, t)[0]
    # 再对 Omega 积分得到角位移
    theta_val, _ = quad(omega_func, 0, T)
    return theta_val

def omega_func(t):
    return quad(d_Omega_d_t, 0, t)[0]

def F(T):
    return theta(T) - theta_stuck

a = 0.0
b  = 90.0

while F(b) < 0:
    b += 1 # 每次增加1单位，直到找到一个足够大的 b
        
print(f"搜索区间为 [{a}, {b}]")
print(f"f({a}) = {F(a):.2f}")
print(f"f({b}) = {F(b):.2f}")
        
    # 使用 brentq 求解，它会在 [a, b] 区间内精确找到 f(T)=0 的根 T
    # brentq 非常高效和稳定
try:
    time_T = brentq(F, a, b)
    print(f"\n转过 {theta_stuck} 度所需的时间 T 约为: {time_T:.4f} 秒")
   
    # ---- 验证结果 ----
    final_angle = theta((time_T))

    print(f"验证：在 {time_T:.4f} 秒内转过的角度为: {final_angle:.4f} 度，与目标值非常接近。")
        
except ValueError:
    print("\n在提供的区间内找不到根，或者区间端点函数值符号相同。请检查区间 [a, b]。")

O = omega_func(time_T)

viz.Visualize.plot_Process_1(L1,L2,distance,force,m,J_1,act_dis, time_T)
