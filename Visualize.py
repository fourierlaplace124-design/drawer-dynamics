import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

class Visualize:
    def plot_Process_1(L1,L2,distance,force,m,J_1,act_dis, time_T):
        #Pre Calculation
        a_x = force / m
        L = np.sqrt(L1**2 + L2**2)
        print("Length of angulus L:", L)
        theta_0 = np.degrees(np.arctan(L2/L1))
        Lambda = L2/2 - act_dis
        def d_Omega_d_t(T):
            return ( 1/2 * Lambda * L2 * force ) / J_1
        
        def theta(T):
        # 先对 d_Omega_d_t 积分得到 Omega
            def omega_func(t):
                return quad(d_Omega_d_t, 0, t)[0]
            # 再对 Omega 积分得到角位移
            theta_val, _ = quad(omega_func, 0, T)
            return theta_val    

        t= np.linspace(0,time_T*2,100)
        b1 = 40
        b2 = b1 + distance

        theta_values = [theta_0 - theta(tp) for tp in t]
 
        # 将结果从 Python 列表转换为 NumPy 数组
        theta = np.array(theta_values) 

        x_masspoint = 0.5 * a_x * t**2
        y_masspoint_value = b1 + distance / 2
        y_masspoint = np.full(x_masspoint.shape, y_masspoint_value)

        x_angulus = x_masspoint + 0.5*L* np.sin(np.radians(theta))
        y_angulus = y_masspoint + 0.5*L* np.cos(np.radians(theta))

        #Start ploting
        plt.figure(figsize=(8, 6))

        #plot the masspoint
        plt.plot(x_masspoint, y_masspoint, color='red', linestyle='-', label='mass point')

        #plot the angulus
        plt.plot(x_angulus, y_angulus, color='blue', linestyle='-', label='angulus')

        plt.title("the trace of process 1") 
        plt.xlabel("X-axis")                    
        plt.ylabel("Y-axis")                    
        plt.grid(True)                         
        plt.legend()                            
        
        # 6. 显示图像
        plt.show()