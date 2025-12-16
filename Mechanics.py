import numpy as np
from scipy.integrate import quad
from scipy.integrate import solve_ivp

class Mechanics: 
    def Sliding_dist(L1,L2,theta_1,theta_2):
        if theta_1 and theta_2 >= 0 and theta_1 and theta_2 <= 90:
            tan_1 = np.tan(np.radians(theta_1))
            tan_2 = np.tan(np.radians(theta_2))
            cot_1 = 1/tan_1
            d = L2 - L1*cot_1
            if d >= 0:
                real_d = d
            else:
                print("No sliding occurs")
                real_d = 0
        else:
            raise ValueError("Angles must be between 0 and 90 degrees")
        return real_d

class DrawerSimulation:
    def set_friction_parameters(self, mu_s, mu_k, alpha, beta):
        """设置非线性摩擦模型的参数"""
        self.mu_s = mu_s  #静摩擦系数
        self.mu_k = mu_k  #动摩擦系数
        self.alpha = alpha #tanh的陡峭系数
        self.beta = beta   #从静摩擦到动摩擦的过渡速度系数
        
        M_total = np.sum(np.diag(self.M_dense)) 
        self.normal_force = M_total * 9.81

    def system_equations(self, t, y):
        n = self.num_dofs
        q = y[:n]
        q_dot = y[n:]

        #计算非线性摩擦力
        v_x = q_dot[0]
        
        #Stribeck效果的简化模型
        friction_coeff = self.mu_k + (self.mu_s - self.mu_k) * np.exp(-self.beta * np.abs(v_x))
        
        #计算摩擦力
        friction_force_x = -self.normal_force * friction_coeff * np.tanh(self.alpha * v_x)
        
        #将摩擦力放入力向量
        F_friction = np.zeros(n)
        F_friction[0] = friction_force_x

        F_external = self.applied_force(t)
        elastic_force = -self.K @ q
        damping_force = -self.C @ q_dot

        total_force = F_external + elastic_force + damping_force + F_friction

        M_inv = np.linalg.inv(self.M)
        q_ddot = M_inv @ total_force

        return np.concatenate([q_dot, q_ddot])