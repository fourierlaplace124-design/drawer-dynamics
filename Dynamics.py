import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 设置后端以支持动画
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
from dataclasses import dataclass
from typing import Tuple, List
from dataclasses import dataclass, field

# 配置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

@dataclass
class DrawerParams:
    """抽屉系统参数"""
    
    # --- 把所有需要外部传入的参数，作为类的属性在这里定义 ---
    # 几何参数
    length: float          # 抽屉长度 (m), L2
    width: float           # 抽屉宽度 (m), L1
    handle_width: float     # 把手到中心线的距离 (m)
    wall_gap: float # 抽屉边缘到墙的距离

    
    # 物理参数
    mass: float            # 抽屉质量 (kg)
    Izz: float             # 绕z轴转动惯量 (kg·m²)
 
    # --- 对于有默认值的参数，也在这里定义 ---
    height: float = 0.2
    contact_stiffness: float = 1e3   # 进一步降低接触刚度
    friction_constant: float = 0.1   # 降低摩擦系数
    stiffness_x: float = 5000.0        # 增加刚度以提供回复力
    stiffness_y: float = 5000.0        # 增加刚度以提供回复力
    stiffness_theta: float = 1000.0     # 增加转动刚度
 
    # --- 对于需要计算得出的属性，先声明它，然后在 __post_init__ 中计算 ---
    # field(init=False) 告诉 dataclass，这个属性不需要在 __init__ 中传入
    width_between_handles_and_wall: float = field(init=False)
 
    def __post_init__(self):
        """
        在对象初始化完成后，自动调用此方法来计算衍生属性。
        """
        # 注意：这里用 self 来访问已经初始化好的属性
        # distance / 2 是把手中心到抽屉中心线的距离
        # width / 2 是把手安装面到抽屉中心线的距离

        self.width_between_handles_and_wall = (self.length + 2 * self.wall_gap) / 2 - self.handle_width

class DrawerDynamics2D:
    """抽屉动力学系统"""
    
    def __init__(self, params: DrawerParams = None):
        """
        初始化抽屉动力学系统
        """
        self.params = params if params is not None else DrawerParams()
        self.wall_gap = self.params.wall_gap
        # 修正墙壁位置计算：墙壁应该在抽屉两侧，距离抽屉边缘wall_gap的距离
        half_drawer_width = self.params.width / 2
        self.wall_positions = [-half_drawer_width - self.wall_gap, half_drawer_width + self.wall_gap]  # 左右墙壁位置
        
        # 构建刚度矩阵
        self.K_2d = np.diag([
            self.params.stiffness_x,
            self.params.stiffness_y,
            self.params.stiffness_theta
        ])
        
        # 仿真结果存储
        self.solution = None
        self.time_points = None
    
    def applied_force(self, t: float, deg: float = 0) -> float:
        """
        施加的外力（x方向）
        持续向正x方向拉抽屉
        外力方向遵循角度制，当拉力水平时，deg = 0
        """
        # 持续的拉力，确保抽屉向正x方向运动
        deg_rad = np.radians(deg)
        return [0.5 * np.cos(deg_rad), 0.5 * np.sin(deg_rad)]  # 牛顿，减小力度避免过于激烈
    
    def system_equations_2d(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        抽屉动力学（x, y, θz 三自由度）
        状态变量: y = [x, y, θz, x_dot, y_dot, θz_dot]
        x: 抽屉抽出距离（只能为正值）
        y: 抽屉左右偏移（受墙壁限制）
        θz: 抽屉旋转角度
        """
        L2 = self.params.length  # 抽屉长度（x方向）
        L1 = self.params.width   # 抽屉宽度（y方向）
        distance = 2 * self.wall_gap + L1  # 修正：应该是宽度L1，不是长度L2

        n = len(y) // 2  # n=3
        q = y[:n]        # [x, y, θz] 位置和转角
        q_dot = y[n:]    # [x_dot, y_dot, θz_dot] 速度和角速度
        
        # 提取状态变量
        x, y_pos, theta_z = q
        x_dot, y_dot, theta_z_dot = q_dot
        
        # 限制抽屉只能向正x方向抽出
        if x < 0:
            x = 0
            x_dot = max(0, x_dot)  # 只允许正向速度
        
        # 应用力（只在正x方向）
        F_applied = self.applied_force(t , 0) if x >= 0 else 0
        
        # 检查四个角是否与墙壁接触
        corner_contacts = self.check_wall_contact(x, y_pos, theta_z)
        
        # 计算每个角的约束力
        F_constraint = self.compute_corner_constraint_forces(
            q, q_dot, corner_contacts, F_applied
        )
        
        # 建立外力向量 [Fx, Fy, Mz]
        F_external = np.zeros(3)
        F_external[0] = F_applied[0] + F_constraint[0]  # x方向力
        F_external[1] = F_applied[1] + F_constraint[1]              # y方向力
        F_external[2] = self.params.handle_width * F_applied[0] + L2 / 2 * F_applied[1] + F_constraint[2]  # z轴力矩

        # 质量矩阵 (3x3)
        M = np.diag([self.params.mass, self.params.mass, self.params.Izz])
        
        sin_alpha_1 = distance / np.sqrt(L1*L1 + L2*L2)
        alpha_1 = np.arcsin(sin_alpha_1)
        cos_alpha_1 = np.cos(alpha_1)

        alpha_2 = np.arctan(L2/L1)
        sin_alpha_2 = np.sin(alpha_2)
        cos_alpha_2 = np.cos(alpha_2)

        theta_stuck_rad = alpha_1 - alpha_2
        theta_stuck = np.degrees(theta_stuck_rad)

        # 修改弹性力逻辑：只对y和theta施加回复力，x方向不施加回复力
        elastic_force = np.zeros(3)
        if abs(theta_z) > abs(theta_stuck_rad):
            # 只对y方向和旋转角度施加回复力
            elastic_force[1] = -self.params.stiffness_y * y_pos  # y方向回复力
            elastic_force[2] = -self.params.stiffness_theta * theta_z  # 旋转回复力
            # x方向不施加回复力，让抽屉能够被拉出
        
        # 求解加速度
        total_force = F_external + elastic_force
        q_ddot = np.linalg.solve(M, total_force)
        
        # 限制x方向运动：不能向负方向移动
        if x <= 0 and q_ddot[0] < 0:
            q_ddot[0] = 0
        if x <= 0 and x_dot < 0:
            q_dot[0] = 0  # 修正速度状态
        
        return np.concatenate([q_dot, q_ddot])

    def check_wall_contact(self, x, y, theta_z):
        """
        检查抽屉四个角是否分别与左右墙壁接触
        
        Returns:
            corner_contacts: 字典，包含每个角的接触状态
            {
                'left_back': bool,    # 左后角是否接触墙壁
                'right_back': bool,   # 右后角是否接触墙壁
                'right_front': bool,  # 右前角是否接触墙壁
                'left_front': bool    # 左前角是否接触墙壁
            }
        """
        # 获取抽屉四个角的位置（考虑旋转）
        corners = self.get_drawer_corners(x, y, theta_z)
        
        # corners顺序: [左后角, 右后角, 右前角, 左前角]
        corner_names = ['left_back', 'right_back', 'right_front', 'left_front']
        corner_contacts = {}
        
        for i, (corner, name) in enumerate(zip(corners, corner_names)):
            corner_x, corner_y = corner
            
            # 检查该角是否接触左墙或右墙
            contact_left_wall = corner_y <= self.wall_positions[0]  # y坐标小于等于左墙位置
            contact_right_wall = corner_y >= self.wall_positions[1]  # y坐标大于等于右墙位置
            
            # 该角接触任一墙壁即为True
            corner_contacts[name] = contact_left_wall or contact_right_wall
        
        return corner_contacts

    def compute_corner_constraint_forces(self, q, q_dot, corner_contacts, F_applied):
        """
        计算每个角的约束力（墙壁法向力和摩擦力）
        
        Args:
            q: 位置状态 [x, y, theta_z]
            q_dot: 速度状态 [x_dot, y_dot, theta_z_dot]
            corner_contacts: 四个角的接触状态字典
            F_applied: 施加的外力
        """
        x, y, theta_z = q
        x_dot, y_dot, theta_z_dot = q_dot
        
        F_constraint = np.zeros(3)  # [Fx, Fy, Mz]
        
        # 获取四个角的位置
        corners = self.get_drawer_corners(x, y, theta_z)
        corner_names = ['left_back', 'right_back', 'right_front', 'left_front']
        
        # 为每个角计算支持力
        for i, (corner, name) in enumerate(zip(corners, corner_names)):
            if corner_contacts[name]:
                corner_x, corner_y = corner
                
                # 计算该角的穿透深度
                penetration_left = max(0, self.wall_positions[0] - corner_y) if corner_y <= self.wall_positions[0] else 0
                penetration_right = max(0, corner_y - self.wall_positions[1]) if corner_y >= self.wall_positions[1] else 0
                
                # 法向力（弹簧模型）
                N_left = self.params.contact_stiffness * penetration_left if penetration_left > 0 else 0
                N_right = self.params.contact_stiffness * penetration_right if penetration_right > 0 else 0
                
                # 计算该角的速度（考虑旋转）
                # 角点相对于质心的位置向量
                r_x = corner_x - x
                r_y = corner_y - y
                
                # 角点的速度 = 质心速度 + 角速度 × 位置向量
                v_corner_x = x_dot - theta_z_dot * r_y
                v_corner_y = y_dot + theta_z_dot * r_x
                
                # 该角的摩擦力 = 摩擦系数 × 正压力
                # 正压力就是该角的法向力
                normal_force = N_left + N_right
                
                # 摩擦力方向与速度方向相反
                friction_direction_x = -np.sign(v_corner_x) if abs(v_corner_x) > 1e-6 else 0
                friction_direction_y = -np.sign(v_corner_y) if abs(v_corner_y) > 1e-6 else 0
                
                # 该角的摩擦力 = 摩擦系数 × 正压力
                f_friction_x = self.params.friction_constant * normal_force * friction_direction_x
                f_friction_y = self.params.friction_constant * normal_force * friction_direction_y
                
                # 累加到总约束力
                # y方向法向力：左墙向右(+)，右墙向左(-)
                F_constraint[1] += N_left - N_right
                
                # x方向摩擦力
                F_constraint[0] += f_friction_x
                
                # 计算力矩：F × r
                # 法向力产生的力矩
                M_normal = (N_left - N_right) * r_x
                # 摩擦力产生的力矩
                M_friction = f_friction_x * r_y - f_friction_y * r_x
                
                F_constraint[2] += M_normal + M_friction
        
        return F_constraint
    
    def get_drawer_corners(self, x, y, theta_z):
        """
        获取抽屉四个角的位置（考虑旋转）
        
        Args:
            x, y: 抽屉中心位置
            theta_z: 绕z轴旋转角度
            
        Returns:
            corners: 四个角的坐标列表 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        # 抽屉在局部坐标系中的四个角
        half_length = self.params.length / 2
        half_width = self.params.width / 2
        
        local_corners = np.array([
            [-half_length, -half_width],  # 左后角
            [half_length, -half_width],   # 右后角
            [half_length, half_width],    # 右前角
            [-half_length, half_width]    # 左前角
        ])
        
        # 旋转矩阵
        cos_theta = np.cos(theta_z)
        sin_theta = np.sin(theta_z)
        R = np.array([[cos_theta, -sin_theta],
                      [sin_theta, cos_theta]])
        
        # 转换到全局坐标系
        global_corners = []
        for corner in local_corners:
            rotated_corner = R @ corner
            global_corner = [x + rotated_corner[0], y + rotated_corner[1]]
            global_corners.append(global_corner)
        
        return global_corners
    
    def compute_penetration(self, x, y, theta_z):
        """
        计算与墙壁的穿透深度
        
        Args:
            x, y: 抽屉中心位置
            theta_z: 旋转角度
            
        Returns:
            penetration_A, penetration_B: 左右墙壁的穿透深度
        """
        corners = self.get_drawer_corners(x, y, theta_z)
        
        # 计算左墙穿透深度
        penetration_A = 0
        for corner in corners:
            if corner[1] < self.wall_positions[0]:
                penetration_A = max(penetration_A, self.wall_positions[0] - corner[1])
        
        # 计算右墙穿透深度
        penetration_B = 0
        for corner in corners:
            if corner[1] > self.wall_positions[1]:
                penetration_B = max(penetration_B, corner[1] - self.wall_positions[1])
        
        return penetration_A, penetration_B
    
    
    
    def simulate(self, t_span=(0, 10), initial_conditions=None, dt=0.01):
        """
        运行仿真
        
        Args:
            t_span: 时间范围 (start, end)
            initial_conditions: 初始条件 [x0, y0, theta0, vx0, vy0, omega0]
            要注意，这里的x,y是抽屉中心的位置
            dt: 时间步长
            
        Returns:
            solution: 仿真结果
        """
        if initial_conditions is None:
            # 默认初始条件：抽屉在原点，给一个小的正x方向初始速度
            initial_conditions = [0, 0, 0, 0.1, 0, 0]  # 给x方向一个小的初始速度
        
        # 时间点
        self.time_points = np.arange(t_span[0], t_span[1], dt)
        
        # 求解微分方程
        self.solution = solve_ivp(
            self.system_equations_2d,
            t_span,
            initial_conditions,
            t_eval=self.time_points,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        return self.solution