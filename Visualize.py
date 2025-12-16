import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

#动画：
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
#速度云图
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.transforms as transforms

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

    def animate_Process_1(L1, L2, distance, force, m, J_1, act_dis, time_T):
        # --- 1. 物理计算 ---
        a_x = force / m
    
        def d_Omega_d_t(T):
            Lambda = L2/2 - act_dis
            moment_of_force = (1/2 * Lambda * L2 * force)
            return moment_of_force / J_1
    
        def theta(T):
            def omega_func(t):
                return quad(d_Omega_d_t, 0, t)[0]
            theta_val, _ = quad(omega_func, 0, T)
            return theta_val
    
        t = np.linspace(0, time_T * 2, 120)
        b1 = 0
    
        # 【修改2】角度从 0 开始，然后随时间变化
        # 原来是: theta_values = [theta_0 - theta(tp) for tp in t]
        theta_values = [-theta(tp) for tp in t]
        theta_arr = np.array(theta_values)
    
        # 质心轨迹 (平移部分)
        x_masspoint = 0.5 * a_x * t**2
        y_masspoint_value = b1 + L2 / 2
        y_masspoint = np.full(x_masspoint.shape, y_masspoint_value)
    
        # --- 2. 初始化画布 ---
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.7)
    
        # --- 3. 创建图形对象 ---
    
        # 创建抽屉相对于其质心的局部坐标
        drawer_depth = L1
        drawer_width = L2
        local_coords = np.array([
            [-drawer_depth / 2, -drawer_width / 2],  # 左下
            [ drawer_depth / 2, -drawer_width / 2],  # 右下
            [ drawer_depth / 2,  drawer_width / 2],  # 右上 (我们将追踪这个点)
            [-drawer_depth / 2,  drawer_width / 2]   # 左上
        ])
    
        # 初始化抽屉 Polygon 对象
        drawer = patches.Polygon(
            local_coords,
            closed=True,
            facecolor="#16F166",
            edgecolor='black',
            lw=1.5
        )
    
        # 创建用于绘制顶点轨迹的 Line2D 对象
        corner_trajectory_line, = ax.plot([], [], 'm-', lw=2, label='Corner Trajectory')
        trajectory_points = []
    
        # 绘制质心轨迹的静态线
        ax.plot(x_masspoint, y_masspoint, 'b--', lw=2, label='Center of Mass Trajectory')
        
        # --- 4. 定义动画函数 ---
        
        def init():
            ax.add_patch(drawer)
            
            max_dim = max(L1, L2) * 1.5 # 稍微增大余量
            ax.set_xlim(np.min(x_masspoint) - max_dim, np.max(x_masspoint) + max_dim)
            ax.set_ylim(np.min(y_masspoint) - max_dim, np.max(y_masspoint) + max_dim)
            ax.legend(loc='upper left')
            
            return drawer, corner_trajectory_line
    
        def update(i):
            # 1. 获取当前质心位置
            mass_point = np.array([x_masspoint[i], y_masspoint[i]])
            
            # 2. 获取当前旋转角度
            current_theta_rad = np.radians(theta_arr[i])
            R = np.array([
                [np.cos(current_theta_rad), -np.sin(current_theta_rad)],
                [np.sin(current_theta_rad),  np.cos(current_theta_rad)]
            ])
    
            # 3. 计算抽屉新的世界坐标
            world_coords = (R @ local_coords.T).T + mass_point
            drawer.set_xy(world_coords)
    
            # 4. 计算并更新顶点轨迹
            tracked_corner_world_pos = world_coords[2]
            trajectory_points.append(tracked_corner_world_pos)
            
            traj_array = np.array(trajectory_points)
            corner_trajectory_line.set_data(traj_array[:, 0], traj_array[:, 1])
    
            return drawer, corner_trajectory_line
    
        # --- 5. 创建并启动动画 ---
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(t),
            init_func=init,
            interval=30,
            blit=True,
            repeat=False
        )
    
        plt.show()

    def animate_Process_total(L1, L2, x_masspoint, y_masspoint, theta_arr, x_history, time_list, handle_width, deg):
        # --- 1. 初始化画布 ---
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.7)
    
        t = time_list
    
        # --- 2. 创建图形对象的局部坐标 ---
        drawer_depth = L1
        drawer_width = L2
    
        # 抽屉的4个角点
        local_coords = np.array([
            [-drawer_depth / 2, -drawer_width / 2],  # 左下
            [ drawer_depth / 2, -drawer_width / 2],  # 右下
            [ drawer_depth / 2,  drawer_width / 2],  # 右上 (我们将追踪这个点)
            [-drawer_depth / 2,  drawer_width / 2],  # 左上
        ])
    
        # 定义箭头属性 (在抽屉的局部坐标系中)
        arrow_local_base = np.array([drawer_depth / 2, handle_width])
        arrow_local_direction = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
        arrow_length = drawer_width / 2
    
        # --- 3. 创建图形对象 ---
        # 初始化抽屉 Polygon 对象
        drawer = patches.Polygon(
            local_coords,
            closed=True,
            facecolor="#16F166",
            edgecolor='black',
            lw=1.5
        )
    
        # 创建用于绘制顶点轨迹的 Line2D 对象
        corner_trajectory_line, = ax.plot([], [], 'm-', lw=2, label='Corner Trajectory')
        trajectory_points = []
    
        # --- 修改: 绘制质心轨迹的静态线应该是 (x_masspoint, y_masspoint) ---
        ax.plot(x_masspoint, y_masspoint, 'b--', lw=2, label='Center of Mass Trajectory')
    
        # 创建 Quiver 对象来代表箭头
        force_arrow = ax.quiver(
            [], [], [], [],
            angles='xy', scale_units='xy', scale=1,
            color='red', width=0.005, label='Force/Direction'
        )
    
        # --- 4. 定义动画函数 ---
        def init():
            ax.add_patch(drawer)
            
            # 确保坐标轴范围足够大
            # xlim, ylim 应该基于 x_masspoint 和 y_masspoint
            all_x = np.array(x_masspoint)
            all_y = np.array(y_masspoint)
            max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min()]).max()
            center_x = (all_x.max()+all_x.min())*0.5
            center_y = (all_y.max()+all_y.min())*0.5
            ax.set_xlim(center_x-max_range*0.5-L1, center_x+max_range*0.5+L1)
            ax.set_ylim(center_y-max_range*0.5-L2, center_y+max_range*0.5+L2)
    
            ax.legend(loc='upper left')
            return drawer, corner_trajectory_line, force_arrow
    
        def update(i):
            # 1. 获取当前质心位置和旋转矩阵
            mass_point = np.array([x_masspoint[i], y_masspoint[i]])
            
            # 假设 theta_arr 是角度，需要转换为弧度
            current_theta_rad = np.radians(theta_arr[i])
            R = np.array([
                [np.cos(current_theta_rad), -np.sin(current_theta_rad)],
                [np.sin(current_theta_rad),  np.cos(current_theta_rad)]
            ])
    
            # 2. 更新抽屉世界坐标
            world_coords = (R @ local_coords.T).T + mass_point
            drawer.set_xy(world_coords)
    
            # 3. 更新顶点轨迹
            tracked_corner_world_pos = world_coords[2] # 追踪右上角
            trajectory_points.append(tracked_corner_world_pos)
            traj_array = np.array(trajectory_points)
            corner_trajectory_line.set_data(traj_array[:, 0], traj_array[:, 1])
    
            # 4. 更新箭头
            world_arrow_base = (R @ arrow_local_base) + mass_point
            world_arrow_vector = (R @ arrow_local_direction) * arrow_length
            
            # --- 这是关键的修复 ---
            force_arrow.set_offsets([world_arrow_base])
            force_arrow.set_UVC(world_arrow_vector[0], world_arrow_vector[1])
    
            return drawer, corner_trajectory_line, force_arrow
    
        # --- 5. 创建并启动动画 ---
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(t),
            init_func=init,
            interval=30,
            blit=False,
            repeat=False
        )
        plt.show()

def animate_Process_total_with_heatmap(
    L1, L2,
    x_masspoint, y_masspoint, theta_arr,
    time_list,
    handle_width, deg,
    grid_x=10, grid_y=5,
    plot_relative_speed=True,
    view_size_factor=2.0
):
    print("正在预处理数据用于生成速度云图...")
    # 安全性检查，防止时间列表为空
    if len(time_list) < 2:
        print("错误：时间列表数据不足，无法计算速度。")
        return
    dt = time_list[1] - time_list[0]
    
    v_com_x = np.gradient(x_masspoint, dt)
    v_com_y = np.gradient(y_masspoint, dt)
    v_com_arr = np.stack((v_com_x, v_com_y), axis=-1)
    
    theta_rad_arr = np.radians(theta_arr)
    omega_arr_rad = np.gradient(theta_rad_arr, dt)
 
    if np.max(np.abs(omega_arr_rad)) < 1e-6:
        print("\n--- 警告 ---")
        print("检测到模拟数据中几乎没有旋转 (角速度 ω ≈ 0)。")
        if plot_relative_speed:
            print("相对速度云图将显示为全零 (一种颜色)。这是正确的物理现象。")
        else:
            print("绝对速度云图将显示为均匀颜色。要看到速度梯度，需要让物体旋转。")
        print("--------------\n")
 
    # --- 在局部坐标系下创建网格单元 ---
    cell_width = L1 / grid_x
    cell_height = L2 / grid_y
    cell_patches = []
    cell_centers_local = []
    for i in range(grid_x):
        for j in range(grid_y):
            # 定义小块的局部坐标
            local_x = -L1/2 + i * cell_width
            local_y = -L2/2 + j * cell_height
            
            # 创建小块的多边形Patch
            cell_coords_local = np.array([
                [local_x, local_y],
                [local_x + cell_width, local_y],
                [local_x + cell_width, local_y + cell_height],
                [local_x, local_y + cell_height],
            ])
            cell_patches.append(patches.Polygon(cell_coords_local, closed=True))
            
            # --- 这是修复NameError的关键部分 ---
            # 计算并记录小块的中心点 (局部坐标)
            center_x = local_x + cell_width / 2
            center_y = local_y + cell_height / 2
            # ------------------------------------
            cell_centers_local.append(np.array([center_x, center_y]))
 
    # --- 预计算最大速度，用于稳定颜色条 ---
    max_speed_to_norm = 0
    # ... (这部分代码无需改动) ...
    for i in range(len(time_list)):
        current_omega = omega_arr_rad[i]
        current_v_com = v_com_arr[i]
        c, s = np.cos(theta_rad_arr[i]), np.sin(theta_rad_arr[i])
        R = np.array([[c, -s], [s, c]])
        for r_local in cell_centers_local:
            # 注意：这里的 r_local 就是刚刚在循环里计算的 cell_centers_local
            r_world = R @ r_local
            v_rot = np.array([-current_omega * r_world[1], current_omega * r_world[0]])
            
            if plot_relative_speed:
                speed = np.linalg.norm(v_rot)
            else:
                v_total = current_v_com + v_rot
                speed = np.linalg.norm(v_total)
 
            if speed > max_speed_to_norm:
                max_speed_to_norm = speed
    
    if max_speed_to_norm < 1e-9:
        max_speed_to_norm = 1.0
 
    print(f"预计算完成。用于颜色映射的最大速度为: {max_speed_to_norm:.4f} m/s")
 
    # --- 设置绘图和颜色映射 (这部分代码无需改动) ---
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    cmap = cm.get_cmap('viridis')
    norm = colors.Normalize(vmin=0, vmax=max_speed_to_norm)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar_label = 'Relative Speed (m/s)' if plot_relative_speed else 'Absolute Speed (m/s)'
    cbar = fig.colorbar(sm, ax=ax, label=cbar_label)
 
    # --- 创建绘图对象 (这部分代码无需改动) ---
    ax.plot(x_masspoint, y_masspoint, 'b--', lw=2, label='Center of Mass Trajectory')
    corner_trajectory_line, = ax.plot([], [], 'm-', lw=2, label='Corner Trajectory')
    trajectory_points = []
    
    drawer_outline = patches.Polygon(np.array([[-L1/2,-L2/2],[L1/2,-L2/2],[L1/2,L2/2],[-L1/2,L2/2]]), 
                                     closed=True, facecolor='none', edgecolor='black', lw=2)
 
    def init():
        ax.add_patch(drawer_outline)
        for patch in cell_patches:
            ax.add_patch(patch)
        ax.legend(loc='upper left')
        view_size = max(L1, L2) * view_size_factor
        ax.set_xlim(x_masspoint[0] - view_size / 2, x_masspoint[0] + view_size / 2)
        ax.set_ylim(y_masspoint[0] - view_size / 2, y_masspoint[0] + view_size / 2)
        return [drawer_outline] + cell_patches + [corner_trajectory_line]
 
    def update(i):
        mass_point = np.array([x_masspoint[i], y_masspoint[i]])
        current_theta_rad = theta_rad_arr[i]
        current_v_com = v_com_arr[i]
        current_omega = omega_arr_rad[i]
 
        transform = transforms.Affine2D().rotate(current_theta_rad).translate(mass_point[0], mass_point[1])
        full_transform = transform + ax.transData
 
        c, s = np.cos(current_theta_rad), np.sin(current_theta_rad)
        R = np.array([[c, -s], [s, c]])
 
        for patch, r_local in zip(cell_patches, cell_centers_local):
            r_world = R @ r_local
            v_rot = np.array([-current_omega * r_world[1], current_omega * r_world[0]])
            
            if plot_relative_speed:
                speed = np.linalg.norm(v_rot)
            else:
                v_total = current_v_com + v_rot
                speed = np.linalg.norm(v_total)
            
            patch.set_facecolor(cmap(norm(speed)))
            patch.set_transform(full_transform)
            
        drawer_outline.set_transform(full_transform)
        
        corner_local = np.array([L1/2, L2/2])
        corner_world = (R @ corner_local) + mass_point
        trajectory_points.append(corner_world)
        traj_array = np.array(trajectory_points)
        corner_trajectory_line.set_data(traj_array[:, 0], traj_array[:, 1])
 
        view_size = max(L1, L2) * view_size_factor
        ax.set_xlim(mass_point[0] - view_size / 2, mass_point[0] + view_size / 2)
        ax.set_ylim(mass_point[1] - view_size / 2, mass_point[1] + view_size / 2)
 
        return [drawer_outline] + cell_patches + [corner_trajectory_line]
 
    ani = animation.FuncAnimation(
        fig, update, frames=len(time_list), init_func=init, interval=30, blit=False, repeat=False
    )
    plt.show()

