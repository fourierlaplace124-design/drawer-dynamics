import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

# --- 1. 创建画布和坐标系 ---
fig, ax = plt.subplots(figsize=(8, 4))

# 设置坐标系范围，并保持 x/y 轴的比例为1:1，这样矩形不会变形
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 5)
ax.set_aspect('equal')

# 隐藏坐标轴刻度，让画面更干净
ax.axis('off')

# --- 2. 定义抽屉和柜子的尺寸和样式 ---

# 柜子参数
cabinet_width = 8.0
cabinet_height = 4.0
cabinet_pos = (0, 0) # 左下角坐标

# 抽屉参数
drawer_width = cabinet_width * 0.95 # 抽屉比柜子窄一点
drawer_height = cabinet_height * 0.8 # 抽屉比柜子开口矮一点
drawer_initial_pos_x = cabinet_pos[0] + (cabinet_width - drawer_width) / 2
drawer_pos_y = cabinet_pos[1] + (cabinet_height - drawer_height) / 2

# 把手参数
handle_width = drawer_width * 0.2
handle_height = drawer_height * 0.1
handle_pos_y = drawer_pos_y + (drawer_height - handle_height) / 2

# --- 3. 创建图形对象 (Patches) ---

# 创建柜子 (静态)
# facecolor是填充色，edgecolor是边框色，lw是线宽
cabinet = patches.Rectangle(
    cabinet_pos, cabinet_width, cabinet_height, 
    facecolor='#8B4513', edgecolor='black', lw=2
)
# 创建一个柜子的开口，用深色模拟内部
cabinet_opening = patches.Rectangle(
    (drawer_initial_pos_x, drawer_pos_y), drawer_width, drawer_height,
    facecolor='#50280b'
)

# 创建抽屉 (动态)
drawer = patches.Rectangle(
    (drawer_initial_pos_x, drawer_pos_y), drawer_width, drawer_height,
    facecolor='#D2B48C', edgecolor='black', lw=1.5,
    label='Drawer' # 添加标签
)

# 创建把手 (动态)
handle_initial_pos_x = drawer_initial_pos_x + drawer_width * 0.75
handle = patches.Rectangle(
    (handle_initial_pos_x, handle_pos_y), handle_width, handle_height,
    facecolor='#C0C0C0', edgecolor='black', lw=1
)

# --- 4. 定义动画函数 ---

# 动画参数
total_frames = 100  # 动画总帧数
pull_distance = 5.0 # 抽屉拉出的最大距离

# `init` 函数：初始化动画的第一帧
def init():
    # 将所有图形对象添加到坐标系中
    ax.add_patch(cabinet)
    ax.add_patch(cabinet_opening)
    ax.add_patch(drawer)
    ax.add_patch(handle)
    # 返回需要更新的动态对象
    return drawer, handle

# `update` 函数：计算并更新每一帧的画面
# i 是当前帧的编号，从 0 到 total_frames-1
def update(i):
    # 计算当前帧抽屉应该拉出的距离 (线性插值)
    # 模拟先快后慢的拉出效果，可以使用sin函数
    progress = i / total_frames
    current_pull = pull_distance * progress # 线性拉出
    # current_pull = pull_distance * np.sin(progress * np.pi / 2) # 非线性拉出（需要 import numpy as np）

    # 更新抽屉的位置
    new_drawer_x = drawer_initial_pos_x + current_pull
    drawer.set_x(new_drawer_x)

    # 更新把手的位置 (把手要跟着抽屉一起动)
    new_handle_x = handle_initial_pos_x + current_pull
    handle.set_x(new_handle_x)

    # 返回被更新的动态对象
    return drawer, handle

# --- 5. 创建并启动动画 ---

# 创建动画对象
# fig: 动画所在的画布
# update: 每一帧的更新函数
# frames: 总帧数
# init_func: 初始化函数
# interval: 每帧之间的间隔（毫秒），20ms 大约等于 50 FPS
# blit: 优化选项，设为True时，动画只会重绘发生变化的部分，速度更快
ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=total_frames,
    init_func=init, 
    interval=20, 
    blit=True
)

# 显示动画
plt.show()

# 如果你想将动画保存为 gif (需要安装 imagemagick 或 ffmpeg)
# print("Saving animation to drawer_pull.gif...")
# ani.save('drawer_pull.gif', writer='imagemagick', fps=30)
# print("Done.")

