"""
学生模板：铝棒热传导问题
文件：heat_diffusion_student.py
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 物理参数
K = 237       # 热导率 (W/m/K)
C = 900       # 比热容 (J/kg/K)
rho = 2700    # 密度 (kg/m^3)
D = K/(C*rho) # 热扩散系数
L = 1         # 铝棒长度 (m)
dx = 0.01     # 空间步长 (m)
dt = 0.5      # 时间步长 (s)
Nx = int(L/dx) + 1 # 空间格点数
Nt = 2000     # 时间步数

def basic_heat_diffusion():
    """
    任务1: 基本热传导模拟
    
    返回:
        np.ndarray: 温度分布数组
    """
    # 计算稳定性参数r
    r = D * dt / (dx**2)
    print(f"Stability parameter r = {r:.4f}")
    
    # 初始化温度数组
    u = np.zeros((Nx, Nt))
    
    # 设置初始条件
    u[1:-1, 0] = 100  # 内部点初始温度为100K
    
    # 显式有限差分法迭代
    for j in range(Nt-1):
        for i in range(1, Nx-1):
            u[i, j+1] = u[i, j] + r * (u[i+1, j] - 2*u[i, j] + u[i-1, j])
    
    return u

def analytical_solution(n_terms=100):
    """
    任务2: 解析解函数
    
    参数:
        n_terms (int): 傅里叶级数项数
    
    返回:
        np.ndarray: 解析解温度分布
    """
    T0 = 100  # 初始温度
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, dt*Nt, Nt)
    
    u = np.zeros((Nx, Nt))
    
    for n in range(1, n_terms+1, 2):  # 只取奇数项
        kn = n * np.pi / L
        for j, time in enumerate(t):
            u[:, j] += (4*T0/(n*np.pi)) * np.sin(kn*x) * np.exp(-kn**2 * D * time)
    
    return u

def stability_analysis():
    """
    任务3: 数值解稳定性分析
    """
    # 使用不稳定的时间步长
    unstable_dt = 0.6  # 使r>0.5
    r = D * unstable_dt / (dx**2)
    print(f"Unstable condition r = {r:.4f}")
    
    # 初始化温度数组
    u = np.zeros((Nx, Nt))
    u[1:-1, 0] = 100  # 初始条件
    
    # 显式有限差分法迭代
    for j in range(Nt-1):
        for i in range(1, Nx-1):
            u[i, j+1] = u[i, j] + r * (u[i+1, j] - 2*u[i, j] + u[i-1, j])
    
    # 绘制结果观察不稳定性
    plt.figure(figsize=(10, 6))
    for j in range(0, Nt, 400):
        plt.plot(np.linspace(0, L, Nx), u[:, j], label=f't={j*unstable_dt:.1f}s')
    plt.title('Numerical solution under unstable condition (r > 0.5)')
    plt.xlabel('Position (m)')
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.grid()
    plt.show()

def different_initial_condition():
    """
    任务4: 不同初始条件模拟
    
    返回:
        np.ndarray: 温度分布数组
    """
    # 使用更少的时间步数以加快计算
    Nt_diff = 1000
    
    # 计算稳定性参数r
    r = D * dt / (dx**2)
    
    # 初始化温度数组
    u = np.zeros((Nx, Nt_diff))
    
    # 设置不同的初始条件
    mid_point = Nx // 2
    u[1:mid_point, 0] = 100  # 左半部分100K
    u[mid_point:-1, 0] = 50   # 右半部分50K
    
    # 显式有限差分法迭代
    for j in range(Nt_diff-1):
        for i in range(1, Nx-1):
            u[i, j+1] = u[i, j] + r * (u[i+1, j] - 2*u[i, j] + u[i-1, j])
    
    return u

def heat_diffusion_with_cooling():
    """
    任务5: 包含牛顿冷却定律的热传导
    """
    h = 0.01  # 冷却系数 (s^-1)
    r = D * dt / (dx**2)
    
    # 初始化温度数组
    u = np.zeros((Nx, Nt))
    u[1:-1, 0] = 100  # 初始条件
    
    # 显式有限差分法迭代（包含冷却项）
    for j in range(Nt-1):
        for i in range(1, Nx-1):
            u[i, j+1] = (1 - 2*r - h*dt)*u[i, j] + r*(u[i+1, j] + u[i-1, j])
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    for j in range(0, Nt, 400):
        plt.plot(np.linspace(0, L, Nx), u[:, j], label=f't={j*dt:.1f}s')
    plt.title('Heat diffusion with Newton cooling (h=0.01 s^-1)')
    plt.xlabel('Position (m)')
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.grid()
    plt.show()

def plot_3d_solution(u, dx, dt, Nt, title):
    """
    绘制3D温度分布图
    
    参数:
        u (np.ndarray): 温度分布数组
        dx (float): 空间步长
        dt (float): 时间步长
        Nt (int): 时间步数
        title (str): 图表标题
    
    返回:
        None
    """
    x = np.arange(0, L + dx, dx)
    t = np.arange(0, Nt*dt, dt)
    X, T = np.meshgrid(x, t)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, T, u.T, cmap='viridis', rstride=50, cstride=5)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Time (s)')
    ax.set_zlabel('Temperature (K)')
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

if __name__ == "__main__":
    print("=== 铝棒热传导问题 ===")
    print(f"热扩散系数 D = {D:.4e} m^2/s")
    
    # 任务1: 基本热传导模拟
    print("\nRunning Task 1: Basic heat diffusion...")
    u_num = basic_heat_diffusion()
    plot_3d_solution(u_num, dx, dt, Nt, "Numerical Solution of Heat Diffusion")
    
    # 任务2: 解析解
    print("\nRunning Task 2: Analytical solution...")
    u_ana = analytical_solution(n_terms=100)
    plot_3d_solution(u_ana, dx, dt, Nt, "Analytical Solution of Heat Diffusion")
    
    # 任务3: 数值解稳定性分析
    print("\nRunning Task 3: Stability analysis...")
    stability_analysis()
    
    # 任务4: 不同初始条件模拟
    print("\nRunning Task 4: Different initial condition...")
    u_diff = different_initial_condition()
    plot_3d_solution(u_diff, dx, dt, 1000, "Heat Diffusion with Different Initial Conditions")
    
    # 任务5: 包含冷却效应的热传导
    print("\nRunning Task 5: Heat diffusion with cooling...")
    heat_diffusion_with_cooling()
