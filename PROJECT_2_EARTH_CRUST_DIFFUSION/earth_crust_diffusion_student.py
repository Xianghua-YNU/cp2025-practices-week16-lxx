"""
地壳热扩散数值模拟
文件：earth_crust_diffusion_student.py
实现显式差分法求解地壳热扩散问题
"""

import numpy as np
import matplotlib.pyplot as plt

def solve_earth_crust_diffusion():
    """
    实现显式差分法求解地壳热扩散问题
    
    返回:
        tuple: (depth_array, temperature_matrix)
        depth_array: 深度坐标数组 (m)
        temperature_matrix: 温度场矩阵 (°C)
    """
    # 物理参数
    D = 0.1  # 热扩散率 (m^2/day)
    A = 10.0  # 年平均地表温度 (°C)
    B = 12.0  # 地表温度振幅 (°C)
    tau = 365.0  # 年周期 (days)
    T_bottom = 11.0  # 20米深处温度 (°C)
    depth_max = 20.0  # 最大深度 (m)
    total_years = 10  # 模拟总年数
    days_per_year = 365  # 每年天数
    
    # 网格参数
    dz = 1.0  # 深度步长 (m)
    dt = 1.0  # 时间步长 (day)
    
    # 计算网格点数和时间步数
    n_depth = int(depth_max / dz) + 1
    n_time = total_years * days_per_year + 1
    
    # 初始化数组
    depth = np.linspace(0, depth_max, n_depth)
    time = np.linspace(0, total_years * tau, n_time)
    T = np.zeros((n_depth, n_time))
    
    # 初始条件 (假设初始温度线性分布)
    T[:, 0] = np.linspace(A, T_bottom, n_depth)
    
    # 稳定性条件检查
    r = D * dt / (dz**2)
    if r > 0.5:
        raise ValueError(f"稳定性条件不满足: r = {r} > 0.5")
    
    # 时间步进
    for n in range(n_time - 1):
        # 边界条件
        T[0, n+1] = A + B * np.sin(2 * np.pi * time[n+1] / tau)  # 地表
        T[-1, n+1] = T_bottom  # 底部
        
        # 内部点更新 (显式格式)
        for i in range(1, n_depth - 1):
            T[i, n+1] = T[i, n] + r * (T[i+1, n] - 2*T[i, n] + T[i-1, n])
    
    return depth, T

if __name__ == "__main__":
    # 运行模拟
    depth, T = solve_earth_crust_diffusion()
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    
    # 选择第10年的四季时间点 (第9年结束后的第0.25, 0.5, 0.75, 1.0年)
    year = 9
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    season_times = [year * 365 + i * 91 for i in range(4)]
    
    for i, t in enumerate(season_times):
        plt.plot(T[:, t], -depth, label=f'{seasons[i]} (Day {t})')
    
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Depth (m)')
    plt.title('Seasonal Temperature Profiles in Earth Crust (Year 10)')
    plt.legend()
    plt.grid(True)
    plt.show()
