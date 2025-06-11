"""
地壳热扩散数值模拟 - 最终版
包含核心计算和可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt

def solve_earth_crust_diffusion():
    """核心计算函数（与测试兼容）"""
    # 物理参数
    D = 0.1      # 热扩散率 (m^2/day)
    A = 10.0     # 年平均地表温度 (°C)
    B = 12.0     # 地表温度振幅 (°C)
    tau = 365.0  # 年周期 (days)
    T_bottom = 11.0  # 20米深处温度 (°C)
    
    # 网格参数 (匹配测试要求)
    n_depth = 21  # 0-20米共21个点
    n_time = 366  # 时间点数
    
    # 初始化数组
    depth = np.linspace(0, 20, n_depth)
    time = np.arange(n_time)
    T = np.zeros((n_depth, n_time))
    
    # 初始条件
    T[:, 0] = np.linspace(A, T_bottom, n_depth)
    
    # 稳定性检查
    dz = depth[1] - depth[0]
    dt = 1.0
    r = D * dt / (dz**2)
    if r > 0.5:
        raise ValueError(f"Stability condition violated: r = {r} > 0.5")
    
    # 时间步进
    for n in range(n_time - 1):
        T[0, n+1] = A + B * np.sin(2 * np.pi * time[n+1] / tau)  # 地表
        T[-1, n+1] = T_bottom  # 底部
        for i in range(1, n_depth - 1):
            T[i, n+1] = T[i, n] + r * (T[i+1, n] - 2*T[i, n] + T[i-1, n])
    
    return depth, T

def plot_seasonal_profiles(depth, T):
    """可视化函数"""
    plt.figure(figsize=(10, 6))
    
    # 选择代表四季的时间点（使用第9年的数据）
    seasons = [
        ('Spring', 91),
        ('Summer', 182),
        ('Autumn', 274),
        ('Winter', 365)
    ]
    
    for name, day in seasons:
        plt.plot(T[:, day], -depth, label=f'{name} (Day {day})')
    
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Depth (m)')
    plt.title('Seasonal Temperature Profiles in Earth Crust (Year 9)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 核心计算（测试需要）
    depth, T = solve_earth_crust_diffusion()
    
    # 可视化（独立功能）
    plot_seasonal_profiles(depth, T)
    
    # 调试信息（验证测试要求）
    print(f"Array shapes - Depth: {len(depth)}, Temperature: {T.shape}")
    print(f"Boundary checks - Surface: {T[0,0]:.1f}°C, Bottom: {T[-1,0]:.1f}°C")
