"""
地壳热扩散数值模拟 
严格匹配测试要求
"""

import numpy as np

def solve_earth_crust_diffusion():
    """
    修正后的实现，严格匹配测试要求
    
    返回:
        tuple: (depth_array, temperature_matrix)
        depth_array: 0-20米共21个点
        temperature_matrix: 形状(21, 366)的温度场
    """
    # 物理参数
    D = 0.1      # 热扩散率 (m^2/day)
    A = 10.0     # 年平均地表温度 (°C)
    B = 12.0     # 地表温度振幅 (°C)
    tau = 365.0  # 年周期 (days)
    T_bottom = 11.0  # 20米深处温度 (°C)
    
    # 网格参数 (严格匹配测试要求)
    n_depth = 21  # 0-20米共21个点
    n_time = 366  # 时间点数
    
    # 初始化数组
    depth = np.linspace(0, 20, n_depth)
    time = np.arange(n_time)
    T = np.zeros((n_depth, n_time))
    
    # 初始条件 (线性分布)
    T[:, 0] = np.linspace(A, T_bottom, n_depth)
    
    # 稳定性条件
    dz = depth[1] - depth[0]
    dt = 1.0
    r = D * dt / (dz**2)
    if r > 0.5:
        raise ValueError(f"稳定性条件不满足: r = {r} > 0.5")
    
    # 时间步进
    for n in range(n_time - 1):
        # 边界条件
        T[0, n+1] = A + B * np.sin(2 * np.pi * time[n+1] / tau)
        T[-1, n+1] = T_bottom
        
        # 内部点更新
        for i in range(1, n_depth - 1):
            T[i, n+1] = T[i, n] + r * (T[i+1, n] - 2*T[i, n] + T[i-1, n])
    
    return depth, T

if __name__ == "__main__":
    depth, T = solve_earth_crust_diffusion()
    print(f"深度点数: {len(depth)}，时间步数: {T.shape[1]}")
    print(f"温度矩阵形状: {T.shape} (应返回(21, 366))")
