"""量子隧穿效应模拟器
文件：quantum_tunneling_solution.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class QuantumTunnelingSolver:
    """量子隧穿求解器类
    
    该类实现了一维含时薛定谔方程的数值求解，用于模拟量子粒子的隧穿效应。
    使用变形的Crank-Nicolson方法进行时间演化，确保数值稳定性和概率守恒。
    """
    
    def __init__(self, Nx=220, Nt=300, x0=40, k0=0.5, d=10, barrier_width=3, barrier_height=1.0):
        """初始化量子隧穿求解器
        
        参数:
            Nx (int): 空间网格点数，默认220
            Nt (int): 时间步数，默认300
            x0 (float): 初始波包中心位置，默认40
            k0 (float): 初始波包动量(波数)，默认0.5
            d (float): 初始波包宽度参数，默认10
            barrier_width (int): 势垒宽度，默认3
            barrier_height (float): 势垒高度，默认1.0
        """
        self.Nx = Nx
        self.Nt = Nt
        self.x0 = x0
        self.k0 = k0
        self.d = d
        self.barrier_width = int(barrier_width)  # 确保是整数
        self.barrier_height = barrier_height
        
        # 创建空间网格
        self.x = np.arange(Nx)
        
        # 设置势垒
        self.V = self.setup_potential()
        
        # 初始化波函数矩阵和系数矩阵
        self.C = np.zeros((Nx, Nt), dtype=complex)  # 复数矩阵
        self.B = np.zeros((Nx, Nt), dtype=complex)  # 复数矩阵
    
    def wavefun(self, x):
        """高斯波包函数
        
        参数:
            x (np.ndarray): 空间坐标数组
            
        返回:
            np.ndarray: 初始波函数值
            
        数学公式:
            ψ(x,0) = exp(ik₀x) * exp(-(x-x₀)²ln10(2)/d²)
        """
        return np.exp(1j * self.k0 * x) * np.exp(-(x - self.x0)**2 * np.log(2) / self.d**2)
    
    def setup_potential(self):
        """设置势垒函数
        
        返回:
            np.ndarray: 势垒数组
            
        说明:
            在空间网格中间位置创建矩形势垒
            势垒位置：从 Nx//2 到 Nx//2+barrier_width
            势垒高度：barrier_height
        """
        V = np.zeros(self.Nx)
        barrier_start = self.Nx // 2
        barrier_end = barrier_start + self.barrier_width
        V[barrier_start:barrier_end] = self.barrier_height
        return V
    
    def build_coefficient_matrix(self):
        """构建变形的Crank-Nicolson格式的系数矩阵
        
        返回:
            np.ndarray: 系数矩阵A
            
        数学原理:
            对于dt=1, dx=1的情况，哈密顿矩阵的对角元素为: -2+2j-V
            非对角元素为1（表示动能项的有限差分）
        """
        main_diag = -2 + 2j - self.V
        off_diag = np.ones(self.Nx-1)
        
        # 构建三对角矩阵
        A = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        return A
    
    def solve_schrodinger(self):
        """求解一维含时薛定谔方程
        
        使用Crank-Nicolson方法进行时间演化
        
        返回:
            tuple: (x, V, B, C) - 空间网格, 势垒, 波函数矩阵, chi矩阵
        """
        A = self.build_coefficient_matrix()
        
        # 设置初始波函数
        self.B[:,0] = self.wavefun(self.x)
        
        # 归一化初始波函数
        norm = np.sqrt(np.sum(np.abs(self.B[:,0])**2))
        self.B[:,0] /= norm
        
        # 时间演化
        for t in range(self.Nt-1):
            # 解线性方程组 Aχ = 4j*B[:,t]
            self.C[:,t+1] = np.linalg.solve(A, 4j * self.B[:,t])
            
            # 更新波函数
            self.B[:,t+1] = self.C[:,t+1] - self.B[:,t]
        
        return self.x, self.V, self.B, self.C
    
    def calculate_coefficients(self):
        """计算透射和反射系数
        
        返回:
            tuple: (T, R) - 透射系数和反射系数
        """
        barrier_start = self.Nx // 2
        barrier_end = barrier_start + self.barrier_width
        
        # 计算最终波函数的概率密度
        final_prob = np.abs(self.B[:,-1])**2
        
        # 计算透射和反射区域的概率
        transmission_prob = np.sum(final_prob[barrier_end:])
        reflection_prob = np.sum(final_prob[:barrier_start])
        
        # 归一化
        total_prob = transmission_prob + reflection_prob
        T = transmission_prob / total_prob
        R = reflection_prob / total_prob
        
        return T, R
    
    def plot_evolution(self, time_indices=None):
        """绘制波函数演化图
        
        参数:
            time_indices (list): 要绘制的时间索引列表，默认为[0, Nt//4, Nt//2, 3*Nt//4, Nt-1]
        """
        if time_indices is None:
            time_indices = [0, self.Nt//4, self.Nt//2, 3*self.Nt//4, self.Nt-1]
        
        plt.figure(figsize=(12, 8))
        
        for i, t in enumerate(time_indices, 1):
            plt.subplot(len(time_indices), 1, i)
            
            # 绘制概率密度
            prob_density = np.abs(self.B[:,t])**2
            plt.plot(self.x, prob_density, 'b-', label='|ψ|²')
            
            # 绘制势垒
            plt.plot(self.x, self.V, 'r-', label='Potential')
            
            plt.title(f'Time step {t}')
            plt.ylabel('Probability Density')
            plt.legend()
        
        plt.xlabel('Position')
        plt.tight_layout()
        plt.show()
    
    def create_animation(self, interval=20):
        """创建波包演化动画
        
        参数:
            interval (int): 动画帧间隔(毫秒)，默认20
            
        返回:
            matplotlib.animation.FuncAnimation: 动画对象
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 初始化线条
        line_prob, = ax.plot([], [], 'b-', label='|ψ|²')
        line_potential, = ax.plot([], [], 'r-', label='Potential')
        
        ax.set_xlim(0, self.Nx)
        ax.set_ylim(0, max(np.max(np.abs(self.B)**2), np.max(self.V)) * 1.1)
        ax.set_xlabel('Position')
        ax.set_ylabel('Probability Density')
        ax.set_title('Quantum Tunneling Simulation')
        ax.legend()
        
        def init():
            line_prob.set_data([], [])
            line_potential.set_data([], [])
            return line_prob, line_potential
        
        def update(frame):
            prob_density = np.abs(self.B[:,frame])**2
            line_prob.set_data(self.x, prob_density)
            line_potential.set_data(self.x, self.V)
            return line_prob, line_potential
        
        ani = animation.FuncAnimation(
            fig, update, frames=self.Nt, init_func=init,
            interval=interval, blit=True
        )
        
        plt.close(fig)
        return ani
    
    def verify_probability_conservation(self):
        """验证概率守恒
        
        返回:
            np.ndarray: 每个时间步的总概率
        """
        return np.sum(np.abs(self.B)**2, axis=0)
    
    def demonstrate(self):
        """演示量子隧穿效应
        
        功能:
            1. 求解薛定谔方程
            2. 计算并显示透射和反射系数
            3. 绘制波函数演化图
            4. 验证概率守恒
            5. 创建并显示动画
            
        返回:
            animation对象
        """
        print("Solving Schrödinger equation...")
        self.solve_schrodinger()
        
        print("\nCalculating transmission and reflection coefficients...")
        T, R = self.calculate_coefficients()
        print(f"Transmission coefficient: {T:.4f}")
        print(f"Reflection coefficient: {R:.4f}")
        print(f"Sum (should be ~1): {T+R:.4f}")
        
        print("\nPlotting time evolution...")
        self.plot_evolution()
        
        print("\nVerifying probability conservation...")
        total_prob = self.verify_probability_conservation()
        print(f"Initial probability: {total_prob[0]:.6f}")
        print(f"Final probability: {total_prob[-1]:.6f}")
        print(f"Max deviation: {np.max(np.abs(total_prob - 1)):.6f}")
        
        print("\nCreating animation...")
        ani = self.create_animation()
        
        return ani


def demonstrate_quantum_tunneling():
    """便捷的演示函数
    
    创建默认参数的求解器并运行演示
    
    返回:
        animation对象
    """
    solver = QuantumTunnelingSolver()
    return solver.demonstrate()


if __name__ == "__main__":
    # 运行演示
    barrier_width = 3
    barrier_height = 1.0
    solver = QuantumTunnelingSolver(barrier_width=barrier_width, barrier_height=barrier_height)
    animation = solver.demonstrate()
