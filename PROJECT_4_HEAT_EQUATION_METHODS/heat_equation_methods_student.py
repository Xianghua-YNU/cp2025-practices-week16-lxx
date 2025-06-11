#!/usr/bin/env python3
"""
热传导方程数值解法比较
文件：heat_equation_methods_student.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from scipy.integrate import solve_ivp
import scipy.linalg
import time

class HeatEquationSolver:
    """
    热传导方程求解器，实现四种不同的数值方法。
    
    求解一维热传导方程：du/dt = alpha * d²u/dx²
    边界条件：u(0,t) = 0, u(L,t) = 0
    初始条件：u(x,0) = phi(x)
    """
    
    def __init__(self, L=20.0, alpha=10.0, nx=21, T_final=25.0):
        """
        初始化热传导方程求解器。
        
        参数:
            L (float): 空间域长度 [0, L]
            alpha (float): 热扩散系数
            nx (int): 空间网格点数
            T_final (float): 最终模拟时间
        """
        self.L = L
        self.alpha = alpha
        self.nx = nx
        self.T_final = T_final
        
        # 空间网格
        self.x = np.linspace(0, L, nx)
        self.dx = L / (nx - 1)
        
        # 初始化解数组
        self.u_initial = self._set_initial_condition()
        
    def _set_initial_condition(self):
        """
        设置初始条件：u(x,0) = 1 当 10 <= x <= 11，否则为 0。
        
        返回:
            np.ndarray: 初始温度分布
        """
        u = np.zeros(self.nx)
        mask = (self.x >= 10) & (self.x <= 11)
        u[mask] = 1.0
        # 应用边界条件
        u[0] = 0.0
        u[-1] = 0.0
        return u
    
    def solve_explicit(self, dt=0.01, plot_times=None):
        """
        使用显式有限差分法（FTCS）求解。
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 计算稳定性参数
        r = self.alpha * dt / (self.dx**2)
        
        # 检查稳定性条件
        if r > 0.5:
            print(f"Warning: Stability condition violated (r = {r:.3f} > 0.5)")
        
        # 初始化解和时间
        u = self.u_initial.copy()
        t = 0.0
        
        # 存储结果
        results = {
            'times': [],
            'solutions': [],
            'method': 'explicit',
            'computation_time': 0.0,
            'stability_parameter': r
        }
        
        # 存储初始条件
        results['times'].append(t)
        results['solutions'].append(u.copy())
        
        start_time = time.time()
        
        # 时间步进循环
        while t < self.T_final:
            # 计算下一时间步
            if t + dt > self.T_final:
                dt = self.T_final - t
                
            # 使用laplace计算空间二阶导数
            laplace_u = laplace(u, mode='reflect') / (self.dx**2)
            
            # 更新解
            u += self.alpha * dt * laplace_u
            
            # 应用边界条件
            u[0] = 0.0
            u[-1] = 0.0
            
            t += dt
            
            # 存储指定时间点的解
            for pt in plot_times:
                if (t >= pt) and (pt not in results['times']):
                    results['times'].append(pt)
                    results['solutions'].append(u.copy())
        
        results['computation_time'] = time.time() - start_time
        return results
    
    def solve_implicit(self, dt=0.1, plot_times=None):
        """
        使用隐式有限差分法（BTCS）求解。
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 计算扩散数
        r = self.alpha * dt / (self.dx**2)
        
        # 构建三对角矩阵（内部节点）
        n = self.nx - 2
        main_diag = np.ones(n) * (1 + 2*r)
        off_diag = np.ones(n-1) * (-r)
        
        # 使用带状矩阵格式
        A = np.zeros((3, n))
        A[0, 1:] = off_diag  # 上对角线
        A[1, :] = main_diag  # 主对角线
        A[2, :-1] = off_diag  # 下对角线
        
        # 初始化解和时间
        u = self.u_initial.copy()
        t = 0.0
        
        # 存储结果
        results = {
            'times': [],
            'solutions': [],
            'method': 'implicit',
            'computation_time': 0.0,
            'stability_parameter': r
        }
        
        # 存储初始条件
        results['times'].append(t)
        results['solutions'].append(u.copy())
        
        start_time = time.time()
        
        # 时间步进循环
        while t < self.T_final:
            if t + dt > self.T_final:
                dt = self.T_final - t
                
            # 构建右端项（内部节点）
            rhs = u[1:-1].copy()
            
            # 求解线性系统
            u_internal = scipy.linalg.solve_banded((1, 1), A, rhs)
            
            # 更新解
            u[1:-1] = u_internal
            
            # 应用边界条件
            u[0] = 0.0
            u[-1] = 0.0
            
            t += dt
            
            # 存储指定时间点的解
            for pt in plot_times:
                if (t >= pt) and (pt not in results['times']):
                    results['times'].append(pt)
                    results['solutions'].append(u.copy())
        
        results['computation_time'] = time.time() - start_time
        return results
    
    def solve_crank_nicolson(self, dt=0.5, plot_times=None):
        """
        使用Crank-Nicolson方法求解。
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 计算扩散数
        r = self.alpha * dt / (2 * self.dx**2)
        
        # 构建左端矩阵 A（内部节点）
        n = self.nx - 2
        main_diag = np.ones(n) * (1 + 2*r)
        off_diag = np.ones(n-1) * (-r)
        
        # 使用带状矩阵格式
        A = np.zeros((3, n))
        A[0, 1:] = off_diag  # 上对角线
        A[1, :] = main_diag  # 主对角线
        A[2, :-1] = off_diag  # 下对角线
        
        # 初始化解和时间
        u = self.u_initial.copy()
        t = 0.0
        
        # 存储结果
        results = {
            'times': [],
            'solutions': [],
            'method': 'crank_nicolson',
            'computation_time': 0.0,
            'stability_parameter': 2*r  # 实际稳定性参数是r=alpha*dt/dx²
        }
        
        # 存储初始条件
        results['times'].append(t)
        results['solutions'].append(u.copy())
        
        start_time = time.time()
        
        # 时间步进循环
        while t < self.T_final:
            if t + dt > self.T_final:
                dt = self.T_final - t
                r = self.alpha * dt / (2 * self.dx**2)
                
            # 构建右端向量
            rhs = r*u[:-2] + (1-2*r)*u[1:-1] + r*u[2:]
            
            # 求解线性系统
            u_internal = scipy.linalg.solve_banded((1, 1), A, rhs)
            
            # 更新解
            u[1:-1] = u_internal
            
            # 应用边界条件
            u[0] = 0.0
            u[-1] = 0.0
            
            t += dt
            
            # 存储指定时间点的解
            for pt in plot_times:
                if (t >= pt) and (pt not in results['times']):
                    results['times'].append(pt)
                    results['solutions'].append(u.copy())
        
        results['computation_time'] = time.time() - start_time
        return results
    
    def _heat_equation_ode(self, t, u_internal):
        """
        用于solve_ivp方法的ODE系统。
        """
        # 重构完整解向量（包含边界条件）
        u_full = np.zeros(self.nx)
        u_full[1:-1] = u_internal
        
        # 使用 laplace 计算二阶导数
        laplace_u = laplace(u_full, mode='reflect') / (self.dx**2)
        
        # 返回内部节点的时间导数
        return self.alpha * laplace_u[1:-1]
    
    def solve_with_solve_ivp(self, method='BDF', plot_times=None):
        """
        使用scipy.integrate.solve_ivp求解。
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 提取内部节点初始条件
        u0 = self.u_initial[1:-1].copy()
        
        start_time = time.time()
        
        # 调用 solve_ivp 求解
        sol = solve_ivp(
            fun=self._heat_equation_ode,
            t_span=(0, self.T_final),
            y0=u0,
            method=method,
            t_eval=plot_times
        )
        
        # 重构包含边界条件的完整解
        solutions = []
        for i in range(sol.y.shape[1]):
            u = np.zeros(self.nx)
            u[1:-1] = sol.y[:, i]
            solutions.append(u)
        
        results = {
            'times': sol.t.tolist(),
            'solutions': solutions,
            'method': f'solve_ivp ({method})',
            'computation_time': time.time() - start_time
        }
        
        return results
    
    def compare_methods(self, dt_explicit=0.01, dt_implicit=0.1, dt_cn=0.5, 
                       ivp_method='BDF', plot_times=None):
        """
        比较所有四种数值方法。
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        print("\nComparing heat equation solution methods:")
        print(f"- Explicit method (dt={dt_explicit})")
        print(f"- Implicit method (dt={dt_implicit})")
        print(f"- Crank-Nicolson method (dt={dt_cn})")
        print(f"- solve_ivp method ({ivp_method})")
        
        # 调用四种求解方法
        results = {
            'explicit': self.solve_explicit(dt=dt_explicit, plot_times=plot_times),
            'implicit': self.solve_implicit(dt=dt_implicit, plot_times=plot_times),
            'crank_nicolson': self.solve_crank_nicolson(dt=dt_cn, plot_times=plot_times),
            'solve_ivp': self.solve_with_solve_ivp(method=ivp_method, plot_times=plot_times)
        }
        
        # 打印每种方法的计算时间和稳定性参数
        print("\nMethod performance:")
        for method, res in results.items():
            if 'stability_parameter' in res:
                print(f"{res['method']:20s}: time = {res['computation_time']:.4f} s, r = {res['stability_parameter']:.4f}")
            else:
                print(f"{res['method']:20s}: time = {res['computation_time']:.4f} s")
        
        return results
    
    def plot_comparison(self, methods_results, save_figure=False, filename='heat_equation_comparison.png'):
        """
        绘制所有方法的比较图。
        """
        plt.figure(figsize=(12, 8))
        
        # 创建 2x2 子图
        for i, (method, res) in enumerate(methods_results.items()):
            plt.subplot(2, 2, i+1)
            
            # 绘制不同时间的解
            for t, u in zip(res['times'], res['solutions']):
                plt.plot(self.x, u, label=f't={t:.1f}')
            
            plt.title(res['method'])
            plt.xlabel('Position x')
            plt.ylabel('Temperature u(x,t)')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        
        if save_figure:
            plt.savefig(filename, dpi=300)
            print(f"Figure saved as {filename}")
        
        plt.show()
    
    def analyze_accuracy(self, methods_results, reference_method='solve_ivp'):
        """
        分析不同方法的精度。
        """
        # 验证参考方法存在
        if reference_method not in methods_results:
            raise ValueError(f"Reference method '{reference_method}' not found in results")
        
        # 获取参考解
        ref_results = methods_results[reference_method]
        
        # 计算各方法与参考解的误差
        accuracy = {}
        
        for method, res in methods_results.items():
            if method == reference_method:
                continue
                
            # 找到共同的时间点
            common_times = set(ref_results['times']) & set(res['times'])
            
            if not common_times:
                print(f"Warning: No common time points between {method} and reference")
                continue
                
            # 计算误差
            errors = []
            for t in common_times:
                # 找到对应时间点的解
                ref_idx = ref_results['times'].index(t)
                method_idx = res['times'].index(t)
                
                ref_sol = ref_results['solutions'][ref_idx]
                method_sol = res['solutions'][method_idx]
                
                # 计算绝对误差
                error = np.abs(method_sol - ref_sol)
                errors.append(error)
            
            if errors:
                max_error = np.max(errors)
                avg_error = np.mean(errors)
                
                accuracy[method] = {
                    'max_error': max_error,
                    'avg_error': avg_error,
                    'common_times': len(common_times)
                }
        
        # 打印精度分析结果
        print("\nAccuracy analysis (compared to reference method):")
        for method, err in accuracy.items():
            print(f"{method:20s}: max error = {err['max_error']:.4e}, avg error = {err['avg_error']:.4e}")
        
        return accuracy


def main():
    """
    HeatEquationSolver类的演示。
    """
    # 创建求解器实例
    solver = HeatEquationSolver(L=20.0, alpha=10.0, nx=101, T_final=25.0)
    
    # 比较所有方法
    results = solver.compare_methods(
        dt_explicit=0.001,
        dt_implicit=0.1,
        dt_cn=0.5,
        ivp_method='BDF',
        plot_times=[0, 1, 5, 15, 25]
    )
    
    # 绘制比较图
    solver.plot_comparison(results, save_figure=True)
    
    # 分析精度
    accuracy = solver.analyze_accuracy(results)
    
    return solver, results, accuracy


if __name__ == "__main__":
    solver, results, accuracy = main()
