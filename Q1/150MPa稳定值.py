import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from scipy.optimize import curve_fit

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
# 常量

pai = 3.1415926 # 圆周率
P_0 = 150 # 初始压力
V = 500*(10/2)**2*pai # 油管容积
P_inside = 150 # 初始内管压力
P_inlet = 160 # 入口压力
P_outlet = 100 # 出口压力

rho = 0.85 # 燃油密度

A = pai * (1.4/2)**2 # 截面积
T = 100 # 划分周期
C = 0.85 # 流量系数
delta_p = P_inlet - P_inside # A小孔两边压差
c = 100/2171.4-np.log(0.85) # 常数



def plot_1(x,y,y_poly_fit):
    # 绘制原始数据
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='原始数据')

    # 绘制多项式拟合曲线
    x_fit = np.linspace(min(x), max(x), 500)
    y_poly_fit = np.polyval(p, x_fit)
    plt.plot(x_fit, y_poly_fit, 'm--', label='三次多项式拟合')

    # 美化图表
    plt.title('压力与弹性模量的关系')
    plt.xlabel('弹性模量 (MPa)')
    plt.ylabel('压力 (MPa)')
    plt.grid(True)  # 添加网格线
    plt.legend()  # 添加图例
    plt.tight_layout()
    # 保存矢量图表
    plt.savefig('figs\\压力与弹性模量的关系.pdf', format='pdf')


def plot_2(T_open,loss):
    # 创建一个图形对象
    plt.figure(figsize=(10, 6))
    
    # 绘制原始数据
    plt.plot(T_open, loss, 'b-', label='Loss', linewidth=2, markersize=6)

    # 美化图表
    plt.title('Loss与T_open的关系', fontsize=16, fontweight='bold')
    plt.xlabel('T_open/ms', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # 添加网格线
    plt.legend(fontsize=12)  # 设置图例字体大小
    plt.tight_layout()

    # 保存矢量图表
    plt.savefig('figs\\150MPa loss与T_open的关系.pdf', format='pdf')


# 压力与弹性模量的关系E = poly3(P)
def E_poly3(x,p):
    return p[0]*x**3+p[1]*x**2+p[2]*x+p[3]
# 密度与压力的关系
def rho_P(P):
    return np.exp(P/E_poly3(P,p)-c)


# E_j(T_0) 为喷出高压油管的燃油流量,因为这里的每一个时刻是1ms，所以等价于该时刻喷出燃油量
# T_0为喷油时刻,t为喷油工作周期,T_0属于[0,100-2.4]
def E_j(t):
    T_0 = 0
    t = t % 100
    T = 2.4 # 喷油工作时长
    if t < T_0 or t >= T_0+T:
        return 0
    elif T_0 <= t < T_0+0.2:
        return 100*t
    elif T_0+0.2 <= t < T_0+2.2:
        return 20
    elif T_0+2.2 <= t < T_0+2.4:
        return 240-100*t


# I_j(T_open) 为该t时刻进入油管的燃油量 T_open为单向阀开启时长，T_0为开始供油时刻
def I_j(t,T_open):
    T_0 = 0
    t = t % 100
    if t < T_0 or t > T_0+T_open:
        return 0
    elif T_0 <= t <= T_0+T_open:
        return C*A*np.sqrt(2*(P_inlet-P_t[int(10*t)])/rho)

# 求解P_t,将求解出的p_t存入P_t中
def P(t,T_open):
    P_t.append(P_0+sum([E_poly3(P_t[int(10*j)],p)/rho_P(P_t[int(10*j)])/V*(I_j(j,T_open)-E_j(j)) for j in np.arange(0, t, 0.1)]))
    return P_t[int(10*t)]

# 主程序

# 导入数据
data = pd.read_excel('data\附件3-弹性模量与压力.xlsx')

# 提取第一列和第二列数据
x = data.iloc[:, 0]
y = data.iloc[:, 1]

# 提取压力和弹性模量数据
x = data['压力(MPa)']
y = data['弹性模量(MPa)']

# 多项式拟合（例如三次多项式）
p = np.polyfit(x, y, 3)  # 拟合三次多项式
y_poly_fit = np.polyval(p, x)
print('多项式拟合系数:', p)

# 绘制图表
#plot_1(x, y, y_poly_fit)

# 优化
# 优化目标：使得P_t与P_inside的均方误差最小
# 优化变量：T_open
# 优化约束：T_open属于[0,100-10]
# 优化方法：穷举搜索

with open('result\\result_150.txt',  'w',encoding='UTF-8') as f:
    T_Open = np.linspace(0, 8, 40)
    loss = []
    for T_open in T_Open:
        P_t = []
        for t in np.arange(0, 100, 0.1):
            P(round(t,1),T_open)
        loss.append(np.mean([(p - P_inside) ** 2 for p in P_t]))
        print('T_open:', T_open, 'loss:', loss[-1])
    T_open_opt = T_Open[np.argmin(loss)]
    print('最优每次开阀时长:', T_open_opt,file=f)
    print('最小均方误差:', min(loss),file=f)
    plot_2(T_Open, loss)

# 当稳定压力为100MPa时，每个周期内开阀时间为0.2876ms，
# 这里是默认压力已经稳定在100MPa并且开阀时间每次都一样,从而反推出开阀时间
# 当稳定压力为150MPa时，每个周期内开阀时间为0.7518ms
# 那么我们能不能在100ms中设置几个固定的单向阀开启时间点，因为开一次要10ms冷却，//
# 所以这些时间点要至少间隔10ms，
# 因为我们反推出的开阀时间很小，
# 我们假设开阀时间在0-1ms，
# 那我们可以每11ms设置一个开阀的时间点。
# 一个100ms周期内放10个。那么优化得到的是一个向量