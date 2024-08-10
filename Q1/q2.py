import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from scipy.optimize import curve_fit
import csv

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
# 常量

pai = 3.1415926 # 圆周率
P_0 = 100 # 初始压力
V = 500*(10/2)**2*pai # 油管容积
P_inside = 100 # 初始内管压力
P_inlet = 160 # 入口压力
P_outlet = 100 # 出口压力

rho = 0.85 # 燃油密度

A = pai * (1.4/2)**2 # 截面积
T = 100 # 划分周期
C = 0.85 # 流量系数
delta_p = P_inlet - P_inside # A小孔两边压差
c = 100/2171.4-np.log(0.85) # 常数
p=[ 1.00037752e-04,-1.08248140e-03, 5.47444434e+00, 1.53186841e+03]
# 变量


T_open_100=2.769230769230769 # 100MPa稳定时，单向阀开启时长

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
    plt.savefig('figs\\loss与T_open的关系.pdf', format='pdf')


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

def find_closest(target, t_n):
    closest_pair = min(t_n, key=lambda x: abs(x[0] - target))
    return closest_pair

# 主程序

# 导入数据
data = pd.read_excel('data\附件3-弹性模量与压力.xlsx')

# 提取第一列和第二列数据
x = data.iloc[:, 0]
y = data.iloc[:, 1]

# 提取压力和弹性模量数据
x = data['压力(MPa)']
y = data['弹性模量(MPa)']

t_n=[]
t_e1=2000
t_e2=5000
t_e3=10000

for delt_t in np.arange(0, 2, 0.1):
    if delt_t==0:
        continue
    P_t=[P_0] # 油管随时间变化的压力
    t=0.1
    print('delt_t:',delt_t)
    while P_t[-1] < 150:
        P(round(t,1),T_open_100+delt_t)
        if P_t[-1]== P_t[-2]:
            while round(t,1) % 100 != 0:
                P_t.append(P_t[-1])
                t+=0.1
        t+=0.1
        print('t:',t,'P_t:',P_t[-1])
    t_n.append((round(t,1), delt_t))

# 将 t_n 列表写入 CSV 文件
with open('t_n_data.csv', 'w', newline='',encoding='UTF-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'delt_t'])
    writer.writerows(t_n)

print("数据已保存到 t_n_data.csv 文件中")

with open('result\\result_2.txt', 'w',encoding='UTF-8') as f:
    # 找到最接近 t_e1, t_e2 和 t_e3 的 t 值以及对应的 delt_t
    closest_t_e1, closest_delt_t_e1 = find_closest(t_e1, t_n)
    closest_t_e2, closest_delt_t_e2 = find_closest(t_e2, t_n)
    closest_t_e3, closest_delt_t_e3 = find_closest(t_e3, t_n)

    print("最接近 t_e1 的 t 值为:", closest_t_e1, "对应的 delt_t 为:", closest_delt_t_e1, file=f)
    print("最接近 t_e2 的 t 值为:", closest_t_e2, "对应的 delt_t 为:", closest_delt_t_e2, file=f)
    print("最接近 t_e3 的 t 值为:", closest_t_e3, "对应的 delt_t 为:", closest_delt_t_e3, file=f)

