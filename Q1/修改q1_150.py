import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from scipy.optimize import curve_fit

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
# 变量
P_inside = 150 # 内管压力
P_inlet = 160 # 入口压力
P_outlet = 0.1013 # 出口压力

# 常量

pai = 3.14 # 圆周率

# 油管
P_0 = 150 # 初始压力
V = 500*(10/2)**2*pai # 油管容积
A = pai * (1.4/2)**2 # 入口截面积
#T = 100 # 划分周期
C = 0.85 # 流量系数
delta_p = P_inlet - P_inside # A小孔两边压差
c = 100/2171.4-np.log(0.85) # 常数
Coefficients=[ 1.00037752e-04,-1.08248140e-03, 5.47444434e+00, 1.53186841e+03]#系数


# 油泵
peak =7.24 # 峰值
valley = 2.41 # 谷值
V_pump_min = 20 # 泵最小容积
V_pump_max = V_pump_min+pai*(5/2)**2*(peak-valley) # 初始容积(泵最大容积)=114.78875000000001


delta_t = 0.01 # 时间间隔

data = pd.read_excel('data\\附件2-针阀运动曲线.xlsx')



def E_j(t):
    T_0 = 0
    t = t % 100
    T = 2.4 # 喷油工作时长
    if t < T_0 or t >= T_0+T:
        res=0
    elif T_0 <= t < T_0+0.2:
        res= 100*t
    elif T_0+0.2 <= t < T_0+2.2:
        res= 20
    elif T_0+2.2 <= t < T_0+2.4:
        res= 240-100*t
    e_j.append(res)
    return res


# I_j(T_open) 为该t时刻进入油管的燃油量 T_open为单向阀开启时长，T_0为开始供油时刻
def I_j(t,T_open):
    T_0 = 0
    t = t % 100
    if t < T_0 or t > T_0+T_open:
        res=0
    elif T_0 <= t <= T_0+T_open:
        res= C*A*np.sqrt(2*(P_inlet-P_t[index(t)])/rho_tube[index(t)])
    i_j.append(res)
    return res

# 压力与弹性模量的关系E = poly3(P)
def E_poly3(x,Coefficients):
    return Coefficients[0]*x**3+Coefficients[1]*x**2+Coefficients[2]*x+Coefficients[3]

# 下标函数
def index(t):
    return int(100*round(t,2))

def Rho_tube(P):
    res= np.exp(P/E_poly3(P,Coefficients))-c
    rho_tube.append(res)
    return res

def P_tube(t):
    res = P_t[index(t-delta_t)]+E_poly3(P_t[index(t-delta_t)],Coefficients)*(max(0,i_j[index(t-delta_t)])-e_j[index(t-delta_t)])*delta_t/rho_tube[index(t-delta_t)]/V
    P_t.append(res)
    return res

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
    plt.savefig('figs\\150MPa loss与T_open的关系_2.pdf', format='pdf')
     

i_j=[0]
e_j=[0]
P_t = [100]
rho_tube=[0.85]
# 主程序
with open('result\\result_150_2.txt',  'w',encoding='UTF-8') as f:
    T_Open = np.linspace(0, 100, 40)
    loss = []
    for T_open in T_Open:
        i_j=[0]
        e_j=[0]
        P_t = [150]
        rho_tube=[0.85]
        for t in np.arange(0, 100, 0.01):
            P_tube(t)
            Rho_tube(P_tube(t))
            I_j(t,T_open)
            E_j(t)
        loss.append(np.sum([(p - P_inside) ** 2 for p in P_t]))
        print('T_open:', T_open, 'loss:', loss[-1])
    T_open_opt = T_Open[np.argmin(loss)]
    print('最优每次开阀时长:', T_open_opt,file=f)
    print('最小误差:', min(loss),file=f)
    plot_2(T_Open,loss)

            

            
            
            
            
