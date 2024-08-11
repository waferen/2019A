import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from scipy.optimize import curve_fit

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
# 变量
P_inside = 100 # 内管压力
P_inlet = 160 # 入口压力
P_outlet = 0.1013 # 出口压力

# 常量

pai = 3.14 # 圆周率

# 油管
P_0 = 100 # 初始压力
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



# 凸轮曲线函数
## 给出时刻t return 极径
def cam_curve(w,t):
    theta = w*t
    return  -(peak - valley)/2 * (np.cos(theta))+ (peak + valley)/2

def Q(S,P_out,P_in,rho):
    return C*S*np.sqrt((P_out-P_in)*2/rho)

# 压力与弹性模量的关系E = poly3(P)
def E_poly3(x,Coefficients):
    return Coefficients[0]*x**3+Coefficients[1]*x**2+Coefficients[2]*x+Coefficients[3]

# 下标函数
def index(t):
    return int(100*round(t,2))

# return 油泵质量
def M_pump(t,T):
    m_pump_new = m_pump[index(t-delta_t)]-rho_pump[index(t-delta_t)]*I[index(t-delta_t)]*delta_t
    if t % T < 0.1:
        print(t)
        m_pump_new+=m_pump[0]
    m_pump.append(m_pump_new)
    return m_pump_new

# return 油泵容积
def V_pump(w,t):
    res = V_pump_min + pai*(5/2)**2*(cam_curve(w,t)-valley)
    v_pump.append(res)
    return res

# return 油泵压力，同时解出油泵密度
def P_pump(t,T):
    res = m_pump[index(t)]/v_pump[index(t)]
    p_pump.append(res)
    rho_pump.append(rho_Pump(res))
    return res

# 密度与压力的关系
def rho_Pump(P):
    res= np.exp(P/E_poly3(P,Coefficients))-c
    rho_pump.append(res)
    return res

# 标记函数
def sgn(x):
    if x>0:
        return 1
    elif x<0:
        return -1
    else:
        return 0

def f(t):
    return (sgn(p_pump[index(t)]-p[index(t)])*(sgn(p_pump[index(t)]-p[index(t)])+1))/2

def I_pump(t):
    if f(t)==0:
        res = 0
    else:
        res = f(t)*Q(A,p_pump[index(t)],p[index(t)],rho_pump[index(t)]) 
    I.append(res)
    return res 

def Rho_tube(P):
    res= np.exp(P/E_poly3(P,Coefficients))-c
    rho_tube.append(res)
    return res

def P_tube(t):
    res = p[index(t-delta_t)]+E_poly3(p[index(t-delta_t)],Coefficients)*(max(0,I[index(t-delta_t)])-e[index(t-delta_t)])*delta_t/rho_tube[index(t-delta_t)]/V
    p.append(res)
    return res

def S(t):
    if t % 100 >3:
        return 0
    else:
        L = np.sin(np.radians(9)) * data.iloc[int(t*100), 1]
        r=1.25
        R=r+L*np.cos(np.radians(9))
        return min(pai*L*(r+R),pai*(1.4/2)**2)

def E(t):
    res = Q(S(t),p[index(t)],P_outlet,rho_tube[index(t)])
    e.append(res)
    return res


# 主程序

# 初始化
rho_pump=[0.85] # 初始泵油密度
m_pump = [rho_pump[0]*V_pump_max] # 初始泵油质量
p_pump = [m_pump[0]/V_pump_max] # 初始泵油压力
v_pump = [V_pump_max] # 初始泵容积
I = [0] 
p = [100] # 油管压力
e = [0]
rho_tube = [0.85] # 燃油密度
# 角速度的遍历范围
w_range = np.arange(0, 2, 0.1)
loss_values = []
w_values = []
# 遍历角速度并计算对应的损耗
for w in w_range:
    # 重置
    rho_pump = [0.5]  # 初始泵油密度
    m_pump = [rho_pump[0] * V_pump_max]  # 初始泵油质量
    p_pump = [160]  # 初始泵油压力
    v_pump = [V_pump_max]  # 初始泵容积
    rho_tube = [0.85] # 燃油密度
    p = [100]  # 油管压力
    I = [0]
    e = [0]
    T = 2 * pai / w  # 周期
    print(w)
    for t in np.arange(0, 100, delta_t):
        M=M_pump(t,T)
        pp=P_pump(t, T)
        P=P_tube(t)
        Rho_tube(P)
        V_pump(w, t)
        ee=E(t)
        i=I_pump(t)
        print(pp)
    p_array = np.array(p)
    loss = (np.mean((p_array - P_inside)**2))
    loss_values.append(loss)
    w_values.append(w)

# 将 w 和 loss 成对存储到 DataFrame
res = pd.DataFrame({'w': w_values, 'loss': loss_values})

# 将 DataFrame 导出到 Excel 文件
res.to_excel('result\\w_loss_values.xlsx', index=False)


# 绘图
# 绘制loss和w的图
plt.figure(figsize=(10, 6))
plt.plot(w_range, loss_values, label='Loss vs Angular Velocity', color='b')
plt.xlabel('Angular Velocity (rad/ms)')
plt.ylabel('Loss')
plt.title('Loss vs Angular Velocity')
plt.legend()
plt.grid(True)

# 找出最小值点
min_loss = min(loss_values)
min_loss_w = w_range[loss_values.index(min_loss)]
plt.scatter(min_loss_w, min_loss, color='r', zorder=5)
plt.text(min_loss_w, min_loss, f'Min Loss: {min_loss:.2f} at w = {min_loss_w:.2f}', fontsize=12)
# 保存图片
plt.savefig('figs\\loss_w_2.pdf', format='pdf')
plt.show()
        
            
            
            
            
            
            
