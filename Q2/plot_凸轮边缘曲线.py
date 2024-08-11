import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd


plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号


# 导入数据
data = pd.read_excel('data\\附件1-凸轮边缘曲线.xlsx')

# 极径
r = data['极径（mm）']
# 极角
theta = data['极角（rad）']

# 找到峰值和谷值
peaks, _ = find_peaks(r)  # 找到峰值
valleys, _ = find_peaks(-r)  # 通过反转找到谷值

# 检查是否找到峰值，如果没有找到，将左边缘作为峰值
if len(peaks) == 0:
    peaks = np.array([0])  # 将左边缘作为峰值
# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(theta, r, 'b-', label='凸轮边缘曲线')

# 标记峰值
plt.plot(theta[peaks], r[peaks], 'ro', label='峰值')
for peak in peaks:
    plt.text(theta[peak], r[peak], f'({theta[peak]:.2f}, {r[peak]:.2f})', 
             fontsize=9, verticalalignment='bottom', horizontalalignment='right')

# 标记谷值
plt.plot(theta[valleys], r[valleys], 'go', label='谷值')
for valley in valleys:
    plt.text(theta[valley], r[valley], f'({theta[valley]:.2f}, {r[valley]:.2f})', 
             fontsize=9, verticalalignment='top', horizontalalignment='right')

# 美化图表
plt.xlabel('极角 (rad)')
plt.ylabel('极径 (mm)')
plt.title('凸轮边缘曲线')
plt.grid(True)
plt.legend()
plt.tight_layout()

# 保存并展示图形
plt.savefig('figs\\凸轮边缘曲线_峰值与谷值标记.pdf', format='pdf')
plt.show()
