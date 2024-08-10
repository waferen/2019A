import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
# 常量

# 导入数据
data = pd.read_csv('t_n_data.csv')
Time = data['Time']
delt_t = data['delt_t']

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制曲线
plt.plot(delt_t, Time, color='b', linewidth=2.5)

# 添加标题和标签
plt.title('到达150MPa所需时间 vs. 单向阀开启时间增量', fontsize=18)
plt.xlabel('Delt_t', fontsize=14)
plt.ylabel('Time', fontsize=14)

# 美化x轴和y轴
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 添加网格线
plt.grid(True, which='both', linestyle='--', linewidth=0.7)

# 保存图表
plt.savefig('figs\\到达150MPa所需时间vs单向阀开启时间增量.pdf', format='pdf')
# 显示图表
plt.show()
