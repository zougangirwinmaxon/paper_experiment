import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# np.random.seed(1)
#绘制每个回合奖励图像#######################################################################################################
#
# # 读取Excel文件
# file_path1 = 'ddpg奖励函数episodes3.xlsx'  # 替换为你的Excel文件路径
# df = pd.read_excel(file_path1, engine='openpyxl')  # 使用openpyxl引擎
#
# # 获取数据
# y = df.iloc[:, 0].values  # 获取第一列的数据作为y值
# x = np.arange(1, len(y) + 1)  # 生成x值，表示每个回合
#
# # 多项式拟合
# degree = 7  # 多项式的阶数
# coefficients = np.polyfit(x, y, degree)  # 进行多项式拟合
# polynomial = np.poly1d(coefficients)  # 生成多项式函数
#
# # 计算每 x 个回合的滑动平均
# window_size = 25
# rolling_avg = np.convolve(y, np.ones(window_size) / window_size, mode='valid')  # 计算滑动平均
# rolling_x = np.arange(1, len(rolling_avg) + 1)  # 修正X轴，使长度匹配
#
# # 绘制图形
# plt.figure(figsize=(8, 5))
# plt.plot(x, y, color='blue', alpha=0.5, label='DDPG Reward (Original)')  # 原始奖励数据
# plt.plot(x, polynomial(x), linewidth=1, color='black', label='Polyfit')  # 多项式拟合曲线
# plt.plot(rolling_x, rolling_avg, linewidth=2, color='red', linestyle='--', label='Rolling Avg (25 Episodes)')  # 滑动平均曲线
#
# # 轴标签、图例和标题
# plt.xlabel("Episodes")
# plt.ylabel("Reward")
# plt.legend()
# plt.title('DDPG Reward Function')
#
# # 显示图像
# plt.show()

# #绘制每个回合成功任务数#####################################################################################################

# 读取Excel文件
file_path5 = 'ddpg每回合成功任务数episodes3.xlsx'  # 替换为你的Excel路径
df = pd.read_excel(file_path5, engine='openpyxl')  # 读取数据

# 获取数据
y = df.iloc[:, 0].values  # 获取第一列的数据作为y值
x = np.arange(1, len(y) + 1)  # 生成x轴，表示每个回合

# 多项式拟合
degree = 7  # 多项式的阶数
coefficients = np.polyfit(x, y, degree)  # 进行多项式拟合
polynomial = np.poly1d(coefficients)  # 生成多项式函数

# 计算每 x 个回合的平均值
window_size = 25
rolling_avg = np.convolve(y, np.ones(window_size)/window_size, mode='valid')  # 计算滑动平均
rolling_x = np.arange(1, len(rolling_avg) + 1)  # 修正X轴，使长度匹配

# 绘制图形
plt.figure(figsize=(8, 5))
plt.plot(x, y, color='green', alpha=0.5, label='Success (Original)')  # 原始数据
plt.plot(x, polynomial(x), linewidth=1.5, color='black', label='Polyfit')  # 多项式拟合曲线
plt.plot(rolling_x, rolling_avg, linewidth=2, color='red', linestyle='--', label='Rolling Avg (25 Episodes)')  # 滑动平均曲线

# 轴标签、图例和标题
plt.xlabel("Episode")
plt.ylabel("Number of Success")
plt.legend()
plt.title('DDPG Success Rate')

# 显示图像
plt.show()

#绘制每个回合所有终端的能耗#####################################################################################################

# 读取Excel文件
file_path4 = 'ddpg每个回合所有终端能耗ep3.xlsx'  # 替换为你的Excel文件路径
df = pd.read_excel(file_path4, engine='openpyxl')  # 使用openpyxl引擎

# 获取数据
y = df.iloc[:, 0].values  # 获取第一列的数据作为y值
x = np.arange(1, len(y) + 1)  # 生成x值，表示回合数，从1开始

# 多项式拟合
degree = 7  # 多项式的阶数
coefficients = np.polyfit(x, y, degree)  # 进行多项式拟合
polynomial = np.poly1d(coefficients)  # 生成多项式函数

# 计算每 x 个回合的滑动平均
window_size = 25
rolling_avg = np.convolve(y, np.ones(window_size) / window_size, mode='valid')  # 计算滑动平均
rolling_x = np.arange(1, len(rolling_avg) + 1)  # 让X轴长度匹配滑动平均

# 绘制图形
plt.figure(figsize=(8, 5))
plt.plot(x, y, color='red', alpha=0.5, label='UE Energy Consumption (Original)')  # 原始数据
plt.plot(x, polynomial(x), linewidth=1.5, color='black', label='Polyfit')  # 多项式拟合曲线
plt.plot(rolling_x, rolling_avg, linewidth=2, color='blue', linestyle='--', label='Rolling Avg (25 Episodes)')  # 滑动平均曲线

# 轴标签、图例和标题
plt.xlabel("Episodes")
plt.ylabel("UE Energy Consumption")
plt.legend()
plt.title('DDPG UE Energy Consumption')

# 显示图像
plt.show()

#绘制每个回合无人机能耗#####################################################################################################

# 读取Excel文件
file_path3 = 'ddpg每个回合无人机的能耗episodes3.xlsx'  # 替换为你的Excel文件路径
df = pd.read_excel(file_path3, engine='openpyxl')  # 使用openpyxl引擎

# 获取数据
y = df.iloc[:, 0].values  # 获取第一列的数据作为y值
x = np.arange(1, len(y) + 1)  # 生成x轴，表示每个回合

# 多项式拟合
degree = 7  # 多项式的阶数
coefficients = np.polyfit(x, y, degree)  # 进行多项式拟合
polynomial = np.poly1d(coefficients)  # 生成多项式函数

# 计算每 x 个回合的滑动平均
window_size = 25
rolling_avg = np.convolve(y, np.ones(window_size) / window_size, mode='valid')  # 计算滑动平均
rolling_x = np.arange(1, len(rolling_avg) + 1)  # 让X轴长度匹配滑动平均

# 绘制图形
plt.figure(figsize=(8, 5))
plt.plot(x, y, color='blue', alpha=0.5, label='UAV Energy Consumption (Original)')  # 原始数据
plt.plot(x, polynomial(x), linewidth=1.5, color='black', label='Polyfit')  # 多项式拟合曲线
plt.plot(rolling_x, rolling_avg, linewidth=2, color='red', linestyle='--', label='Rolling Avg (25 Episodes)')  # 滑动平均曲线

# 轴标签、图例和标题
plt.xlabel("Episode")
plt.ylabel("UAV Energy Consumption")
plt.legend()
plt.title('DDPG UAV Energy Consumption')

# 显示图像
plt.show()

#########################################################################################################################
#
# # 读取Excel文件
# file_path3 = 'ddpg每个回合成功卸载到无人机的任务数episodes3.xlsx'  # 替换为你的Excel文件路径
# df = pd.read_excel(file_path3, engine='openpyxl')  # 使用openpyxl引擎
#
# # 获取数据
# y = df.iloc[:, 0].values  # 获取第一列的数据作为y值
# x = np.arange(1, len(y) + 1)  # 生成x轴，表示每个回合
#
# # 多项式拟合
# degree = 7  # 多项式的阶数
# coefficients = np.polyfit(x, y, degree)  # 进行多项式拟合
# polynomial = np.poly1d(coefficients)  # 生成多项式函数
#
# # 计算每 x 个回合的滑动平均
# window_size = 25
# rolling_avg = np.convolve(y, np.ones(window_size) / window_size, mode='valid')  # 计算滑动平均
# rolling_x = np.arange(1, len(rolling_avg) + 1)  # 让X轴长度匹配滑动平均
#
# # 绘制图形
# plt.figure(figsize=(8, 5))
# plt.plot(x, y, color='blue', alpha=0.5, label='UAV success (Original)')  # 原始数据
# plt.plot(x, polynomial(x), linewidth=1.5, color='black', label='Polyfit')  # 多项式拟合曲线
# plt.plot(rolling_x, rolling_avg, linewidth=2, color='red', linestyle='--', label='Rolling Avg (25 Episodes)')  # 滑动平均曲线
#
# # 轴标签、图例和标题
# plt.xlabel("Episode")
# plt.ylabel("UAV SUCCESS")
# plt.legend()
# plt.title('DDPG UAV SUCCESS')
#
# # 显示图像
# plt.show()
#
# #########################################################################################################################
#
# # 读取Excel文件
# file_path3 = 'ddpg每个回合成功卸载到卫星的任务数episodes3.xlsx'  # 替换为你的Excel文件路径
# df = pd.read_excel(file_path3, engine='openpyxl')  # 使用openpyxl引擎
#
# # 获取数据
# y = df.iloc[:, 0].values  # 获取第一列的数据作为y值
# x = np.arange(1, len(y) + 1)  # 生成x轴，表示每个回合
#
# # 多项式拟合
# degree = 7  # 多项式的阶数
# coefficients = np.polyfit(x, y, degree)  # 进行多项式拟合
# polynomial = np.poly1d(coefficients)  # 生成多项式函数
#
# # 计算每 x 个回合的滑动平均
# window_size = 25
# rolling_avg = np.convolve(y, np.ones(window_size) / window_size, mode='valid')  # 计算滑动平均
# rolling_x = np.arange(1, len(rolling_avg) + 1)  # 让X轴长度匹配滑动平均
#
# # 绘制图形
# plt.figure(figsize=(8, 5))
# plt.plot(x, y, color='blue', alpha=0.5, label='satellite success (Original)')  # 原始数据
# plt.plot(x, polynomial(x), linewidth=1.5, color='black', label='Polyfit')  # 多项式拟合曲线
# plt.plot(rolling_x, rolling_avg, linewidth=2, color='red', linestyle='--', label='Rolling Avg (25 Episodes)')  # 滑动平均曲线
#
# # 轴标签、图例和标题
# plt.xlabel("Episode")
# plt.ylabel("SATELLITE SUCCESS")
# plt.legend()
# plt.title('DDPG SATELLITE SUCCESS')
#
# # 显示图像
# plt.show()
