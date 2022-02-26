import numpy as np
from layer_net import LayerNet
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.ensemble import RandomForestRegressor  # 随机森林
# from sklearn.model_selection import cross_val_score  # 交叉验证
# import pickle
# import matplotlib.pyplot as plt
# f = open('params/params.pkl', 'rb')
# data = pickle.load(f)
# print(data)

# 读取保存的测试集数据
x_test = np.load('params/X_test.npy')
t_test = np.load('params/T_test.npy')

# # 读取保存的训练集的数据
# x_train = np.load('params/X_train.npy')
# t_train = np.load('params/T_train.npy')
# train_size = x_train.shape[0]
# batch_size = 512   # 小样本数量
"""
下面采用随机森林的方法预测结果
"""
# # 随机森林模型引入
# forest_reg = RandomForestRegressor()


# # 模型拟合效果的指标（包括MAR、MSE、RMSE)
# def model_goodness(model, x, y):
#     prediction = model.predict(x)
#     mae = mean_absolute_error(y, prediction)
#     mse = mean_squared_error(y, prediction)
#     rmse = np.sqrt(mse)
#     print('MAE:', mae)
#     print('MSE:', mse)
#     print('RMSE:', rmse)


# # 模型泛化能力的指标计算函数
# def display_scores(scores):
#     print("Scores:", scores)
#     print("Mean:", scores.mean())
#     print("Standard deviation:", scores.std())


# forest_reg.fit(x_train, t_train)
# # 评价模型
# model_goodness(forest_reg, x_train, t_train)
# # 10折交叉验证来验证模型的泛化性能
# scores = cross_val_score(forest_reg, x_train,
#                          t_train, scoring='neg_mean_absolute_error', cv=10)
# mae_scores = np.abs(-scores)
# display_scores(mae_scores)

"""
下面采用神经网络的方法预测结果
"""
# 建立前馈网络
net = LayerNet(input_size=x_test.shape[1], hidden_size1=512, hidden_size2=256,
               hidden_size3=128, hidden_size4=64, output_size=1)
# 导入权重参数
net.load_params()
# 得到测试集的预测输出
t_predict = net.predict(x_test)
# 计算测试集总的MAE
MAE = net.mae_value(x_test, t_test)
# 因为price数据在保存时取了对数，在应用R^2时应进行反操作
SS_R = np.sum((np.exp(t_test) - np.exp(t_predict))**2)
SS_T = np.sum((np.exp(t_test) - np.mean(np.exp(t_test)))**2)
R2 = 1 - (float(SS_R))/(float(SS_T))

print("MAE: " + str(MAE))
print("SSR: " + str(SS_R))
print("SST: " + str(SS_T))
print("R2: " + str(R2))
