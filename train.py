# 两层网络实现mnist数据集分类
import numpy as np
# from mnist import load_mnist
from layer_net import LayerNet
import matplotlib.pyplot as plt


# 读取保存的训练集的数据
x_train = np.load('params/X_train.npy')
t_train = np.load('params/T_train.npy')
# 打印训练集数据形状
print(x_train.shape)
print(t_train.shape)

train_loss_list = []
train_acc_list = []
test_acc_list = []
# 超参数人为修改
# 二手车数据集中训练集的数目为100000
sampling_num = 10000  # 循环取样次数（梯度法重复次数）200000
train_size = x_train.shape[0]
batch_size = 512   # 小样本数量
learning_rate = 0.01  # 学习率
# 代数
epoch = 200
# 实例化
net = LayerNet(input_size=x_train.shape[1], hidden_size1=512, hidden_size2=256,
               hidden_size3=128, hidden_size4=64, output_size=1)

epo = 0
for i in range(sampling_num):
    # mini-batch学习
    # 批的选择
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    # grads = net.numerical_gradient(x_batch, t_batch)
    grads = net.gradient(x_batch, t_batch)  # 高速模式
    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4', 'W5', 'b5',):
        net.params[key] -= learning_rate * grads[key]

    if i % epoch == 0:
        # 记录
        print("epoch:{}".format(epo))
        train_loss = net.loss(x_batch, t_batch)
        train_mae = net.mae_value(x_batch, t_batch)
        train_loss_list.append(train_loss)
        print("train_loss:" + str(train_loss) + " | train_mae:" + str(train_mae))
        epo = epo + 1

# 保存训练的权重
net.save_params()
print("Saved Network Parameters!")
# 因为第一次误差较大，影响图片呈现，故删去
x = np.arange(len(train_loss_list)-1)
plt.plot(x, train_loss_list[1:], label="loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("train_loss")
plt.legend()
plt.show()
