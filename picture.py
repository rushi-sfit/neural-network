

import matplotlib.pyplot as plt
from scipy import signal
import numpy as np



# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")

    return np.asfarray(data, float)


# 不同长度数据，统一为一个标准，倍乘x轴
def multiple_equal(x, y):
    x_len = len(x)
    y_len = len(y)
    times = x_len/y_len
    y_times = [i * times for i in y]
    return y_times


if __name__ == "__main__":

    # train_loss_path = r"C:\Users\lenovo\Desktop\文献\文献代码\Differential-Privacy-Based-Federated-Learning-master\fashion-mnist_LDPGJoPEQ_acc_10niid.txt"
    train_ACC_path = r"C:\Users\lenovo\Desktop\文献\文献代码\Differential-Privacy-Based-Federated-Learning-master\mnist_LDP4GJoPEQ_acc_10niid.txt"#LDP
    # train_LOSS_path = r"C:\Users\lenovo\Desktop\文献\文献代码\Differential-Privacy-Based-Federated-Learning-master\fashion-mnist_JoPEQ_loss_FedavgR=2_10niid.txt"
    # train_acc_path = r"C:\Users\lenovo\Desktop\文献\文献代码\Differential-Privacy-Based-Federated-Learning-master\cifar-10_1_dp_acc_xiugai_10niid.txt"
    train_yuan_loss_path = r"C:\Users\lenovo\Desktop\文献\文献代码\Differential-Privacy-Based-Federated-Learning-master\mnist_4_gasdp_acc_yuan_10niid.txt"#DP-FL
    # train_yuan_acc_path = r"C:\Users\lenovo\Desktop\文献\文献代码\Differential-Privacy-Based-Federated-Learning-master\fashion-mnist_bianJoPEQ_loss_FedavgR=2_10niid.txt"
    # train_prune_loss_path = r"C:\Users\lenovo\Desktop\文献\文献代码\Differential-Privacy-Based-Federated-Learning-master\fashion-mnist_no_dp_loss_10niid.txt"
    train_prune_acc_path =r"C:\Users\lenovo\Desktop\文献\文献代码\Differential-Privacy-Based-Federated-Learning-master\mnist_4_gasdp_acc_xin_10niid.txt"#IsmDP-FL
    # y_prune = data_read(train_prune_loss_path)
    y_prune_acc = data_read(train_prune_acc_path)
    # y_train_loss = data_read(train_loss_path)
    # y_train_Loss = data_read(train_LOSS_path)
    y_train_yuan_loss = data_read(train_yuan_loss_path)
    # y_train_yuan_acc = data_read(train_yuan_acc_path)
    # x_train_loss = range(len(y_train_loss))
    # x_train_Loss =range(len(y_train_Loss))
    x_train_yuan_loss = range(len(y_train_yuan_loss))
    # x_train_yuan_acc = range(len(y_train_yuan_acc))
    y_train_ACC = data_read(train_ACC_path)
    # y_train_acc = data_read(train_acc_path)

    x_train_ACC = range(len(y_train_ACC))
    # x_train_acc = range(len(y_train_acc))
    # x_prune = range(len(y_prune))
    x_prune_acc = range(len(y_prune_acc))

    b, a = signal.butter(1, 0.05)  # 设计滤波器
    smoothed_y = signal.filtfilt(b, a, y_train_yuan_loss)  # 应用滤波器
    # smoothed_x = signal.filtfilt(b, a, y_train_Loss)
    # smoothed = signal.filtfilt(b, a, y_train_yuan_acc)
    smoothed1 = signal.filtfilt(b, a, y_train_ACC)
    smoothed2 = signal.filtfilt(b, a, y_prune_acc)
    plt.figure()

    # 去除顶部和右边框框
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(x_train_yuan_loss,smoothed_y,"--",color='blue',linewidth=1, label="DP-FL")

    # plt.plot(x_train_Loss, smoothed_x,  color='purple', ls="--",label="JoPEQ(lattice=1)")
    #
    # plt.plot(x_train_yuan_acc,smoothed,"--",color='red',linewidth=1, label="JoPEQ(lattice=2)")
    # #
    # # # plt.plot(x_train_acc, y_train_acc,  color='red', ls="--",label="PPFL")
    plt.plot(x_train_ACC, smoothed1, "--",color='black',linewidth=1, label="LDP")
    plt.plot(x_prune_acc,smoothed2,"--",color="green",linewidth=1, label="IsmDP-FL")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Accuracy curve')
    plt.show()