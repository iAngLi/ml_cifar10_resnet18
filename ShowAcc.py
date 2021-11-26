import numpy as np
import matplotlib.pyplot as plt


def Plot():
    plt.figure()
    plt.title('ResNet18-CIFAR_10')
    plt.xlabel('ITER')
    plt.ylabel('ACC')
    plt.grid(True)
    return plt


# 训练
if __name__ == "__main__":
    dataSet = []
    with open('log.txt') as fileIn:
        for line in fileIn.readlines():
            lineArr = line.strip().split(' ')
            dataSet.append([int(lineArr[2]), float(lineArr[7].strip('%'))])

    dataSet = np.array(dataSet)
    plt = Plot()
    plt.plot(dataSet[:, 0], dataSet[:, 1], 'b-')
    plt.show()
