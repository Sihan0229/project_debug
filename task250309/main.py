import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl
import os
mpl.rcParams['font.sans-serif'] = ['STZhongsong']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


if __name__=='__main__':
    x = [1,2,3,4,5]
    xindex=['5','10','20','30','40']
    plt.xticks(x, xindex)  # 保持下标的一致性
    y=[0.7,0.8,0.9,1.0]

    plt.yticks(y)

    plt.xlabel('the number of returned images')  # x轴标题
    plt.ylabel('Accuracy')  # y轴标题
    plt.grid(color='gray',linestyle='-',linewidth=1,alpha=0.3) #设置背景网格

    # plt.plot(x,y,marker='d')


    plt.ylim((0.6,1.0))
    Accuracy = np.load('./CmpResult/Accuracy.npy')
    print(Accuracy)
    plt.plot(x, Accuracy[0], linestyle='-', marker='o', markersize=6)
    plt.plot(x, Accuracy[1], linestyle='-', marker='d', markersize=6)
    plt.plot(x, Accuracy[2], linestyle='-', marker='s', markersize=6)
    plt.plot(x, Accuracy[3], linestyle='-', marker='<', markersize=6)
    plt.legend(['50', '100', '150', '200'])  # 设置折线名称

    plt.show()  # 显示折线图

