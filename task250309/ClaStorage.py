import numpy as np
import os

def getFileSize(filePath):
    fsize = os.path.getsize(filePath.encode())
    fsize = fsize/float(1024 * 1024)
    return round(fsize, 8)

if __name__=='__main__':
    ClassNum=[50,100,150,200]
    IndexStorage=[]
    for Cla in ClassNum:
        filepath='./'+str(Cla)+'/EncTree.pkl'
        IndexStorage.append(getFileSize(filepath))  #索引存储空间
    print(IndexStorage)
    np.save('./Result/GenIndexStorage.npy',IndexStorage)

    GenKeyStorage=[]
    for Cla in ClassNum:
        file=['FeatH1','FeatH2','QueryH1','QueryH2']    #对每个类别的密钥文件（FeatH1, FeatH2, QueryH1, QueryH2）进行存储空间统计
        Sum=0
        for f in file:
            filepath='./'+str(Cla)+'/Key/'+f+'.npy'
            Sum+=getFileSize(filepath)
        GenKeyStorage.append(round(Sum,4))  # 索引存储空间
    np.save('./Result/GenKeyStorage.npy',GenKeyStorage)
    print(GenKeyStorage)
    #
    # GenKeyH1=np.load('./Result/GenKeyH1Time.npy')
    # GenKeyH2=np.load('./Result/GenKeyH2Time.npy')
    # GenKeyTime=[]
    # for i in range(4):
    #     GenKeyTime.append(GenKeyH2[i]+GenKeyH1[i])
    # print(GenKeyTime)
    # np.save('./Result/GenKeyTime.npy',GenKeyTime)


    # 计算模型存储空间
    Length = ['24_64','32_64','48_64','64_64']  #计算不同模型文件（如 24_64.hdf5）的大小，对应论文中对模型参数规模或轻量化设计的存储分析。
    ModelCost = []
    for i in range(4):
        dir = './' + str(ClassNum[i]) + "/Model/" + Length[i]+'.hdf5'
        ModelCost.append(getFileSize(dir))
    np.save('./Result/ModelCost.npy',ModelCost)
    print(ModelCost)
