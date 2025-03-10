import numpy as np
import h5py
from sklearn.decomposition import PCA

def saveFeatures(dir,Feats,Paths):
    h5y = h5py.File(dir, 'w')
    h5y.create_dataset('feats', data=Feats)
    h5y.create_dataset('paths', data=Paths)
    h5y.close()

def loadFeatures(dir):
    f = h5py.File(dir, 'r')
    feats = f['feats'][:]
    paths = f["paths"][:]
    f.close()
    return feats, paths

if __name__=='__main__':
    # Center_num=200
    # MaxNum=80
    # th=0.8
    # centers=[]
    # path='./200/512features.h5'
    # feats,labels=loadFeatures(path)
    # print(feats.shape)
    # pca = PCA(n_components=64)
    # feats=np.squeeze(feats)
    # LowFeats=pca.fit_transform(feats)
    # S=[]
    # for cla in range(Center_num):
    #     S=LowFeats[cla*MaxNum]
    #     for i in range(cla*MaxNum+1,cla*MaxNum+MaxNum):
    #         S+=LowFeats[i]
    #     S/=MaxNum
    #     centers.append(S)
    # centers=np.squeeze(centers)
    #
    # centers[centers>=th]=1
    # centers[centers<th]=0
    # np.save('./200/64HashCode.npy',centers)
    centers=np.load('./200/64HashCode.npy')
    print(centers.shape)
    res=[]
    sum=0
    for i in range(len(centers)):
        dis=[]
        for j in range(len(centers)):
            num=0
            for k in range(64):
                if centers[i][k]!=centers[j][k]:
                    num+=1
            if num<=3:
                sum+=1
            dis.append(num)
        res.append(dis)
    for i in range(len(centers)):
        print(res[i])
    print(sum,(sum-len(centers))/2)

