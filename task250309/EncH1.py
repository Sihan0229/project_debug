import numpy as np
import h5py
import random
np.set_printoptions(threshold=np.inf)

def saveFeatures(dir,Feats,Paths):  #将特征和路径保存到HDF5文件
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

def encFeat(feat,keyF): #处理单个图像特征的加密H（2）
    MO, Gamma, Alpha,p2=keyF
    exFeat=np.concatenate((feat, [-0.5*(np.linalg.norm(feat) ** 2)],Alpha),axis=0)
    while (1):
        xi = np.random.randint(low=1, high=p2, size=len(feat)*2)
        if Gamma > 2 * abs(np.max(xi)):
            break
    EncFeat=np.dot((Gamma*exFeat+xi),MO)
    return EncFeat

def encFeats(feats,keyF):   #处理批量特征的加密
    MO, Gamma, p2 = keyF
    Alpha=np.random.randint(low=1,high=p2,size=len(feats[0])-1)
    a2 = -0.5 * (np.linalg.norm(feats, axis=1) ** 2).reshape((-1, 1))
    a3 = np.tile(Alpha, [len(feats), 1])
    exFeats = np.concatenate((feats, a2, a3), axis=1)
    EncFeats = np.dot((Gamma * exFeats), MO)
    return EncFeats

def encQuery(query,keyQ):   #处理查询的加密
    MU,Gamma,p1,p2=keyQ
    Size=len(query)
    Delta=random.randint(1,p2)
    Beta=np.random.randint(low=1,high=p2,size=Size-1)

    while(1):
        xi=np.random.randint(low=1,high=p2,size=Size*2)
        if Gamma > 2*abs(np.max(xi)):
            break

    exQuery=np.concatenate((query*Delta,[Delta],Beta))
    EncQuery=np.dot(MU,(Gamma*exQuery.T+xi.T))
    return EncQuery

def encQuerys(querys,keyQ):
    MU, Deta, Belta = keyQ
    a2 = np.random.randint(1,2,size=(len(querys),1))
    a3 = np.tile(Belta, [len(querys), 1])
    exQuerys = Deta*np.concatenate((querys, a2, a3), axis=1)
    EncQuerys = np.dot(exQuerys, MU)
    return EncQuerys