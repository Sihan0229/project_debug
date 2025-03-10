import math
import h5py
import numpy as np


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

def EncFeat(feat,keyF,size):
    M,U,A=keyF
    M_=np.linalg.inv(M)
    feat = np.linalg.norm(feat)
    feat = feat.reshape(size,size)
    ExpFeat=np.zeros((size,2*size))
    for i in range(size):
        ExpFeat[:,2*i]=feat[:,i]
        ExpFeat[:,2*i+1]=U[:,i]
    encfeat=np.dot(ExpFeat,M_)+A
    return encfeat

def EncAllFeats(feats,keyF,size):
    M,U,A=keyF
    M_=np.linalg.inv(M)
    feats = feats * np.linalg.norm(feats, axis = 1, keepdims=True)
    Feats = []
    for i in range(len(feats)):
        Feats.append(feats[i].reshape(size,size))
    Feats = np.array(Feats)
    a2=np.tile(U,(len(Feats),1,1))
    ExpFeat=np.zeros((len(Feats),size,2*size))
    for i in range(size):
        ExpFeat[:,:,2*i]=Feats[:,:,i]
        ExpFeat[:,:,2*i+1]=a2[:,:,i]
    encfeats=np.dot(ExpFeat,M_)+A
    return encfeats

def EncQuery(Query,keyQ,size):
    M,A,B=keyQ
    Query = Query * np.linalg.norm(Query)
    Query = Query.reshape(size,size)
    ExpQuery = np.zeros((size, 2*size))
    V = np.random.randint(1, 1000, (size, size))
    for i in range(size):
        ExpQuery[:,2*i] = Query[:,i]
        ExpQuery[:,2*i+1] = V[:,i]
    encquery = np.dot(ExpQuery,M.T)+B.T
    return encquery
