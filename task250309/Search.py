import numpy as np
import h5py
import time
import EncH1
import EncH2
import BuildIndex
import math
import os
import pickle

class Node(object):
    def __init__(self):
        self.value=None
        self.childlist=[]
        self.leafchild=[]
        self.leafpath=[]
        self.level=None
        self.father=None
        self.IND=[]
        self.Size=0

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

KKK=0
def _FindRoot(Tree, qFeat, root,keyQ):
    MU,Gamma,p1,p2=keyQ
    global KKK
    if KKK < 0:
        return []
    res = []
    if len(Tree[root].childlist) == 0:
        res.append(root)
        KKK -= Tree[root].Size
        return res
    Tree[root].childlist = sorted(Tree[root].childlist, key=lambda x: (np.dot(Tree[x].value, qFeat)/(Gamma**2))%p1,reverse=True)
    for i in Tree[root].childlist:
        if KKK <= 0:
            break
        res += _FindRoot(Tree, qFeat, i,keyQ)
    return res

def Search(EncqH1,EncqH2,qpath,Tree,Nodenum,root,K,KeyQueryH1):
    global KKK
    KKK = K
    root = _FindRoot(Tree, EncqH1, root,KeyQueryH1)
    # print(int(os.path.split(os.path.split(qpath.decode())[0])[1]),root)
    Encfeats=Tree[root[0]].leaffeature
    Enclabels=Tree[root[0]].leafpath
    print(Encfeats.shape)
    for i in range(1,len(root)):
        Encfeats=np.concatenate((Encfeats,Tree[root[i]].leaffeature),axis=0)
        Enclabels=np.concatenate((Enclabels,Tree[root[i]].leafpath),axis=0)
    Encfeats=np.array(Encfeats)
    Enclabels=np.array(Enclabels)

    a=np.dot(Encfeats,EncqH2.T)
    b=np.diagonal(a,axis1=1,axis2=2)
    c=np.sum(b,axis=1)
    indexList=np.argsort(-c)
    num=0
    for ind in indexList[0:K]:
        if os.path.split(os.path.split(Enclabels[ind])[0])[1]==os.path.split(os.path.split(qpath)[0])[1]:
            num+=1

    return num/K

if __name__=='__main__':
    Accuracy=[]
    SearchTime=[]
    GenTrapdoorTime=[]
    ClassNum=[50,100,150,200]
    LengthH1=[24,32,48,64]
    for Cla in range(len(ClassNum)):
        KeyFeatH1 = np.load('./' + str(ClassNum[Cla]) + '/Key/FeatH1.npy',allow_pickle=True)
        KeyFeatH2 = np.load('./' + str(ClassNum[Cla]) + '/Key/FeatH2.npy',allow_pickle=True)
        KeyQueryH1 = np.load('./' + str(ClassNum[Cla]) + '/Key/QueryH1.npy',allow_pickle=True)
        KeyQueryH2 = np.load('./' + str(ClassNum[Cla]) + '/Key/QueryH2.npy',allow_pickle=True)
        p='./' + str(ClassNum[Cla]) + '/Features/test.h5'
        qfeats,qpaths=loadFeatures(p)
        Len=LengthH1[Cla]
        size = int(math.sqrt(len(qfeats[0]) - Len))
        qH1=[]
        qH2=[]
        stime=time.time()
        Centernum=ClassNum[Cla]
        for i in range(len(qfeats)):
            qH1.append(qfeats[i][0:Len])
            qH2.append(qfeats[i][Len:])
        EncqH1=[]
        EncqH2=[]
        for i in range(len(qH1)):
            EncqH1.append(EncH1.encQuery(qH1[i],KeyQueryH1))
            EncqH2.append(EncH2.EncQuery(qH2[i],KeyQueryH2,size))
        etime=time.time()
        GenTrapdoorTime.append(round((etime-stime)*1000/len(qfeats),4))
        Tree = []
        with open('./' + str(ClassNum[Cla]) + '/EncTree.pkl', 'rb') as f:
            while True:
                try:
                    Tree.append(pickle.load(f))
                except EOFError:
                    break
        node_num = len(Tree)
        print(node_num)
        #find root node
        root=-1
        for i in range(node_num):
            if Tree[i].father is None:
                root=i
                break

        KList=[5,10,20,30,40]
        Acc=[]
        Time=[]
        for K in KList:
            Rate=0
            Starttime=time.time()
            for i in range(len(qfeats)):
                rate=Search(EncqH1[i],EncqH2[i],qpaths[i],Tree,node_num,root,K,KeyQueryH1)
                Rate+=rate
            Endtime=time.time()
            Acc.append(round(Rate/len(qfeats),4))
            Time.append(round((Endtime-Starttime)*1000/len(qfeats),4))
        Accuracy.append(Acc)
        SearchTime.append(Time)
    np.save('./Result/Accuracy.npy',Accuracy)
    np.save('./Result/SearchTime.npy',SearchTime)
    np.save('./Result/GenTrapdoorTime.npy',GenTrapdoorTime)
    print("====================Accuracy====================")
    print(Accuracy)
    print("===================SearchTime===================")
    print(SearchTime)
    print("===================GenTrapdoorTime===================")
    print(GenTrapdoorTime)

