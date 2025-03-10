import math
import time
import os
import numpy as np
import h5py
from sklearn.cluster import KMeans
import EncH2
import EncH1
import pickle
# 用于保存和加载特征数据到HDF5文件，处理特征提取后的数据存储。
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
# 计算文件大小，用于评估存储开销。
def getFileSize(filePath):
    fsize = os.path.getsize(filePath.encode())
    fsize = fsize/float(1024 * 1024)
    return round(fsize, 8)
# 定义了树节点的结构，包含值、子节点列表、叶子节点、特征、路径等信息，这与层次索引树有关。
class Node(object):
    def __init__(self):
        self.value=None
        self.childlist=[]
        self.leafchild=[]
        self.leaffeature=[]
        self.leafpath=[]
        self.level=None
        self.father=None
        self.IND=[]
        self.Size=0

# 用于计算节点之间的距离，在构建树结构时用于合并节点。
def Findnearest(Tree,node):
    x=0
    y=0
    mini=1e18  #记录当前找到的最小距离
    for i in node:
        for j in node:
            if i!=j:    # 避免节点与自身进行比较，因为节点到自身的距离为 0，这不是我们想要的结果
                dis=np.linalg.norm(Tree[i].value-Tree[j].value) # 它计算的是Tree[i].value和Tree[j].value之间的欧几里得距离。
                if dis < mini:  # 更新最小距离
                    x=i
                    y=j
                    mini=dis
    return x,y

def Calmaxvalue(ANode,childnode):
    Maxn=-1e18  #记录当前找到的最大距离
    for i in childnode:
        for j in childnode:
            if i!=j:
                dis=np.linalg.norm(ANode[i].value-ANode[j].value)
                if dis > Maxn:
                    Maxn = dis
    return Maxn
# 处理节点的合并，根据标志位决定不同的合并方式
def merge(Tree,node,x,y,flag,nodenum):
    if flag==1:
        b=Node()
        b.childlist.append(x)
        b.childlist.append(y)
        if Tree[x].level != 0 and Tree[y].level != 0:   # 层级都不为 0，则将它们的leafchild列表合并
            b.leafchild = Tree[x].leafchild + Tree[y].leafchild
        else:   # 否则，将层级为 0 的节点直接添加到b的leafchild列表中
            if Tree[x].level == 0:
                b.leafchild.append(x)
            else:
                b.leafchild = b.leafchild + Tree[x].leafchild
            if Tree[y].level == 0:
                b.leafchild.append(y)
            else:
                b.leafchild = b.leafchild + Tree[y].leafchild

        b.level = max(Tree[x].level, Tree[y].level) + 1 # 更新新节点b的层级为Tree[x]和Tree[y]的最大层级加 1
        b.value = (Tree[x].value + Tree[y].value) / 2.0 #计算新节点b的值为Tree[x]和Tree[y]的值的平均值
        b.IND = Calmaxvalue(Tree, b.leafchild)  # 调用Calmaxvalue函数计算新节点b的IND值
        b.Size = Tree[x].Size + Tree[y].Size    # 更新新节点b的大小为Tree[x]和Tree[y]的大小之和
        Tree.append(b)
        Tree[x].father=nodenum
        Tree[y].father=nodenum
        node.remove(x)
        node.remove(y)
        node.append(nodenum)
    else:
        Tree[x].childlist.append(y)
        if Tree[y].level == 0:
            Tree[x].leafchild.append(y)
        else:
            Tree[x].leafchild = Tree[x].leafchild + Tree[y].leafchild
        Tree[x].IND = Calmaxvalue(Tree, Tree[x].leafchild)
        t = Tree[x].childlist[0]
        ans = Tree[t].value
        for i in Tree[x].childlist[1:len(Tree[x].childlist)]:
            ans = ans + Tree[i].value
        Tree[x].value = ans / len(Tree[x].childlist)
        Tree[x].Size += Tree[y].Size
        Tree[y].father = x
        node.remove(y)
    return Tree,node
# 使用KMeans对特征进行聚类，分成不同的中心
def Cluster(file,lenH1,Center_num):
    p='./'+file+'/Features/train.h5'
    feats,paths=loadFeatures(p)
    Init=np.load('./'+file+'/'+str(lenH1)+'HashCode.npy')
    km = KMeans(n_clusters=Center_num, init=Init,n_init=1)
    H1=[]
    # 从 feats 中提取前 lenH1 个特征，存储在列表 H1 中。
    for i in range(len(feats)):
        H1.append(feats[i][0:lenH1])
    # 使用 fit_predict 方法对 H1 中的数据进行聚类，并返回每个数据点所属的聚类标签 y1，
    # 同时获取聚类中心 centers。
    y1=km.fit_predict(H1)
    centers=km.cluster_centers_

    # 初始化四个空列表 H1、H2、Path 和 Size，
    # 然后遍历每个聚类中心，将属于该聚类中心的数据分别存储在 h1、h2 和 path 中，
    # 最后将这些列表添加到对应的 H1、H2 和 Path 列表中，并记录每个聚类中心的数据数量。
    H1=[]
    H2=[]
    Path=[]
    Size=[]
    # print(feats.shape)
    for i in range(Center_num):
        h1=[]
        h2=[]
        path=[]
        for j in range(len(feats)):
            if y1[j]==i:
                h1.append(feats[j][0:lenH1])
                h2.append(feats[j][lenH1:])
                path.append(paths[j])
        # print(len(h1),end="\t")
        H1.append(h1)
        H2.append(h2)
        Path.append(path)
        Size.append(len(h1))
    # print("\n")
    return centers,H1,H2,Path,Size
# 构建树结构，首先调用Cluster获取聚类中心，
# 然后通过循环合并节点，直到只剩一个根节点。
# 这里根据不同的中心数量设置阈值Tht，用于控制合并的条件。
def BTree(file,lenH1,Center_num):
    centers,H1,H2,Path,Size=Cluster(file,lenH1,Center_num)
    Tree=[]
    node=[]
    for i in range(Center_num):
        b=Node()
        b.value=centers[i]
        b.level=0
        b.IND=0
        b.leaffeature=H2[i] #图像特征、叶节点
        b.leafpath=Path[i]
        b.Size=Size[i]
        node.append(i)
        Tree.append(b)
    Tht=0
    if Center_num==50:
        Tht=0.40
    elif Center_num==100:
        Tht=0.45
    elif Center_num==150:
        Tht=0.445
    else:
        Tht=0.475
    nodenum=Center_num
    while(len(node)>1):
        x,y=Findnearest(Tree,node)  #找到 node 列表中距离最近的两个节点 x 和 y
        if Tree[x].level == Tree[y].level:  #如果 Tree[x] 和 Tree[y] 的 level 相等，则调用 merge 函数进行合并，flag 设为 1，并将 nodenum 加 1
            Tree,node=merge(Tree,node,x,y,1,nodenum)    #flag = 1
            nodenum+=1
        else:   #如果 Tree[x] 和 Tree[y] 的 level 不相等，则计算 Maxn、dis 和 DM
            Maxn=max(Tree[x].IND,Tree[y].IND)
            dis=np.linalg.norm(Tree[x].value-Tree[y].value)
            DM=abs(dis-Maxn) / Maxn
            if DM > Tht:    #如果 DM 大于 Tht，则调用 merge 函数进行合并，flag 设为 1，并将 nodenum 加 1；
                Tree,node=merge(Tree,node,x,y,1,nodenum)
                nodenum+=1
            else:   #否则，交换 x 和 y 的值（如果 Tree[x].level < Tree[y].level），然后调用 merge 函数进行合并，flag 设为 0。
                if Tree[x].level < Tree[y].level:
                    temp=x
                    x=y
                    y=temp
                Tree,node=merge(Tree,node,x,y,0,nodenum)

    return Tree,nodenum



if __name__=='__main__':
    ClassList=[50,100,150,200]
    lenH1=[24,32,48,64]
    GenIndexTime=[]
    for i in range(len(ClassList)):
        stime=time.time()
        # 加载密钥
        KeyFeatH1 = np.load('./' + str(ClassList[i]) + '/Key/FeatH1.npy', allow_pickle=True)
        KeyFeatH2 = np.load('./' + str(ClassList[i]) + '/Key/FeatH2.npy', allow_pickle=True)
        Tree,nodenum=BTree(str(ClassList[i]),lenH1[i],ClassList[i])
        for j in range(nodenum):    #构建加密树
            Tree[j].value = EncH1.encFeat(Tree[j].value, KeyFeatH1) #节点加密：基于LWE的kNN安全算法
        for j in range(ClassList[i]):
            Tree[j].leaffeature = EncH2.EncAllFeats(np.array(Tree[j].leaffeature), KeyFeatH2, 8)    #叶子特征加密：随机矩阵

        with open('./' + str(ClassList[i]) + '/EncTree.pkl', 'wb') as f:
            for j in range(len(Tree)):
                pickle.dump(Tree[j], f)
        etime = time.time() # 记录不同类别数下的索引生成时间，索引构建时间对比
        GenIndexTime.append(round(etime-stime,4))

    np.save('./Result/GenIndexTime.npy',GenIndexTime)
    print(GenIndexTime)
    # 计算存储空间，计算加密索引文件大小，存储成本分析。
    GenIndexStorage=[]
    for i in range(len(ClassList)):
        file='./'+str(ClassList[i])+'/EncTree.pkl'
        GenIndexStorage.append(round(getFileSize(file),4))
    np.save('./Result/GenIndexStorage.npy',GenIndexStorage)
    print(GenIndexStorage)

