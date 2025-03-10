import numpy as np
import os
import time



def getparameter(size,cla):
    Range=1000
    U=np.random.randint(1,Range,(size,size))
    V=np.random.randint(1,Range,(size,size))
    A=np.random.randint(1,Range,(size,2*size))
    M=np.random.randint(1,Range,(2*size,2*size))
    B=np.zeros((2*size,size))
    for i in range(2*size):
        for j in range(size):
            B[i][j]=M[i][2*j+1]
    keyF=[M,U,A]
    keyQ=[M,V,B]
    KeyF=np.array(keyF,dtype=object)
    KeyQ=np.array(keyQ,dtype=object)
    if os.path.exists('./'+str(cla)+'/Key/FeatH2.npy'):
        os.remove('./'+str(cla)+'/Key/FeatH2.npy')
    np.save('./'+str(cla)+'/Key/FeatH2.npy',KeyF)

    if os.path.exists('./'+str(cla)+'/Key/QueryH2.npy'):
        os.remove('./'+str(cla)+'/Key/QueryH2.npy')
    np.save('./'+str(cla)+'/Key/QueryH2.npy',KeyQ)


if __name__=='__main__':
    ClassNum=[50,100,150,200]
    for cla in ClassNum:
        stime = time.time()
        getparameter(8,cla)
        etime = time.time()
        print(round((etime - stime)*1000, 4))


