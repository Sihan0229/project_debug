import numpy as np
import os
import random
np.set_printoptions(threshold=np.inf)
import time
import math

def isPrime(n):
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def genParameter(d,cla):
    MO = np.random.rand(2 * d, 2 *d)
    MU = np.linalg.inv(MO)

    p1=random.getrandbits(32)
    p2=random.getrandbits(8)
    while (1):
        if not isPrime(p1):
            p1 += -1
        else:
            break

    while (1):
        if not isPrime(p2):
            p2 += -1
        else:
            break

    Alpha=np.random.randint(low=1,high=p2,size=d-1)
    while(1):
        Gamma=random.randint(1,p1)
        xi1=np.random.randint(low=1,high=p2,size=2*d)
        if Gamma>2*abs(np.max(xi1)):
            break
    keyF = [MO, Gamma, Alpha,p2]
    keyQ = [MU, Gamma, p1, p2]

    if os.path.exists('./'+str(cla)+'/Key/FeatH1.npy'):
        os.remove('./'+str(cla)+'/Key/FeatH1.npy')
    np.save('./'+str(cla)+'/Key/FeatH1.npy', keyF)

    if os.path.exists('./'+str(cla)+'/Key/QueryH1.npy'):
        os.remove('./'+str(cla)+'/Key/QueryH1.npy')
    np.save('./'+str(cla)+'/Key/QueryH1.npy', keyQ)





if __name__ == '__main__':
    Dimension=[24,32,48,64]
    ClassNum=[50,100,150,200]
    GenKeyH1Time=[]
    for i in range(len(Dimension)):
        stime = time.time()
        genParameter(Dimension[i],ClassNum[i])
        etime = time.time()
        print(etime - stime)

