# -*- coding: utf-8 -*-
from flow import *

if __name__ =='__main__':
    t=1
    for i in range(1,26):
        for j in range(i+1,27):
            t+=1
            matrix = np.loadtxt('dataset/UCI/letter/'+str(i)+'.txt', delimiter=' ')
            matrix1 = np.loadtxt('dataset/UCI/letter/'+str(j)+'.txt', delimiter=' ')
            matrix=np.vstack((matrix,matrix1))
            np.savetxt( 'dataset/UCI/temp.txt', matrix, fmt='%.1f')
            print(i,j,t)
            runCBLP(fileName='dataset/' + fileList['temp'], percent=0.1)

