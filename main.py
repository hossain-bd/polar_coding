# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 10:30:51 2018

@author: nazmul
"""

#from IPython import get_ipython
#
#get_ipython().magic('reset -f')

import functions as polar
import numpy as np
import random
import sys


'''
Channel Initialization
'''
N = 16
(W, K)= polar.bsc(.1,N) # bsc(p,N); p=Error Probability, N=Length of codeword

#K = 5
#frozen_pos = np.array((0, 2 ,7),np.int) 
#(W, K)= polar.bsc(.3,N)
#frozen_bits = N - K
#frozen_pos = random.sample(range(N), frozen_bits)

'''
Random binary Input
'''
u = []
for i in range(K):
    u.append(int(random.randint(0,1)))

if (len(u) != K):
    sys.exit("Error! Channel capacity is lower than information (u) to be encoded")
else:
    u=u

'''
Manual Character Input
''' 
info = 'Hossain'
e=[]
for i, value in enumerate(info):
    e.append(bin(ord(info[i]))[2:])

u_char = []
for i in range(len(e)):
    u_char.append ([int(x) for x in str(e[0])])
    
if (len(u_char[0]) > K):
    print('Channel capacity:',K)
    print('Required Capacity:',len(u_char))
    sys.exit("Error! Channel capacity is lower than information (u_char) to be encoded")
elif (len(u_char[0]) < K):
    K1 = K
    K = len(u_char)
    #sys.exit("Error! Channel capacity is Higher than information to be encoded")
else:
    u_char=u_char

print('\nGiven information:',info)

#print('W =',W)
print('Actual Channel Capacity without noise =',K1)
print('Noiseless Capacity to be used =',K)
print('Unused Noiseless Capacity =',K1-K)
print('Noisy channel parts to be ignored =',N-K1)

print('\n############################################################')
print('Information bits to be encoded:\n',u_char)
print('############################################################')
      
p1=polar.polar_encoder(N,K)

d=[]
for i in range(len(u_char)):
    d.append(p1.polar_codeword(u_char))
    print('\nInformation to be encoded ({0}): {1}'.format(i,u_char[i]))
    print('Polar encoded Codeword({0}): {1}'.format(i,d[i]))