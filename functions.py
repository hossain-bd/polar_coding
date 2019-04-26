# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 16:23:54 2018

@author: nazmul
"""

import numpy as np
import random

def logical_xor(str1, str2):
    return bool(str1) ^ bool(str2)

def sxor(s1,s2):    
    # convert strings to a list of character pair tuples
    # go through each tuple, converting them to ASCII code (ord)
    # perform exclusive or on the ASCII code
    # then convert the result back to ASCII (chr)
    # merge the resulting array of characters as a string
    return ''.join(chr(ord(a) ^ ord(b)) for a,b in zip(s1,s2))

def xor(a,b):
    print('a:',a)
    print('b:',b)
    print ('a in binary {}'.format(bin(a)))
    print ('b in binary {}'.format(bin(b)))
    return a^b, bin(a^b)

def sample_encode(N,k,F):
    y1 = k*k
    y2 = k*(2-k)
    return y1,y2


def log_2_N(num):
    return int(np.log2(num))

def Fn(n):
    '''
    Fn is the n-fold Kronecker product of Arikan's
    standard polarizing kernel F.
    '''

    if n == 1:
        return np.array([1, ])
    n_kronecker = log_2_N(n) - 1  # number of Kronecker products to calculate
    F_kr = np.array([[1, 0], [1, 1]],np.int)
    Fn = F_kr
    for i in range(n_kronecker):
        Fn = np.kron(Fn, F_kr)
    return Fn

def bit_reverse_1(value,bits):
    reversed = int('{:0{bits}b}'.format(value, bits=bits)[::-1],2)
    return reversed

def _reversed_ (value,numbits):
    # original is a single integer number 
    _reversed_ = sum(1<<(numbits-1-i) for i in range(numbits) if value>>i&1)
    return _reversed_

def _reversed_positions_(n):
    value = np.arange(n, dtype=int)
    numbits = int(np.log2(n))
    _reversed_positions_ = np.array([_reversed_(e, numbits) for e in value], dtype=value.dtype)
    return _reversed_positions_

#def Polar_Encoder(K,N,F,u):
#    #d = np.zeros(len(u), dtype=int)
#    d = []
#    for i, value in enumerate(u):
#        d.append(int(u[i]))
#    
#    for i, value in enumerate(F):
#        d.insert(F[i],0)
#
#    x = [0]*N
#    x[0] = ((d[0]^d[4])^d[2])^d[1]
#    x[1] = (d[1]^d[5])^d[3]
#    x[2] = (d[2]^d[6])^d[3]
#    x[3] = d[3]^d[7]
#    x[4] = (d[4]^d[6])^d[5]
#    x[5] = d[5]^d[7]
#    x[6] = d[6]^d[7]
#    x[7] = d[7]
#    
#    print('input information: ',u)
#    print('Encoded code word: ',x )
#    y=str(x)
#    print(''.join(y))

def bsc(p,N):
    '''
    Binary Symmetric Channel (BSC)
    W(0|0) = W(1|1)
    W(1|0) = W(0|1)

    It returns a probability vector
    of BSC for a given probability P
    0 <= P <= 1
    '''
    if (p >= 0.0 and p <= 1.0):
        W = np.array([[1 - p, p], [p, 1 - p]], dtype=float)
    else:
        print ("Value Error!: p is out of range!",
               "p should be [0.0 <= p <= 1.0].")
        return np.array([], dtype=float)
    C=int((1-p) * N)
    print('\n################################################################')
    print('# Binary Symmetric Channel is initialized and going to be used #')
    print('################################################################\n')
    return W,C

#frozen_pos = np.array((0, 2 ,7),np.int) #**********
#u = [1,1,1,0,1]
    
# polar_encoder(N,K,frozen_pos,u)
    
class polar_encoder():
    def __init__(self, N, K):
        '''
        Example: polar_encoder(N,K,frozen_pos,u)
        
        '''

        frozen_bits = N - K
        # random.sample(range(N),bits( < N))
        # creates a frozen bits position vector within N ranged array
        frozen_pos = random.sample(range(N), frozen_bits)
        # info_pos: creates an index vector which holds the info bits only
        info_pos = np.delete(np.arange(N), frozen_pos)
        
        self.G = Fn(N) # G = Fn, the generator matrix of polar code
        self.frozen_pos = frozen_pos
        self.info_pos = info_pos
        self.N = N
        print('\n#######################################')
        print('###### Polar Encoder initialized ######')
        print('#######################################\n')
        print('Information bits to be encoded per codeword:',K)
        print('Length of code word in bits: {} bits'.format(N))
        print('Code Rate (R = K/N):',K/N)
        print('Frozen positions:',frozen_pos)
        print('Info positions:',info_pos)
        print('\n###################################################\n')
        #self.polar_codeword()
        
        return None
        
    def polar_codeword(self,u):
 
        # d: Creates N sized empty array without data initialization
        self.d = np.empty(self.N, dtype=int)
        # d[info_pos]: matches the data of u with d by info_pos indices
        self.d[self.info_pos] = u
        # ***  x = G.u = (Fn . u) ***
        self.d = np.dot(self.d, self.G) %2
        #print('\nInformation to be encoded: {}'.format(u))
        #print('Polar encoded Codeword: {}'.format(self.d))
        # Modulus 2 is required to create binary only (0,1) after dot product
        
        return self.d
        
    def encode():
        