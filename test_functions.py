# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 10:10:09 2018

@author: nazmul
"""

import nympy as np

def log_2_N(num):
    return int(np.log2(num))

def Fn(n):
    '''
    Fn is the n-fold Kronecker product of Arikan's
    standard polarizing kernel F.
    '''
    # this matrix defines the actual channel combining.
    if n == 1:
        return np.array([1, ])
    n_kronecker = log_2_N(n) - 1  # number of Kronecker products to calculate
    F_kr = np.array([[1, 0], [1, 1]],np.int)
    Fn = F_kr
    for i in range(n_kronecker):
        Fn = np.kron(Fn, F_kr)
    return Fn

def polar_encoder(N, K, frozen_pos, u):
        print('###### Polar Encoder initialized ######')
        print('Information bits to be encoded per codeword:',K)
        print('Length of code word in bits:',N,'bits')
        
def polar_codeword(self):
    # info_pos: creates an index vector which holds the info bits only
    info_pos = np.delete(np.arange(self.N), self.frozen_pos) 
    # d: Creates N sized empty array without data initialization
    d = np.empty(self.N, dtype=int)
    # d[info_pos]: matches the data of u with d by info_pos indices
    d[info_pos] = self.u
    
    # x = G.u = (Fn . u) 
    d = np.dot((d, self.G)) 
    #d = self.G
    #self.d = self.d.astype(dtype=int)
    # Modulus 2 is required to create binary only (0,1) after dot product
    return d