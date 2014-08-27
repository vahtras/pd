#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    q = 0.32
    pi = np.pi
    
    q_std = np.r_[ 0.2 : 1.0 : 10j ]
    x = np.r_ [ -2.0 : 2.0 : 1000j ]

    plt.figure(1)

    for i in range(len( q_std)) :
        y = q / ( pi**(1.5)*q_std[i] ** 3 ) * np.exp ( -x ** 2 / q_std[i] **2 )

        plt.subplot( int( "33%d"% i ) )
        plt.title('Broadening: %.1f' %q_std[i] )
        plt.ylim( [ -0.1, 0.36 ] )
        plt.plot(x, y )
    plt.show()
