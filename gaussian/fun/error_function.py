#!/usr/bin/env python
#-*- coding: utf-8 -*-

import math
import numpy as np

class ErrorFunction(object):

    """
    Container class for the error function"""
    def __init__(self, *args, **kwargs ):
        pass


    def gauss_dist():
        """ Plot the gaussian distribution given"""
        pass


if __name__ == '__main__':
    # R_qq is the standard deviation of the charge gaussian distribution
    # 

    r = np.c_[ 1.0 : 5.0 : 10j ]
    R_qq = np.c_[ 0.1 : 1 : 10j ]

    for dist in r:
        for dev in R_qq:
            val = math.erf( dist/ dev )

            if val < 0.95 or val > 1.05:
                print dist, dev, val
