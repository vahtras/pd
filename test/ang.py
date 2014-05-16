import sys
from math import *
theta = float(sys.argv[1])
thetar = theta*pi/180
angr  = acos(-0.5*sin(thetar)**2 + cos(thetar)**2)
ang = angr*180/pi
print ang
