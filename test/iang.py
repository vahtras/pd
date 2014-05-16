import sys
from math import *
ang = float(sys.argv[1])
angr = ang*pi/180
thetar  = pi-asin(sqrt(2*(1-cos(angr))/3))
theta = thetar*180/pi
print theta
