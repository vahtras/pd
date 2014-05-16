import math
import numpy
angles = numpy.array([2*math.pi/5*j for j in range(5)])
x = numpy.cos(angles)
y = numpy.sin(angles)
carbons = numpy.zeros((5, 3))
carbons[:, 0 ] = x
carbons[:, 1 ] = y
print carbons
print math.acos(numpy.dot(carbons[1], carbons[2]))*180/math.pi
print math.acos(numpy.dot(carbons[0] - carbons[1], carbons[2] - carbons[1]))*180/math.pi

