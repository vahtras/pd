from math import sin, cos, asin, acos, sqrt, pi

deg = lambda x : 180*x/pi
rad = lambda x : pi*x/180

zmat = """X
C 1 halfcc
O 1 ox 1 90
C 1 halfcc 3 90 2 180
H 2 ch 1 hcc 3 hcco
H 2 ch 1 hcc 3 -hcco
H 4 ch 1 hcc 3 hcco
H 4 ch 1 hcc 3 -hcco
Variables:
halfcc=%f
ox=%f
ch=%f
hcc=%f
hcco=%f"""

# given
cc = 1.470
co = 1.435
ch = 1.083
h2cc = 158*pi/180
hch = 116*pi/180


#derived
halfcc = cc/2
ox = sqrt(co**2 - halfcc**2)
#print "cc=%f co=%f ox=%f" % (cc, co, ox)
hcc = acos(cos(hch/2)*cos(h2cc))
#print "hcc", hcc*180/pi
#
#cos 2*delta =  (cos(alpha) - cos2(beta)) / sin2(beta)
#   -> 2*delta = pm acos() + n2pi
#   -> delta =  pm 1/2 acos() + n*pi
#
hacos =.5*(acos((cos(hch) - cos(hcc)**2) / sin(hcc)**2))
#print deg(hacos)
#print deg(-hacos)
#print deg(hacos+pi)
#print deg(-hacos+pi)
#
hcco = pi-hacos


print zmat % (halfcc, ox, ch, deg(hcc), deg(hcco))
