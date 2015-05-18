ANGSTROM = 1/0.52917721092 #Bohr
ANGSTROM3 = ANGSTROM**3  # a.u.

#The AU tag is only concerned with coordinates, as in dalton,
#convert all coordinates and properties to AU if the pot is in AA to make tests pass
DIATOMIC = """AA
2 0 1 1
1  0.000  0.000  %f 0.000 %f
2  0.000  0.000  %f 0.000 %f
"""

# Appelquist data

H2 = {
    "R": .7413,
    "ALPHA_H": 0.168,
    "ALPHA_ISO": 0.80,
    "ALPHA_PAR": 1.92,
    "ALPHA_ORT": 0.24,
    }
H2 = { key:(val*ANGSTROM3 if 'ALPHA' in key else val) for key, val in zip(H2.keys(),
    H2.values())}
H2["POTFILE"] = DIATOMIC % (0, H2["ALPHA_H"], H2["R"], H2["ALPHA_H"])

N2 = {
    "R": 1.0976,
    "ALPHA_N": 0.492,
    "ALPHA_ISO": 1.76,
    "ALPHA_PAR": 3.84,
    "ALPHA_ORT": 0.72,
    }
N2 = { key:(val*ANGSTROM3 if 'ALPHA' in key else val) for key, val in zip(N2.keys(),
    N2.values())}
N2["POTFILE"] = DIATOMIC % (0, N2["ALPHA_N"], N2["R"], N2["ALPHA_N"])

O2 = {
    "R": 1.2074,
    "ALPHA_O": 0.562,
    "ALPHA_ISO": 1.60,
    "ALPHA_PAR": 3.11,
    "ALPHA_ORT": 0.85,
    }
O2 = { key:(val*ANGSTROM3 if 'ALPHA' in key else val) for key, val in zip(O2.keys(),
    O2.values())}
O2["POTFILE"] = DIATOMIC % (0, O2["ALPHA_O"], O2["R"], O2["ALPHA_O"])

Cl2 = {
    "R": 1.988,
    "ALPHA_Cl": 1.934,
    "ALPHA_ISO": 4.61,
    "ALPHA_PAR": 7.62,
    "ALPHA_ORT": 3.10,
    }
Cl2 = { key:(val*ANGSTROM3 if 'ALPHA' in key else val) for key, val in zip(Cl2.keys(), Cl2.values())}
Cl2["POTFILE"] = DIATOMIC % (0, Cl2["ALPHA_Cl"], Cl2["R"], Cl2["ALPHA_Cl"])

HCl = {
    "R": 1.2745,
    "ALPHA_H": 0.059,
    "ALPHA_Cl": 2.39,
    "ALPHA_ISO": 2.63,
    "ALPHA_PAR": 3.13,
    "ALPHA_ORT": 2.39,
    }
HCl = { key:(val*ANGSTROM3 if 'ALPHA' in key else val) for key, val in zip(HCl.keys(), HCl.values())}
HCl["POTFILE"] = DIATOMIC % (0, HCl["ALPHA_H"], HCl["R"], HCl["ALPHA_Cl"])

HBr = {
    "R": 1.408,
    "ALPHA_H": 0.071,
    "ALPHA_Br": 3.31,
    "ALPHA_ISO": 3.61,
    "ALPHA_PAR": 4.22,
    "ALPHA_ORT": 3.31,
    }
HBr = { key:(val*ANGSTROM3 if 'ALPHA' in key else val) for key, val in zip(HBr.keys(), HBr.values())}
HBr["POTFILE"] = DIATOMIC % (0, HBr["ALPHA_H"], HBr["R"], HBr["ALPHA_Br"])

HI = {
    "R": 1.609,
    "ALPHA_H": 0.129,
    "ALPHA_I": 4.89,
    "ALPHA_ISO": 5.45,
    "ALPHA_PAR": 6.58,
    "ALPHA_ORT": 4.89,
    }
HI = { key:(val*ANGSTROM3 if 'ALPHA' in key else val) for key, val in zip(HI.keys(),
    HI.values())}
HI["POTFILE"] = DIATOMIC % (0, HI["ALPHA_H"], HI["R"], HI["ALPHA_I"])

CO = {
    "R": 1.1282,
    "ALPHA_C": 1.624,
    "ALPHA_O": 0.071,
    "ALPHA_ISO": 1.95,
    "ALPHA_PAR": 2.60,
    "ALPHA_ORT": 1.625,
    }
CO = { key:(val*ANGSTROM3 if 'ALPHA' in key else val) for key, val in zip(CO.keys(),
    CO.values())}
CO["POTFILE"] = DIATOMIC % (0, CO["ALPHA_C"], CO["R"], CO["ALPHA_O"])

CH4 = {
    "R": 1.095,
    "A": 109.4712206,
    "ALPHA_C": 5.925,
    "ALPHA_H": 0.911,
    "ALPHA_ISO": 17.4107029,
    }
CH4["POTFILE"] = """AA
5 0 1 1
1  1.095  0.000000  0.000000 0 %f
2 -0.365  1.032380  0.000000 0 %f
3 -0.365 -0.516188 -0.894064 0 %f
4 -0.365 -0.516188  0.894064 0 %f
5  0.000  0.000000  0.000000 0 %f
""" % (
    CH4["ALPHA_H"], CH4["ALPHA_H"], CH4["ALPHA_H"], CH4["ALPHA_H"], 
    CH4["ALPHA_C"]
    )

CH3OH = {
    "ALPHA_C": 5.9250,
    "ALPHA_O": 3.1379,
    "ALPHA_H": 0.911,
    "ALPHA_ISO": 20.5824,
    }
CH3OH["POTFILE"] = """AA
6 0 1 1
1   1.713305   0.923954   0.000000  0 0.9110
2  -0.363667  -1.036026  -0.000000  0 0.9110
3  -0.363667   0.518013  -0.897225  0 0.9110
4  -0.363667   0.518013   0.897225  0 0.9110
5   0.000000   0.000000   0.000000  0 5.9250
6   1.428000   0.000000   0.000000  0 3.137975
"""
 
                                  
C2H6 = {
    "RCH": 1.095,
    "RCC": 1.54,
    "A": 109.4712206,
    "ALPHA_C": 0.878,
    "ALPHA_H": 0.135,
    "ALPHA_ISO": 4.47,
    }
C2H6["POTFILE"] = """AU
8 0 1 1
1   -0.365000   1.032376   0.000000  0 %f
2   -0.365000  -0.516188  -0.894064  0 %f
3   -0.365000  -0.516188   0.894064  0 %f
4    1.905000  -1.032376   0.000000  0 %f
5    1.905000   0.516188   0.894064  0 %f
6    1.905000   0.516188  -0.894064  0 %f
7    0.000000   0.000000   0.000000  0 %f
8    1.540000   0.000000   0.000000  0 %f
""" % (
    C2H6["ALPHA_H"], C2H6["ALPHA_H"], C2H6["ALPHA_H"], C2H6["ALPHA_H"], C2H6["ALPHA_H"], C2H6["ALPHA_H"],  
    C2H6["ALPHA_C"], C2H6["ALPHA_C"],
    )

C3H8={
    "ALPHA_ISO": 44.40404,
    }
C3H8["POTFILE"] = """AA
11 0 1 1
1   1.905000  -1.032376   0.000000  0 0.9110
2  -1.608333   1.451926  -0.000000  0 0.9110
3  -0.365000  -0.516188   0.894064  0 0.9110
4  -0.365000  -0.516188  -0.894064  0 0.9110
5   1.905000   0.516188   0.894064  0 0.9110
6   1.905000   0.516188  -0.894064  0 0.9110
7  -0.148333   1.968114  -0.894064  0 0.9110
8  -0.148333   1.968114   0.894064  0 0.9110
9   0.000000   0.000000   0.000000  0 5.9250
10   1.540000   0.000000   0.000000  0 5.9250
11  -0.513333   1.451926   0.000000  0 5.9250
"""

CP = {
    "ALPHA_ISO": 60.735010
    }
CP["POTFILE"] = """AA
15 0 1 1
1    0.78418   -0.964861   0.271829  0 5.9250
2   -0.021977   1.280733   0.13293   0 5.9250
3   -1.250485   0.34601    0.014939  0 5.9250
4   -0.678518  -1.078375  -0.162462  0 5.9250
5    1.178045   0.411707  -0.257511  0 5.9250
6    0.09593    1.611652   1.165061  0 0.9110
7   -0.114695   2.173133  -0.483073  0 0.9110
8   -1.88227    0.621163  -0.827512  0 0.9110
9   -1.869117   0.409921   0.9084    0 0.9110
1   -0.715272  -1.366045  -1.213973  0 0.9110
1   -1.234928  -1.826854   0.398658  0 0.9110
1    1.407035  -1.773721  -0.107177  0 0.9110
1    0.853771  -0.969225   1.362129  0 0.9110
1    1.272126   0.366363  -1.344795  0 0.9110
1    2.119951   0.782327   0.14393   0 0.9110
"""

NP = {
    "ALPHA_ISO": 66.8759
    }
NP["POTFILE"] = """AA
17 0 1 1
1   1.905000   0.516188   0.894064  0 0.911
2   1.905000  -1.032376   0.000000  0 0.911
3   1.905000   0.516188  -0.894064  0 0.911
4  -1.608333   1.451926  -0.000000  0 0.911
5  -0.148333   1.968114   0.894064  0 0.911
6  -0.148333   1.968114  -0.894064  0 0.911
7  -1.608333  -0.725963  -1.257405  0 0.911
8  -0.148333  -0.209775  -2.151468  0 0.911
9  -0.148333  -1.758339  -1.257405  0 0.911
10  -0.148333  -0.209775   2.151468  0 0.911
11  -1.608333  -0.725963   1.257405  0 0.911
11  -0.148333  -1.758339   1.257405  0 0.911
13   0.000000   0.000000   0.000000  0 5.9250
14   1.540000   0.000000   0.000000  0 5.9250
15  -0.513333   1.451926   0.000000  0 5.9250
16  -0.513333  -0.725963  -1.257405  0 5.9250
17  -0.513333  -0.725963   1.257405  0 5.9250
"""

DME = { "ALPHA_ISO": 35.2263 }
DME["POTFILE"] = """AA
9 0 1 1
1  1.1668 -0.2480 0.0000 0 5.9250
2 -1.1668 -0.2480 0.0000 0 5.9250
3  0.0000  0.5431 0.0000 0 3.1379
4  2.019   0.433  0.000  0 0.911
5 -2.019   0.433  0.000  0 0.911
6  1.206  -0.888  0.8944 0 0.911
7  1.206  -0.888 -0.8944 0 0.911
8 -1.206  -0.888  0.8944 0 0.911
9 -1.206  -0.888 -0.8944 0 0.911
"""                     

# Other model systems

H2O_DIMER = """AU
2 1 1 0
1 0.00000  0.00000  0.48861 0.0 0.00000 -0.00000 -0.76539  6.61822
1 0.00000  0.00000  5.48861 0.0 0.00000 -0.00000 -0.76539  6.61822 
"""
