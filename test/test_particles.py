import unittest
import numpy as np
from ..particles import PointDipole, PointDipoleList
from .util import iterize

DECIMALS = 1
ANGSTROM = 1.88971616463 #Bohr
ANGSTROM3 = 1/1.48184712e-1 # a.u.
DIATOMIC = """AU
2 0 1 1
1  0.000  0.000  %f 0.000 %f
1  0.000  0.000  %f 0.000 %f
"""

# Appelquist data

H2 = {
    "R": .7413,
    "ALPHA_H": 0.168,
    "ALPHA_ISO": 0.80,
    "ALPHA_PAR": 1.92,
    "ALPHA_ORT": 0.24,
    }
H2["POTFILE"] = DIATOMIC % (0, H2["ALPHA_H"], H2["R"], H2["ALPHA_H"])

N2 = {
    "R": 1.0976,
    "ALPHA_N": 0.492,
    "ALPHA_ISO": 1.76,
    "ALPHA_PAR": 3.84,
    "ALPHA_ORT": 0.72,
    }
N2["POTFILE"] = DIATOMIC % (0, N2["ALPHA_N"], N2["R"], N2["ALPHA_N"])

O2 = {
    "R": 1.2074,
    "ALPHA_O": 0.562,
    "ALPHA_ISO": 1.60,
    "ALPHA_PAR": 3.11,
    "ALPHA_ORT": 0.85,
    }
O2["POTFILE"] = DIATOMIC % (0, O2["ALPHA_O"], O2["R"], O2["ALPHA_O"])

Cl2 = {
    "R": 1.2074,
    "ALPHA_Cl": 0.562,
    "ALPHA_ISO": 1.60,
    "ALPHA_PAR": 3.11,
    "ALPHA_ORT": 0.85,
    }
Cl2["POTFILE"] = DIATOMIC % (0, Cl2["ALPHA_Cl"], Cl2["R"], Cl2["ALPHA_Cl"])

HCl = {
    "R": 1.2745,
    "ALPHA_H": 2.39,
    "ALPHA_Cl": 0.059,
    "ALPHA_ISO": 2.63,
    "ALPHA_PAR": 3.13,
    "ALPHA_ORT": 2.39,
    }
HCl["POTFILE"] = DIATOMIC % (0, HCl["ALPHA_H"], HCl["R"], HCl["ALPHA_Cl"])

HBr = {
    "R": 1.408,
    "ALPHA_H": 3.31,
    "ALPHA_Br": 0.071,
    "ALPHA_ISO": 3.61,
    "ALPHA_PAR": 4.22,
    "ALPHA_ORT": 3.31,
    }
HBr["POTFILE"] = DIATOMIC % (0, HBr["ALPHA_H"], HBr["R"], HBr["ALPHA_Br"])

HI = {
    "R": 1.609,
    "ALPHA_H": 4.89,
    "ALPHA_I": 0.129,
    "ALPHA_ISO": 5.45,
    "ALPHA_PAR": 6.58,
    "ALPHA_ORT": 4.89,
    }
HI["POTFILE"] = DIATOMIC % (0, HI["ALPHA_H"], HI["R"], HI["ALPHA_I"])

CO = {
    "R": 1.1282,
    "ALPHA_C": 1.624,
    "ALPHA_O": 0.071,
    "ALPHA_ISO": 1.95,
    "ALPHA_PAR": 2.60,
    "ALPHA_ORT": 1.625,
    }
CO["POTFILE"] = DIATOMIC % (0, CO["ALPHA_C"], CO["R"], CO["ALPHA_O"])

CH4 = {
    "R": 1.095,
    "A": 109.4712206,
    "ALPHA_C": 0.878,
    "ALPHA_H": 0.135,
    "ALPHA_ISO": 2.58,
    }
CH4["POTFILE"] = """AA
5 0 1 1
1  1.095  0.000000  0.000000 0 %f
1 -0.365  1.032380  0.000000 0 %f
1 -0.365 -0.516188 -0.894064 0 %f
1 -0.365 -0.516188  0.894064 0 %f
1  0.000  0.000000  0.000000 0 %f
""" % (
    CH4["ALPHA_H"], CH4["ALPHA_H"], CH4["ALPHA_H"], CH4["ALPHA_H"], 
    CH4["ALPHA_C"]
    )

CH3OH = {
    "ALPHA_C": 0.878,
    "ALPHA_O": 0.465,
    "ALPHA_H": 0.135,
    "ALPHA_ISO": 3.05,
    }
CH3OH["POTFILE"] = """AA
6 0 1 1
1   1.713305   0.923954   0.000000  0 0.135
1  -0.363667  -1.036026  -0.000000  0 0.135
1  -0.363667   0.518013  -0.897225  0 0.135
1  -0.363667   0.518013   0.897225  0 0.135
1   0.000000   0.000000   0.000000  0 0.878
1   1.428000   0.000000   0.000000  0 0.465
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
1   -0.365000  -0.516188  -0.894064  0 %f
1   -0.365000  -0.516188   0.894064  0 %f
1    1.905000  -1.032376   0.000000  0 %f
1    1.905000   0.516188   0.894064  0 %f
1    1.905000   0.516188  -0.894064  0 %f
1    0.000000   0.000000   0.000000  0 %f
1    1.540000   0.000000   0.000000  0 %f
""" % (
    C2H6["ALPHA_H"], C2H6["ALPHA_H"], C2H6["ALPHA_H"], C2H6["ALPHA_H"], C2H6["ALPHA_H"], C2H6["ALPHA_H"],  
    C2H6["ALPHA_C"], C2H6["ALPHA_C"],
    )

C3H8={
    "ALPHA_ISO": 6.58,
    }
C3H8["POTFILE"] = """AA
11 0 1 1
1   1.905000  -1.032376   0.000000  0 0.135
1  -1.608333   1.451926  -0.000000  0 0.135
1  -0.365000  -0.516188   0.894064  0 0.135
1  -0.365000  -0.516188  -0.894064  0 0.135
1   1.905000   0.516188   0.894064  0 0.135
1   1.905000   0.516188  -0.894064  0 0.135
1  -0.148333   1.968114  -0.894064  0 0.135
1  -0.148333   1.968114   0.894064  0 0.135
1   0.000000   0.000000   0.000000  0 0.878
1   1.540000   0.000000   0.000000  0 0.878
1  -0.513333   1.451926   0.000000  0 0.878
"""

CP = {
    "ALPHA_ISO": 9.00
    }
CP["POTFILE"] = """AA
15 0 1 1
1    0.78418   -0.964861   0.271829  0 0.878
2   -0.021977   1.280733   0.13293   0 0.878
3   -1.250485   0.34601    0.014939  0 0.878
4   -0.678518  -1.078375  -0.162462  0 0.878
5    1.178045   0.411707  -0.257511  0 0.878
6    0.09593    1.611652   1.165061  0 0.135
7   -0.114695   2.173133  -0.483073  0 0.135
8   -1.88227    0.621163  -0.827512  0 0.135
9   -1.869117   0.409921   0.9084    0 0.135
1   -0.715272  -1.366045  -1.213973  0 0.135
1   -1.234928  -1.826854   0.398658  0 0.135
1    1.407035  -1.773721  -0.107177  0 0.135
1    0.853771  -0.969225   1.362129  0 0.135
1    1.272126   0.366363  -1.344795  0 0.135
1    2.119951   0.782327   0.14393   0 0.135
"""

NP = {
    "ALPHA_ISO": 9.91,
    }
NP["POTFILE"] = """AA
17 0 1 1
1   1.905000   0.516188   0.894064  0 0.135
1   1.905000  -1.032376   0.000000  0 0.135
1   1.905000   0.516188  -0.894064  0 0.135
1  -1.608333   1.451926  -0.000000  0 0.135
1  -0.148333   1.968114   0.894064  0 0.135
1  -0.148333   1.968114  -0.894064  0 0.135
1  -1.608333  -0.725963  -1.257405  0 0.135
1  -0.148333  -0.209775  -2.151468  0 0.135
1  -0.148333  -1.758339  -1.257405  0 0.135
1  -0.148333  -0.209775   2.151468  0 0.135
1  -1.608333  -0.725963   1.257405  0 0.135
1  -0.148333  -1.758339   1.257405  0 0.135
1   0.000000   0.000000   0.000000  0 0.878
1   1.540000   0.000000   0.000000  0 0.878
1  -0.513333   1.451926   0.000000  0 0.878
1  -0.513333  -0.725963  -1.257405  0 0.878
1  -0.513333  -0.725963   1.257405  0 0.878
"""

DME = { "ALPHA_ISO": 5.22}
DME["POTFILE"] = """AA
9 0 1 1
1  1.1668 -0.2480 0.0000 0 0.878
1 -1.1668 -0.2480 0.0000 0 0.878
1  0.0000  0.5431 0.0000 0 0.465
1  2.019   0.433  0.000  0 0.135
1 -2.019   0.433  0.000  0 0.135
1  1.206  -0.888  0.8944 0 0.135
1  1.206  -0.888 -0.8944 0 0.135
1 -1.206  -0.888  0.8944 0 0.135
1 -1.206  -0.888 -0.8944 0 0.135
"""                     
        
class PointDipoleListTest(unittest.TestCase):
    def setUp(self):
        # mimic file object
        pf = iterize("""AU
3 1 0 1
1  0.000  0.000  0.698 -0.703 -0.000 0.000 -0.284 4.230
1 -1.481  0.000 -0.349  0.352  0.153 0.000  0.127 1.089
1  1.481  0.000 -0.349  0.352 -0.153 0.000  0.127 1.089
2  0.000  0.000  0.000  0.000  0.000 0.000  0.000 0.000"""
        )
        self.pdl = PointDipoleList(pf)

    def test_number(self):
        self.assertEqual(len(self.pdl), 3)

    def test_charge(self):
        #round-off error in this example
        self.assertAlmostEqual(self.pdl.charge(), .001) 

    def test_dipole_tensor_zero(self):
        Tij = self.pdl.dipole_tensor()
        zeromat = np.zeros((3,3))
        self.assertTrue (np.allclose(Tij[0, :, 0, :], zeromat))
        self.assertTrue (np.allclose(Tij[1, :, 1, :], zeromat))
        self.assertTrue (np.allclose(Tij[2, :, 2, :], zeromat))
        self.assertFalse(np.allclose(Tij[0, :, 1, :], zeromat))
        self.assertFalse(np.allclose(Tij[0, :, 2, :], zeromat))
        self.assertFalse(np.allclose(Tij[1, :, 2, :], zeromat))

    def test_dipole_tensor_values_01(self):
        x = 1.481
        y = 0
        z = 0.698 + 0.349
        r_5 = (x*x + y*y + z*z)**2.5
        T01 = self.pdl.dipole_tensor()[0, :, 1, :]
        T01xx =  (2*x*x - y*y - z*z) / r_5
        self.assertAlmostEqual(T01xx, T01[0,0])
        T01xy =  0.0
        self.assertAlmostEqual(T01xy, T01[0,1])
        T01xz =  3*x*z  / r_5
        self.assertAlmostEqual(T01xz, T01[0,2])
        T01yy = (2*y*y - x*x - z*z) / r_5
        self.assertAlmostEqual(T01yy, T01[1,1])
        T01yz =  0.0
        self.assertAlmostEqual(T01yz, T01[1,2])
        T01zz = (2*z*z - x*x - y*y) / r_5
        self.assertAlmostEqual(T01zz, T01[2,2])

    def test_dipole_tensor_values_02(self):
        x = -1.481
        y = 0
        z = 0.698 + 0.349
        r_5 = (x*x + y*y + z*z)**2.5
        T02 = self.pdl.dipole_tensor()[0, :, 2, :]
        T02xx =  (2*x*x - y*y - z*z) / r_5
        self.assertAlmostEqual(T02xx, T02[0,0])
        T02xy =  0.0
        self.assertAlmostEqual(T02xy, T02[0,1])
        T02xz =  3*x*z  / r_5
        self.assertAlmostEqual(T02xz, T02[0,2])
        T02yy = (2*y*y - x*x - z*z) / r_5
        self.assertAlmostEqual(T02yy, T02[1,1])
        T02yz =  0.0
        self.assertAlmostEqual(T02yz, T02[1,2])
        T02zz = (2*z*z - x*x - y*y) / r_5
        self.assertAlmostEqual(T02zz, T02[2,2])

    def test_dipole_tensor_values_12(self):
        x = - 2.962
        y = 0
        z = 0
        r_5 = (x*x + y*y + z*z)**2.5
        T12 = self.pdl.dipole_tensor()[1, :, 2, :]
        T12xx =  (2*x*x - y*y - z*z) / r_5
        self.assertAlmostEqual(T12xx, T12[0,0])
        T12xy =  0.0
        self.assertAlmostEqual(T12xy, T12[0,1])
        T12xz =  3*x*z  / r_5
        self.assertAlmostEqual(T12xz, T12[0,2])
        T12yy = (2*y*y - x*x - z*z) / r_5
        self.assertAlmostEqual(T12yy, T12[1,1])
        T12yz =  0.0
        self.assertAlmostEqual(T12yz, T12[1,2])
        T12zz = (2*z*z - x*x - y*y) / r_5
        self.assertAlmostEqual(T12zz, T12[2,2])

    def test_H2_iso(self):

        h2 = PointDipoleList(iterize(H2["POTFILE"]))

        self.assertAlmostEqual(h2.alpha_iso(), H2["ALPHA_ISO"], places=DECIMALS)

    def test_H2_par(self):

        h2 = PointDipoleList(iterize(H2["POTFILE"]))
        h2_alpha = h2.alpha()
        h2_alpha_par = h2_alpha[2, 2]
        
        self.assertAlmostEqual(h2_alpha_par, H2["ALPHA_PAR"], places=DECIMALS)

    def test_H2_ort(self):

        h2 = PointDipoleList(iterize(H2["POTFILE"]))
        h2_alpha = h2.alpha()
        h2_alpha_ort = h2_alpha[0, 0]
        
        self.assertAlmostEqual(h2_alpha_ort, H2["ALPHA_ORT"], places=DECIMALS)

    def test_N2_iso(self):

        n2 = PointDipoleList(iterize(N2["POTFILE"]))

        self.assertAlmostEqual(n2.alpha_iso(), N2["ALPHA_ISO"], places=DECIMALS)

    def test_N2_par(self):

        n2 = PointDipoleList(iterize(N2["POTFILE"]))
        n2_alpha = n2.alpha()
        n2_alpha_par = n2_alpha[2, 2]
        
        self.assertAlmostEqual(n2_alpha_par, N2["ALPHA_PAR"], places=DECIMALS)

    def test_N2_ort(self):

        n2 = PointDipoleList(iterize(N2["POTFILE"]))
        n2_alpha = n2.alpha()
        n2_alpha_ort = n2_alpha[0, 0]
        
        self.assertAlmostEqual(n2_alpha_ort, N2["ALPHA_ORT"], places=DECIMALS)

    def test_O2_iso(self):

        o2 = PointDipoleList(iterize(O2["POTFILE"]))

        self.assertAlmostEqual(o2.alpha_iso(), O2["ALPHA_ISO"], places=DECIMALS)

    def test_O2_par(self):

        o2 = PointDipoleList(iterize(O2["POTFILE"]))
        o2_alpha = o2.alpha()
        o2_alpha_par = o2_alpha[2, 2]
        
        self.assertAlmostEqual(o2_alpha_par, O2["ALPHA_PAR"], places=DECIMALS)

    def test_O2_ort(self):

        o2 = PointDipoleList(iterize(O2["POTFILE"]))
        o2_alpha = o2.alpha()
        o2_alpha_ort = o2_alpha[0, 0]
        
        self.assertAlmostEqual(o2_alpha_ort, O2["ALPHA_ORT"], places=DECIMALS)

    def test_Cl2_iso(self):

        cl2 = PointDipoleList(iterize(Cl2["POTFILE"]))

        self.assertAlmostEqual(cl2.alpha_iso(), Cl2["ALPHA_ISO"], places=DECIMALS)

    def test_Cl2_par(self):

        cl2 = PointDipoleList(iterize(Cl2["POTFILE"]))
        cl2_alpha = cl2.alpha()
        cl2_alpha_par = cl2_alpha[2, 2]
        
        self.assertAlmostEqual(cl2_alpha_par, Cl2["ALPHA_PAR"], places=DECIMALS)

    def test_Cl2_ort(self):

        cl2 = PointDipoleList(iterize(Cl2["POTFILE"]))
        cl2_alpha = cl2.alpha()
        cl2_alpha_ort = cl2_alpha[0, 0]
        
        self.assertAlmostEqual(cl2_alpha_ort, Cl2["ALPHA_ORT"], places=DECIMALS)

    def test_HCl_iso(self):

        hcl = PointDipoleList(iterize(HCl["POTFILE"]))

        self.assertAlmostEqual(hcl.alpha_iso(), HCl["ALPHA_ISO"], places=DECIMALS)

    def test_HCl_par(self):

        hcl = PointDipoleList(iterize(HCl["POTFILE"]))
        hcl_alpha = hcl.alpha()
        hcl_alpha_par = hcl_alpha[2, 2]
        
        self.assertAlmostEqual(hcl_alpha_par, HCl["ALPHA_PAR"], places=DECIMALS)

    def test_HCl_ort(self):

        hcl = PointDipoleList(iterize(HCl["POTFILE"]))
        hcl_alpha = hcl.alpha()
        hcl_alpha_ort = hcl_alpha[0, 0]
        
        self.assertAlmostEqual(hcl_alpha_ort, HCl["ALPHA_ORT"], places=DECIMALS)

    def test_HBr_iso(self):

        hbr = PointDipoleList(iterize(HBr["POTFILE"]))

        self.assertAlmostEqual(hbr.alpha_iso(), HBr["ALPHA_ISO"], places=DECIMALS)

    def test_HBr_par(self):

        hbr = PointDipoleList(iterize(HBr["POTFILE"]))
        hbr_alpha = hbr.alpha()
        hbr_alpha_par = hbr_alpha[2, 2]
        
        self.assertAlmostEqual(hbr_alpha_par, HBr["ALPHA_PAR"], places=DECIMALS)

    def test_HBr_ort(self):

        hbr = PointDipoleList(iterize(HBr["POTFILE"]))
        hbr_alpha = hbr.alpha()
        hbr_alpha_ort = hbr_alpha[0, 0]
        
        self.assertAlmostEqual(hbr_alpha_ort, HBr["ALPHA_ORT"], places=DECIMALS)

    def test_HI_iso(self):

        hi = PointDipoleList(iterize(HI["POTFILE"]))

        self.assertAlmostEqual(hi.alpha_iso(), HI["ALPHA_ISO"], places=DECIMALS)

    def test_HI_par(self):

        hi = PointDipoleList(iterize(HI["POTFILE"]))
        hi_alpha = hi.alpha()
        hi_alpha_par = hi_alpha[2, 2]
        
        self.assertAlmostEqual(hi_alpha_par, HI["ALPHA_PAR"], places=DECIMALS)

    def test_HI_ort(self):

        hi = PointDipoleList(iterize(HI["POTFILE"]))
        hi_alpha = hi.alpha()
        hi_alpha_ort = hi_alpha[0, 0]
        
        self.assertAlmostEqual(hi_alpha_ort, HI["ALPHA_ORT"], places=DECIMALS)

    def test_CO_iso(self):

        co = PointDipoleList(iterize(CO["POTFILE"]))

        self.assertAlmostEqual(co.alpha_iso(), CO["ALPHA_ISO"], places=DECIMALS)

    def test_CO_par(self):

        co = PointDipoleList(iterize(CO["POTFILE"]))
        co_alpha = co.alpha()
        co_alpha_par = co_alpha[2, 2]
        
        self.assertAlmostEqual(co_alpha_par, CO["ALPHA_PAR"], places=DECIMALS)

    def test_CO_ort(self):

        co = PointDipoleList(iterize(CO["POTFILE"]))
        co_alpha = co.alpha()
        co_alpha_ort = co_alpha[0, 0]
        
        self.assertAlmostEqual(co_alpha_ort, CO["ALPHA_ORT"], places=DECIMALS)

    def test_CH4_iso(self):

        ch4 = PointDipoleList(iterize(CH4["POTFILE"]))
        self.assertAlmostEqual(ch4.alpha_iso(), CH4["ALPHA_ISO"], places=DECIMALS)

    def test_CH3OH_iso(self):

        ch3oh = PointDipoleList(iterize(CH3OH["POTFILE"]))
        self.assertAlmostEqual(ch3oh.alpha_iso(), CH3OH["ALPHA_ISO"], places=DECIMALS)

    def test_C2H6_iso(self):

        c2h6 = PointDipoleList(iterize(C2H6["POTFILE"]))
        self.assertAlmostEqual(.1*c2h6.alpha_iso(), .1*C2H6["ALPHA_ISO"], places=DECIMALS)

    def test_C3H8_iso(self):

        c3h8 = PointDipoleList(iterize(C3H8["POTFILE"]))
        self.assertAlmostEqual(c3h8.alpha_iso(), C3H8["ALPHA_ISO"], places=DECIMALS)

    def test_CP_iso(self):

        cp = PointDipoleList(iterize(CP["POTFILE"]))
        self.assertAlmostEqual(.1*cp.alpha_iso(), .1*CP["ALPHA_ISO"], places=DECIMALS)

    def test_NP_iso(self):

        np = PointDipoleList(iterize(NP["POTFILE"]))
        self.assertAlmostEqual(.1*np.alpha_iso(), .1*NP["ALPHA_ISO"], places=DECIMALS)

    def test_DME_iso(self):

        dme = PointDipoleList(iterize(DME["POTFILE"]))
        self.assertAlmostEqual(dme.alpha_iso(), DME["ALPHA_ISO"], places=DECIMALS)
                        
        
if __name__ == "__main__":
    unittest.main()

