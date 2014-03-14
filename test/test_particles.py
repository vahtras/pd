import unittest
import numpy as np
from ..particles import PointDipole, PointDipoleList

DECIMALS = 1
ANGSTROM = 1.88971616463 #Bohr
ANGSTROM3 = 1/1.48184712e-1 # a.u.
DIATOMIC = """AU
2 1 0 1
1  0.000  0.000  %f 0.000 0.000 0.000 0.000 %f
1  0.000  0.000  %f 0.000 0.000 0.000 0.000 %f
"""

# Appelquist data

H2 = {
    "R": .7413 * ANGSTROM,
    "ALPHA_H": 0.168 * ANGSTROM3,
    "ALPHA_ISO": 0.80 * ANGSTROM3,
    "ALPHA_PAR": 1.92 * ANGSTROM3,
    "ALPHA_ORT": 0.24 * ANGSTROM3,
    }
H2["POTFILE"] = DIATOMIC % (0, H2["ALPHA_H"], H2["R"], H2["ALPHA_H"])

N2 = {
    "R": 1.0976 * ANGSTROM,
    "ALPHA_N": 0.492 * ANGSTROM3,
    "ALPHA_ISO": 1.76 * ANGSTROM3,
    "ALPHA_PAR": 3.84 * ANGSTROM3,
    "ALPHA_ORT": 0.72 * ANGSTROM3,
    }
N2["POTFILE"] = DIATOMIC % (0, N2["ALPHA_N"], N2["R"], N2["ALPHA_N"])

O2 = {
    "R": 1.2074 * ANGSTROM,
    "ALPHA_O": 0.562 * ANGSTROM3,
    "ALPHA_ISO": 1.60 * ANGSTROM3,
    "ALPHA_PAR": 3.11 * ANGSTROM3,
    "ALPHA_ORT": 0.85 * ANGSTROM3,
    }
O2["POTFILE"] = DIATOMIC % (0, O2["ALPHA_O"], O2["R"], O2["ALPHA_O"])

Cl2 = {
    "R": 1.2074 * ANGSTROM,
    "ALPHA_Cl": 0.562 * ANGSTROM3,
    "ALPHA_ISO": 1.60 * ANGSTROM3,
    "ALPHA_PAR": 3.11 * ANGSTROM3,
    "ALPHA_ORT": 0.85 * ANGSTROM3,
    }
Cl2["POTFILE"] = DIATOMIC % (0, Cl2["ALPHA_Cl"], Cl2["R"], Cl2["ALPHA_Cl"])

HCl = {
    "R": 1.2745 * ANGSTROM,
    "ALPHA_H": 2.39 * ANGSTROM3,
    "ALPHA_Cl": 0.059 * ANGSTROM3,
    "ALPHA_ISO": 2.63 * ANGSTROM3,
    "ALPHA_PAR": 3.13 * ANGSTROM3,
    "ALPHA_ORT": 2.39 * ANGSTROM3,
    }
HCl["POTFILE"] = DIATOMIC % (0, HCl["ALPHA_H"], HCl["R"], HCl["ALPHA_Cl"])

HBr = {
    "R": 1.408 * ANGSTROM,
    "ALPHA_H": 3.31 * ANGSTROM3,
    "ALPHA_Br": 0.071 * ANGSTROM3,
    "ALPHA_ISO": 3.61 * ANGSTROM3,
    "ALPHA_PAR": 4.22 * ANGSTROM3,
    "ALPHA_ORT": 3.31 * ANGSTROM3,
    }
HBr["POTFILE"] = DIATOMIC % (0, HBr["ALPHA_H"], HBr["R"], HBr["ALPHA_Br"])

HI = {
    "R": 1.609 * ANGSTROM,
    "ALPHA_H": 4.89 * ANGSTROM3,
    "ALPHA_I": 0.129 * ANGSTROM3,
    "ALPHA_ISO": 5.45 * ANGSTROM3,
    "ALPHA_PAR": 6.58 * ANGSTROM3,
    "ALPHA_ORT": 4.89 * ANGSTROM3,
    }
HI["POTFILE"] = DIATOMIC % (0, HI["ALPHA_H"], HI["R"], HI["ALPHA_I"])

CO = {
    "R": 1.1282 * ANGSTROM,
    "ALPHA_C": 1.624 * ANGSTROM3,
    "ALPHA_O": 0.071 * ANGSTROM3,
    "ALPHA_ISO": 1.95 * ANGSTROM3,
    "ALPHA_PAR": 2.60 * ANGSTROM3,
    "ALPHA_ORT": 1.625 * ANGSTROM3,
    }
CO["POTFILE"] = DIATOMIC % (0, CO["ALPHA_C"], CO["R"], CO["ALPHA_O"])

class PointDipoleTest(unittest.TestCase):
    """Test basic particle properties"""

    def setUp(self):
        self.particle = PointDipole(0.,0.,0.,1.0,0.1,0.2, 0.3,0.05)

    def test_coor(self):
        np.allclose(self.particle.r, (0., 0., 0.))

    def test_charge(self):
        self.assertEqual(self.particle.q, 1.0)

    def test_dipole(self):
        np.allclose(self.particle.p, (0.1, 0.2, 0.3))

    def test_alpha(self):
        self.assertEqual(self.particle.a[0,0], 0.05)

    def test_str(self):
        self.particle.fmt = "%5.2f"
        self.assertEqual(str(self.particle),
            " 0.00 0.00 0.00 1.00 0.10 0.20 0.30 0.05"
            )
        
class PointDipoleListTest(unittest.TestCase):
    def setUp(self):
        # mimic file object
        pf = iter("""AU
3 1 0 1
1  0.000  0.000  0.698 -0.703 -0.000 0.000 -0.284 4.230
1 -1.481  0.000 -0.349  0.352  0.153 0.000  0.127 1.089
1  1.481  0.000 -0.349  0.352 -0.153 0.000  0.127 1.089
2  0.000  0.000  0.000  0.000  0.000 0.000  0.000 0.000""".split("\n"))
        self.pdl = PointDipoleList(pf)

    def test_number(self):
        self.assertEqual(len(self.pdl), 4)

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

        h2 = PointDipoleList(iter(H2["POTFILE"].strip().split('\n')))

        self.assertAlmostEqual(h2.alpha_iso(), H2["ALPHA_ISO"], places=DECIMALS)

    def test_H2_par(self):

        h2 = PointDipoleList(iter(H2["POTFILE"].strip().split('\n')))
        h2_alpha = h2.alpha()
        h2_alpha_par = h2_alpha[2, 2]
        
        self.assertAlmostEqual(h2_alpha_par, H2["ALPHA_PAR"], places=DECIMALS)

    def test_H2_ort(self):

        h2 = PointDipoleList(iter(H2["POTFILE"].strip().split('\n')))
        h2_alpha = h2.alpha()
        h2_alpha_ort = h2_alpha[0, 0]
        
        self.assertAlmostEqual(h2_alpha_ort, H2["ALPHA_ORT"], places=DECIMALS)

    def test_N2_iso(self):

        n2 = PointDipoleList(iter(N2["POTFILE"].strip().split('\n')))

        self.assertAlmostEqual(n2.alpha_iso(), N2["ALPHA_ISO"], places=DECIMALS)

    def test_N2_par(self):

        n2 = PointDipoleList(iter(N2["POTFILE"].strip().split('\n')))
        n2_alpha = n2.alpha()
        n2_alpha_par = n2_alpha[2, 2]
        
        self.assertAlmostEqual(n2_alpha_par, N2["ALPHA_PAR"], places=DECIMALS)

    def test_N2_ort(self):

        n2 = PointDipoleList(iter(N2["POTFILE"].strip().split('\n')))
        n2_alpha = n2.alpha()
        n2_alpha_ort = n2_alpha[0, 0]
        
        self.assertAlmostEqual(n2_alpha_ort, N2["ALPHA_ORT"], places=DECIMALS)

    def test_O2_iso(self):

        o2 = PointDipoleList(iter(O2["POTFILE"].strip().split('\n')))

        self.assertAlmostEqual(o2.alpha_iso(), O2["ALPHA_ISO"], places=DECIMALS)

    def test_O2_par(self):

        o2 = PointDipoleList(iter(O2["POTFILE"].strip().split('\n')))
        o2_alpha = o2.alpha()
        o2_alpha_par = o2_alpha[2, 2]
        
        self.assertAlmostEqual(o2_alpha_par, O2["ALPHA_PAR"], places=DECIMALS)

    def test_O2_ort(self):

        o2 = PointDipoleList(iter(O2["POTFILE"].strip().split('\n')))
        o2_alpha = o2.alpha()
        o2_alpha_ort = o2_alpha[0, 0]
        
        self.assertAlmostEqual(o2_alpha_ort, O2["ALPHA_ORT"], places=DECIMALS)

    def test_Cl2_iso(self):

        cl2 = PointDipoleList(iter(Cl2["POTFILE"].strip().split('\n')))

        self.assertAlmostEqual(cl2.alpha_iso(), Cl2["ALPHA_ISO"], places=DECIMALS)

    def test_Cl2_par(self):

        cl2 = PointDipoleList(iter(Cl2["POTFILE"].strip().split('\n')))
        cl2_alpha = cl2.alpha()
        cl2_alpha_par = cl2_alpha[2, 2]
        
        self.assertAlmostEqual(cl2_alpha_par, Cl2["ALPHA_PAR"], places=DECIMALS)

    def test_Cl2_ort(self):

        cl2 = PointDipoleList(iter(Cl2["POTFILE"].strip().split('\n')))
        cl2_alpha = cl2.alpha()
        cl2_alpha_ort = cl2_alpha[0, 0]
        
        self.assertAlmostEqual(cl2_alpha_ort, Cl2["ALPHA_ORT"], places=DECIMALS)

    def test_HCl_iso(self):

        hcl = PointDipoleList(iter(HCl["POTFILE"].strip().split('\n')))

        self.assertAlmostEqual(hcl.alpha_iso(), HCl["ALPHA_ISO"], places=DECIMALS)

    def test_HCl_par(self):

        hcl = PointDipoleList(iter(HCl["POTFILE"].strip().split('\n')))
        hcl_alpha = hcl.alpha()
        hcl_alpha_par = hcl_alpha[2, 2]
        
        self.assertAlmostEqual(hcl_alpha_par, HCl["ALPHA_PAR"], places=DECIMALS)

    def test_HCl_ort(self):

        hcl = PointDipoleList(iter(HCl["POTFILE"].strip().split('\n')))
        hcl_alpha = hcl.alpha()
        hcl_alpha_ort = hcl_alpha[0, 0]
        
        self.assertAlmostEqual(hcl_alpha_ort, HCl["ALPHA_ORT"], places=DECIMALS)

    def test_HBr_iso(self):

        hbr = PointDipoleList(iter(HBr["POTFILE"].strip().split('\n')))

        self.assertAlmostEqual(hbr.alpha_iso(), HBr["ALPHA_ISO"], places=DECIMALS)

    def test_HBr_par(self):

        hbr = PointDipoleList(iter(HBr["POTFILE"].strip().split('\n')))
        hbr_alpha = hbr.alpha()
        hbr_alpha_par = hbr_alpha[2, 2]
        
        self.assertAlmostEqual(hbr_alpha_par, HBr["ALPHA_PAR"], places=DECIMALS)

    def test_HBr_ort(self):

        hbr = PointDipoleList(iter(HBr["POTFILE"].strip().split('\n')))
        hbr_alpha = hbr.alpha()
        hbr_alpha_ort = hbr_alpha[0, 0]
        
        self.assertAlmostEqual(hbr_alpha_ort, HBr["ALPHA_ORT"], places=DECIMALS)

    def test_HI_iso(self):

        hi = PointDipoleList(iter(HI["POTFILE"].strip().split('\n')))

        self.assertAlmostEqual(hi.alpha_iso(), HI["ALPHA_ISO"], places=DECIMALS)

    def test_HI_par(self):

        hi = PointDipoleList(iter(HI["POTFILE"].strip().split('\n')))
        hi_alpha = hi.alpha()
        hi_alpha_par = hi_alpha[2, 2]
        
        self.assertAlmostEqual(hi_alpha_par, HI["ALPHA_PAR"], places=DECIMALS)

    def test_HI_ort(self):

        hi = PointDipoleList(iter(HI["POTFILE"].strip().split('\n')))
        hi_alpha = hi.alpha()
        hi_alpha_ort = hi_alpha[0, 0]
        
        self.assertAlmostEqual(hi_alpha_ort, HI["ALPHA_ORT"], places=DECIMALS)

    def test_CO_iso(self):

        co = PointDipoleList(iter(CO["POTFILE"].strip().split('\n')))

        self.assertAlmostEqual(co.alpha_iso(), CO["ALPHA_ISO"], places=DECIMALS)

    def test_CO_par(self):

        co = PointDipoleList(iter(CO["POTFILE"].strip().split('\n')))
        co_alpha = co.alpha()
        co_alpha_par = co_alpha[2, 2]
        
        self.assertAlmostEqual(co_alpha_par, CO["ALPHA_PAR"], places=DECIMALS)

    def test_CO_ort(self):

        co = PointDipoleList(iter(CO["POTFILE"].strip().split('\n')))
        co_alpha = co.alpha()
        co_alpha_ort = co_alpha[0, 0]
        
        self.assertAlmostEqual(co_alpha_ort, CO["ALPHA_ORT"], places=DECIMALS)


                        
        
if __name__ == "__main__":
    unittest.main()

