import unittest
import numpy as np
from ..particles import PointDipole, PointDipoleList
from .util import iterize

I_3 = np.identity(3)
ORIGO = np.zeros(3)
BETA_ZERO = np.zeros((3, 3, 3))

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

# Other model systems

H2O_DIMER = """AU
2 1 1 0
1 0.00000  0.00000  0.48861 0.0 0.00000 -0.00000 -0.76539  6.61822
1 0.00000  0.00000  5.48861 0.0 0.00000 -0.00000 -0.76539  6.61822 
"""
        
class PointDipoleListTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_H2_iso(self):

        h2 = PointDipoleList.from_string(H2["POTFILE"])
        self.assertAlmostEqual(h2.alpha_iso(), H2["ALPHA_ISO"], places=DECIMALS)

    def test_H2_par(self):

        h2 = PointDipoleList.from_string(H2["POTFILE"])
        h2_alpha = h2.alpha()
        h2_alpha_par = h2_alpha[2, 2]
        
        self.assertAlmostEqual(h2_alpha_par, H2["ALPHA_PAR"], places=DECIMALS)

    def test_H2_ort(self):

        h2 = PointDipoleList.from_string(H2["POTFILE"])
        h2_alpha = h2.alpha()
        h2_alpha_ort = h2_alpha[0, 0]
        
        self.assertAlmostEqual(h2_alpha_ort, H2["ALPHA_ORT"], places=DECIMALS)

    def test_H2_individual_beta(self):

        h2 = PointDipoleList.from_string(H2["POTFILE"])
        ha, hb = h2
        
        np.testing.assert_array_equal(ha.b, BETA_ZERO)

    def test_N2_iso(self):

        n2 = PointDipoleList.from_string(N2["POTFILE"])

        self.assertAlmostEqual(n2.alpha_iso(), N2["ALPHA_ISO"], places=DECIMALS)

    def test_N2_par(self):

        n2 = PointDipoleList.from_string(N2["POTFILE"])
        n2_alpha = n2.alpha()
        n2_alpha_par = n2_alpha[2, 2]
        
        self.assertAlmostEqual(n2_alpha_par, N2["ALPHA_PAR"], places=DECIMALS)

    def test_N2_ort(self):

        n2 = PointDipoleList.from_string(N2["POTFILE"])
        n2_alpha = n2.alpha()
        n2_alpha_ort = n2_alpha[0, 0]
        
        self.assertAlmostEqual(n2_alpha_ort, N2["ALPHA_ORT"], places=DECIMALS)

    def test_O2_iso(self):

        o2 = PointDipoleList.from_string(O2["POTFILE"])

        self.assertAlmostEqual(o2.alpha_iso(), O2["ALPHA_ISO"], places=DECIMALS)

    def test_O2_par(self):

        o2 = PointDipoleList.from_string(O2["POTFILE"])
        o2_alpha = o2.alpha()
        o2_alpha_par = o2_alpha[2, 2]
        
        self.assertAlmostEqual(o2_alpha_par, O2["ALPHA_PAR"], places=DECIMALS)

    def test_O2_ort(self):

        o2 = PointDipoleList.from_string(O2["POTFILE"])
        o2_alpha = o2.alpha()
        o2_alpha_ort = o2_alpha[0, 0]
        
        self.assertAlmostEqual(o2_alpha_ort, O2["ALPHA_ORT"], places=DECIMALS)

    def test_Cl2_iso(self):

        cl2 = PointDipoleList.from_string(Cl2["POTFILE"])

        self.assertAlmostEqual(cl2.alpha_iso(), Cl2["ALPHA_ISO"], places=DECIMALS)

    def test_Cl2_par(self):

        cl2 = PointDipoleList.from_string(Cl2["POTFILE"])
        cl2_alpha = cl2.alpha()
        cl2_alpha_par = cl2_alpha[2, 2]
        
        self.assertAlmostEqual(cl2_alpha_par, Cl2["ALPHA_PAR"], places=DECIMALS)

    def test_Cl2_ort(self):

        cl2 = PointDipoleList.from_string(Cl2["POTFILE"])
        cl2_alpha = cl2.alpha()
        cl2_alpha_ort = cl2_alpha[0, 0]
        
        self.assertAlmostEqual(cl2_alpha_ort, Cl2["ALPHA_ORT"], places=DECIMALS)

    def test_HCl_iso(self):

        hcl = PointDipoleList.from_string(HCl["POTFILE"])

        self.assertAlmostEqual(hcl.alpha_iso(), HCl["ALPHA_ISO"], places=DECIMALS)

    def test_HCl_par(self):

        hcl = PointDipoleList.from_string(HCl["POTFILE"])
        hcl_alpha = hcl.alpha()
        hcl_alpha_par = hcl_alpha[2, 2]
        
        self.assertAlmostEqual(hcl_alpha_par, HCl["ALPHA_PAR"], places=DECIMALS)

    def test_HCl_ort(self):

        hcl = PointDipoleList.from_string(HCl["POTFILE"])
        hcl_alpha = hcl.alpha()
        hcl_alpha_ort = hcl_alpha[0, 0]
        
        self.assertAlmostEqual(hcl_alpha_ort, HCl["ALPHA_ORT"], places=DECIMALS)

    def test_HBr_iso(self):

        hbr = PointDipoleList.from_string(HBr["POTFILE"])

        self.assertAlmostEqual(hbr.alpha_iso(), HBr["ALPHA_ISO"], places=DECIMALS)

    def test_HBr_par(self):

        hbr = PointDipoleList.from_string(HBr["POTFILE"])
        hbr_alpha = hbr.alpha()
        hbr_alpha_par = hbr_alpha[2, 2]
        
        self.assertAlmostEqual(hbr_alpha_par, HBr["ALPHA_PAR"], places=DECIMALS)

    def test_HBr_ort(self):

        hbr = PointDipoleList.from_string(HBr["POTFILE"])
        hbr_alpha = hbr.alpha()
        hbr_alpha_ort = hbr_alpha[0, 0]
        
        self.assertAlmostEqual(hbr_alpha_ort, HBr["ALPHA_ORT"], places=DECIMALS)

    def test_HI_iso(self):

        hi = PointDipoleList.from_string(HI["POTFILE"])

        self.assertAlmostEqual(hi.alpha_iso(), HI["ALPHA_ISO"], places=DECIMALS)

    def test_HI_par(self):

        hi = PointDipoleList.from_string(HI["POTFILE"])
        hi_alpha = hi.alpha()
        hi_alpha_par = hi_alpha[2, 2]
        
        self.assertAlmostEqual(hi_alpha_par, HI["ALPHA_PAR"], places=DECIMALS)

    def test_HI_ort(self):

        hi = PointDipoleList.from_string(HI["POTFILE"])
        hi_alpha = hi.alpha()
        hi_alpha_ort = hi_alpha[0, 0]
        
        self.assertAlmostEqual(hi_alpha_ort, HI["ALPHA_ORT"], places=DECIMALS)

    def test_CO_iso(self):

        co = PointDipoleList.from_string(CO["POTFILE"])

        self.assertAlmostEqual(co.alpha_iso(), CO["ALPHA_ISO"], places=DECIMALS)

    def test_CO_par(self):

        co = PointDipoleList.from_string(CO["POTFILE"])
        co_alpha = co.alpha()
        co_alpha_par = co_alpha[2, 2]
        
        self.assertAlmostEqual(co_alpha_par, CO["ALPHA_PAR"], places=DECIMALS)

    def test_CO_ort(self):

        co = PointDipoleList.from_string(CO["POTFILE"])
        co_alpha = co.alpha()
        co_alpha_ort = co_alpha[0, 0]
        
        self.assertAlmostEqual(co_alpha_ort, CO["ALPHA_ORT"], places=DECIMALS)

    def test_CH4_iso(self):

        ch4 = PointDipoleList.from_string(CH4["POTFILE"])
        self.assertAlmostEqual(ch4.alpha_iso(), CH4["ALPHA_ISO"], places=DECIMALS)

    def test_CH3OH_iso(self):

        ch3oh = PointDipoleList.from_string(CH3OH["POTFILE"])
        self.assertAlmostEqual(ch3oh.alpha_iso(), CH3OH["ALPHA_ISO"], places=DECIMALS)

    def test_C2H6_iso(self):

        c2h6 = PointDipoleList.from_string(C2H6["POTFILE"])
        self.assertAlmostEqual(.1*c2h6.alpha_iso(), .1*C2H6["ALPHA_ISO"], places=DECIMALS)

    def test_C3H8_iso(self):

        c3h8 = PointDipoleList.from_string(C3H8["POTFILE"])
        self.assertAlmostEqual(c3h8.alpha_iso(), C3H8["ALPHA_ISO"], places=DECIMALS)

    def test_CP_iso(self):

        cp = PointDipoleList.from_string(CP["POTFILE"])
        self.assertAlmostEqual(.1*cp.alpha_iso(), .1*CP["ALPHA_ISO"], places=DECIMALS)

    def test_NP_iso(self):

        np = PointDipoleList.from_string(NP["POTFILE"])
        self.assertAlmostEqual(.1*np.alpha_iso(), .1*NP["ALPHA_ISO"], places=DECIMALS)

    def test_DME_iso(self):

        dme = PointDipoleList.from_string(DME["POTFILE"])
        self.assertAlmostEqual(dme.alpha_iso(), DME["ALPHA_ISO"], places=DECIMALS)

    def test_h2o_dimer_finite_field_p(self):

        h2o_dimer = PointDipoleList.from_string(H2O_DIMER)
        alphas = h2o_dimer.solve_Applequist_equation()

        h2o_dimer.solve_scf_for_external([0, 0, .005])
        p1 = h2o_dimer[0].dp
        h2o_dimer.solve_scf_for_external([0, 0, -.005])
        p2 = h2o_dimer[0].dp
        dPdE = (p1 - p2)/0.01

        np.allclose(alphas[0][:,2], dPdE)
        

#
# Some refactoring tests
#
    def test_form_Applequist_rhs(self):
        h2 = PointDipoleList.from_string(H2["POTFILE"])
        h2_rhs = h2.form_Applequist_rhs()
        h2_rhs_ref = np.array([
            [0.168, 0, 0], 
            [0., 0.168, 0.], 
            [0., 0., 0.168],
            [0.168, 0, 0], 
            [0., 0.168, 0.], 
            [0., 0., 0.168]
            ])
        np.testing.assert_array_equal(h2_rhs, h2_rhs_ref)

    def test_form_Applequist_coefficient_matrix(self):
        h2 = PointDipoleList.from_string(H2["POTFILE"])
        L_h2_ref = np.array([
            [1., 0., 0., 0.41240819, 0., 0.],
            [0., 1., 0., 0., 0.41240819, 0.],
            [0., 0., 1., 0., 0.,-0.82481638],
            [0.41240819, 0., 0., 1., 0., 0.],
            [0., 0.41240819, 0., 0., 1., 0.],
            [0., 0.,-0.82481638, 0., 0., 1.]
            ])
        L_h2 = h2.form_Applequist_coefficient_matrix()
        np.testing.assert_array_almost_equal(L_h2, L_h2_ref)

    def test_solve_Applequist_equation(self):
        h2 = PointDipoleList.from_string(H2["POTFILE"])
        alphas = h2.solve_Applequist_equation()
        alphas_ref =  np.array([[
            [0.11894578, 0., 0.],
            [0., 0.11894578,  0.],
            [0., 0., 0.95899377]
            ], [
            [0.11894578, 0., 0.],
            [0., 0.11894578,  0.],
            [0., 0., 0.95899377]
            ]])
        np.testing.assert_almost_equal(alphas, alphas_ref)

    def test_induced_parallel_dipole_on_one_atom(self):
        h2 = PointDipoleList.from_string(H2["POTFILE"])
        E_external = np.array([0., 0., 1.,])
        p_ind_ref =  np.array([0., 0., 0.95899377])
        h2.solve_scf_for_external(E_external, max_it = 100)
        np.testing.assert_almost_equal(h2[0].p, p_ind_ref, decimal=6)

    def test_induced_orthogonal_dipole_on_one_atom(self):
        h2 = PointDipoleList.from_string(H2["POTFILE"])
        E_external = np.array([1., 0., 0.,])
        p_ind_ref =  np.array([0.11894578, 0., 0.])
        h2.solve_scf_for_external(E_external, max_it = 100)
        np.testing.assert_almost_equal(h2[0].p, p_ind_ref, decimal=6)
        
    def test_evaluate_local_field_at_atoms(self):
        h2 = PointDipoleList.from_string(H2["POTFILE"])
        E_external = np.array([0., 0., 1.,])
        E_local = h2.evaluate_field_at_atoms()
        np.testing.assert_almost_equal(E_local, [[0, 0, 0], [0, 0, 0]])

    def test_evaluate_total_field_at_atoms(self):
        h2 = PointDipoleList(iterize(H2["POTFILE"]))
        E_external = np.array([0., 0., 1.,])
        E_local = h2.evaluate_field_at_atoms(external=E_external)
        np.testing.assert_array_almost_equal(E_local, [[0, 0, 1], [0, 0, 1]])
        

        
if __name__ == "__main__":
    unittest.main()

