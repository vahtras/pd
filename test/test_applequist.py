import unittest
import numpy as np
from ..particles import PointDipoleList
from ..constants import *
from data_applequist import *

DECIMALS = 1


class ApplequistTest(unittest.TestCase):

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
        
        np.testing.assert_array_equal(ha._b0, ZERO_RANK_3_TENSOR)

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
        
        self.assertAlmostEqual(cl2_alpha_par, Cl2["ALPHA_PAR"], DECIMALS )

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
        self.assertAlmostEqual(0.1*ch3oh.alpha_iso(), 0.1*CH3OH["ALPHA_ISO"], places=DECIMALS)

    def test_C2H6_iso(self):

        c2h6 = PointDipoleList.from_string(C2H6["POTFILE"])
        self.assertAlmostEqual(.1*c2h6.alpha_iso(), .1*C2H6["ALPHA_ISO"], places=DECIMALS)

    def test_C3H8_iso(self):

        c3h8 = PointDipoleList.from_string(C3H8["POTFILE"])
        self.assertAlmostEqual(c3h8.alpha_iso(), C3H8["ALPHA_ISO"], places=DECIMALS)

    def test_CP_iso(self):

        cp = PointDipoleList.from_string(CP["POTFILE"])
        self.assertAlmostEqual(.01*cp.alpha_iso(), .01*CP["ALPHA_ISO"], places=DECIMALS)

    def test_NP_iso(self):

        np = PointDipoleList.from_string(NP["POTFILE"])
        self.assertAlmostEqual(.01*np.alpha_iso(), .01*NP["ALPHA_ISO"], places=DECIMALS)

    def test_DME_iso(self):

        dme = PointDipoleList.from_string(DME["POTFILE"])
        self.assertAlmostEqual(0.1*dme.alpha_iso(), 0.1*DME["ALPHA_ISO"], places=DECIMALS)
