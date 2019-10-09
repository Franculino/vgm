#from __future__ import division

cdef extern from "math.h":
    double sin(double)
    double cos(double)
    double exp(double)
    double sqrt(double)
    double M_PI

import g_math
import units
import numpy

__all__ = ['Physiology']

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

cdef class Physiology(object):
    """This class implements physiological parameters and functions related to 
    blood flow.
    """

    cdef public dict _sf
    
    def __init__(self, defaultUnits={'length': 'um', 'mass': 'ug', 
                                     'time': 'ms'}):
        """Initializes the Physiology object.
        INPUT: defaultUnits: The default units to be used for input and output
                             as a dictionary, e.g.: {'length': 'm', 
                             'mass': 'kg', 'time': 's'}
        OUTPUT: None
        """
        
        self.tune_to_default_units(defaultUnits)
        
        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    
    
    def tune_to_default_units(self, defaultUnits):
        """Tunes the Physiology object to a set of default units. This results
        in faster execution time and less calling parameters.
        INPUT: defaultUnits: The default units to be used for input and output
                             as a dictionary, e.g.: {'length': 'm', 
                             'mass': 'kg', 'time': 's'}
        OUTPUT: None
        """
        self._sf = {}
        sf = self._sf
        sf['um -> du'] = units.scaling_factor_du('um', defaultUnits)
        sf['mm/s -> du'] = units.scaling_factor_du('mm/s', defaultUnits)
        sf['kg/m^3 -> du'] = units.scaling_factor_du('kg/m^3', defaultUnits)
        sf['mmHg -> du'] = units.scaling_factor_du('mmHg', defaultUnits)
        sf['Pa*s -> du'] = units.scaling_factor_du('Pa*s', defaultUnits)
        sf['fL -> du'] = units.scaling_factor_du('fL', defaultUnits)
        
        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    
        
    cpdef blood_velocity(self, double d, str kind):
        """Returns the blood velocity according to a fit to Zweifach's data of
        cat model (Zweifach 1974).
        INPUT: d: Diameter of the blood vessel.
               kind: Type of blood vessel. This may be either 'a' or 'v' for 
                     'artery' or 'vein' respectively.
        OUTPUT: blood velocity
        WARNING: This function only produces reliable results for diameters 
                 between 8 and 60 micron vessel diameter!
        """
        
        # The fit function assumes diameters in [micron] and returns flow speed
        # in [mm/s]. Conversion to default units is performed as required.

        cdef double sf, wa, wv
        
        d = d / self._sf['um -> du']
        sf = self._sf['mm/s -> du']
        
        d = max(min(d, 60), 8)
        
        if kind[0] == 'a':
            wa = 0.06154
            return (9.73 - 6.558*cos(d*wa) - 1.74 * sin(d*wa) - \
                    1.395 * cos(2.0*d*wa) - 0.1891 * sin(2.0*d*wa)) * sf
        elif kind[0] == 'v':           
            wv = 0.08286 
            return (5.87 - 0.7492*cos(d*wv) - 3.979*sin(d*wv) - \
                    0.1182 * cos(2*d*wv) - 0.5815 * sin(2*d*wv) - \
                    0.2073 * cos(3*d*wv) - 0.05505 * sin(3*d*wv)) * sf


    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------


    cpdef blood_density_average(self):
        """Returns an average whole-blood density value.
        Value is taken from en.Wikipedia.org for human blood (1025 kg/m^3 and 
        1125 kg/m^3 are the respective densities of plasma and cells)
        INPUT: None
        OUTPUT: Average whole-blood density.
        """
        
        return 1060.0 * self._sf['kg/m^3 -> du']
        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------


    cpdef blood_density(self, double diameter, str kind):
        """Returns a whole-blood density value, according to the given vessel 
        specifications.
        
        Blood density is computed from vessel volume, the hematocrit and the 
        densities of plasma and red blood cells (1025 kg/m^3 and 1125 kg/m^3 
        respectively, taken from en.Wikipedia.org for human blood)
        INPUT: diameter: Vessel diameter [m]
               kind: Type of blood vessel. This may be either 'a' or 'v' for 
                     'artery' or 'vein' respectively.
        OUTPUT: Blood density              
        """
        
        cdef double rho_plasma, rho_rbc, ht
            
        # rho_wholeBlood = (volume_rbc * density_rbc + 
        #                   volume_plasma * density_plasma) / total_volume
        rho_plasma = 1025. * self._sf['kg/m^3 -> du']
        rho_rbc = 1125. * self._sf['kg/m^3 -> du']

        ht = self.tube_hematocrit(diameter, kind)
        return ht * rho_rbc + (1.-ht) * rho_plasma


    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------


    cpdef plasma_density(self):
        """Returns a whole-blood density value, according to the given vessel
        specifications.

        Blood density is computed from vessel volume, the hematocrit and the
        densities of plasma and red blood cells (1025 kg/m^3 and 1125 kg/m^3
        respectively, taken from en.Wikipedia.org for human blood)
        INPUT: diameter: Vessel diameter [m]
               kind: Type of blood vessel. This may be either 'a' or 'v' for
                     'artery' or 'vein' respectively.
        OUTPUT: Blood density
        """

        cdef double rho_plasma 

        rho_plasma = 1025. * self._sf['kg/m^3 -> du']

        return rho_plasma

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------


    cpdef blood_pressure(self, double d, str kind):    
        """Returns a blood pressure value, according to the given vessel 
        specifications.
        
        Data from Lipowsky 2005 (Microvascular Rheology and Hemodynamics) have
        been fitted using Matlab's cftool. 
        INPUT: d: Diameter of the blood vessel.
               kind: Type of blood vessel. This may be either 'a' or 'v' for 
                     'artery' or 'vein' respectively.
        OUTPUT: Blood pressure
        WARNING: This function only produces reliable results for diameters
                 between 8 and 60 micron!    
        """

        cdef double sf, pressure

        # The fit function assumes diameters in [micron] and returns pressures 
        # in [mmHg]. Conversion to default units is performed as required.
        #d = d / self._sf['um -> du']
        #sf = self._sf['mmHg -> du']
        
        d = max(min(d, 60), 8)
        if kind[0] == 'a':  
            pressure =  8.218e-06 * d**4. - 0.001258 * d**3. + \
                        0.05714 * d**2. + 0.4105 * d + 27.8
        elif kind[0] == 'v':
            pressure = 4.351e-06 * d**4. - 0.0006778 * d**3. + \
                       0.04041 * d**2. - 1.286 * d + 41.9                       
        return pressure 

        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
       
    cpdef blood_pressure_shapiro_lipowsky(self, double d, str kind):    
        """Returns a blood pressure value, according to the given vessel 
        specifications.
        
        Combination of Data from Shapiro 1971 and from Lipowsky 2005 (Microvascular Rheology and Hemodynamics).
        INPUT: d: Diameter of the blood vessel (see pressureValuesPialShapiro.py).
               kind: Tybe of blood vessel. This may be either 'a' or 'v' for 
                     'artery' or 'vein' respectively.
        OUTPUT: Blood pressure
        WARNING: This function only produces reliable results for diameters
                 between 8 and 60 micron!    
        """

        cdef double sf, pressure

        # The fit function assumes diameters in [micron] and returns pressures 
        # in [mmHg]. Conversion to default units is performed as required.
        #d = d / self._sf['um -> du']
        #sf = self._sf['mmHg -> du']
        
        d = max(min(d, 60), 8)

        if kind[0] == 'a':  
            pressure =  4.51116440e-05 * d**4. + -5.71837950e-03 * d**3. + \
                        2.26876475e-01 * d**2. + -2.09920929e+00 * d + 2.34116358e+01
        elif kind[0] == 'v':
            pressure = 4.351e-06 * d**4. - 0.0006778 * d**3. + \
                       0.04041 * d**2. - 1.286 * d + 41.9 - 16.7

        return pressure

        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
       
    cpdef blood_pressure_pial_literature(self, double d, str kind):    
        """Returns a blood pressure value, according to the given vessel 
        specifications.
        
        Combination of Data from Shapiro 1971 and from Lipowsky 2005 (Microvascular Rheology and Hemodynamics).
        INPUT: d: Diameter of the blood vessel (see pressureValuesPialShapiro.py).
               kind: Type of blood vessel. This may be either 'a' or 'v' for 
                     'artery' or 'vein' respectively.
        OUTPUT: Blood pressure
        WARNING: This function only produces reliable results for diameters
                 12.5 - 191 mum (arteries) and 34.2 - 180 mum 
        """

        cdef double sf, pressure

        # The fit function assumes diameters in [micron] and returns pressures 
        # in [mmHg]. Conversion to default units is performed as required.
        #d = d / self._sf['um -> du']
        #sf = self._sf['mmHg -> du']
        
        za=[2.59476905e-05,-8.91218311e-03,1.06888397e+00,2.04594862e+01]
        zv=[-1.46211218e-05,4.18982898e-03,-3.70140492e-01,2.41445015e+01]


        if kind[0] == 'a':  
            if d < 12.5:
                d=12.5
            elif d > 191.0:
                d=191.0
            pressure =  za[0] * d**3. + za[1] * d**2. + za[2] * d + za[3]
        elif kind[0] == 'v':
            if d < 34.2:
                d=34.2
            elif d > 180:
                d=180.0
            pressure =  zv[0] * d**3. + zv[1] * d**2. + zv[2] * d + zv[3]

        return pressure

        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
        
    cpdef tube_hematocrit(self, double diameter, str kind):
        """Returns the tube or vessel-hematocrit value, according to the given
        vessel specifications.
        
        Data from Lipowsky 1980 (In Vivo Measurements of 'Apparent Viscosity' 
        and Microvessel Hematocrit in the Mesentery of the Cat) have been 
        fitted using Matlab's cftool. Values backed by data in diameter range 
        [7,70] microns.     
        INPUT: diameter: Vessel diameter
               kind: Type of blood vessel. This may be either 'a' or 'v' for 
                     'artery' or 'vein' respectively.
        OUTPUT: Hematocrit [fraction ([0,1])]
        WARNING: This function only produces reliable results for diameters 
                 between 7 and 70 micron!      
        """

        cdef double d, w, ht

        # The fit function assumes diameters in [micron]. Conversion to default 
        # units is performed as required.
        d = diameter / self._sf['um -> du']
        
        # If d is out of bounds, clip to boundary:
        d = min(max(d, 7), 70)
        
        if kind == 'a':   
            ht = 0.002187 * d**2. + 0.1688 * d + 6.422
        elif kind == 'v':    
            w = 0.06673 
            ht = 20.09 - 4.229 * cos(d*w) - 9.744 * sin(d*w) - \
                 1.018 * cos(2.0*d*w) - 3.095 * sin(2.0*d*w) - \
                 0.1733 * cos(3.0*d*w) - 0.6688 * sin(3.0*d*w)
        else:
            print('WARNING: wrong input "%s" to tube_hematocrit!\n\
                   Vessel kind must be either a or v. \n\
                   Htt = 0.0 returned' % kind)
            return 0.

        return ht/100  # convert % to fraction ([0,1])

        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------


    cpdef discharge_to_tube_hematocrit(self, double discharge_ht, double d,
                                       bint invivo):
        """Converts discharge hematocrit to tube hematocrit based on the
        formulas provided by Pries 2005 ('Microvascular
        blood viscosity in vivo and the endothelial surface layer')
        It can be chosen from an invivo and an invitro formulation.
        INPUT: discharge_ht: Discharge Ht expressed as a fraction [0,1]
               d: Diameter of the vessel (in microns)
               invivo: Boolean, whether or not to consider ESL influence.
        OUTPUT: tube_hematocrit: Tube hematocrit expressed as a fraction
                [0,1]
        """
        
        cdef double htd, dph, htt, x

        # The fit function assumes diameters in [micron]. Conversion to default 
        # units is performed as required.    
        d = d / self._sf['um -> du']
        htd = discharge_ht

        if invivo:
            dph = self.physical_vessel_diameter(d)
            x = 1 + 1.7 * exp(-0.415*dph) - 0.6 * exp(-0.011*dph)
            htt = htd**2 + htd * (1-htd) * x
            return htt / (d / dph)**2
            # Note that the above differs from the (incorrect) formula in Pries
            # et al. 2005, which would result in: return htd * (d / dph)**2
        else:
            x = 1 + 1.7 * exp(-0.415*d) - 0.6 * exp(-0.011*d)
            htt = htd**2 + htd * (1-htd) * x
            return htt


    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------


    cpdef tube_to_discharge_hematocrit(self, double tube_ht, double d, 
                                       bint invivo):
        """Converts tube hematocrit to discharge hematocrit based on the
        formulas provided by Pries 2005 ('Microvascular
        blood viscosity in vivo and the endothelial surface layer')
        It can be chosen from an invivo and an invitro formulation.
        Special cases: If the diameter is > 1000 micrometers the invitro formulation 
        does not work anymore. The discharge hematocrit is set equal to the tube hematocrit. 
        In extreme cases in both formulations a htd > 1.0 can be obtained. htd is always 
        bounded to 1.0.
        INPUT: tube_ht: Discharge Ht expressed as a fraction [0,1]
               d: Diameter of the vessel (in microns)
               invivo: Boolean, whether or not to consider ESL influence.
        OUTPUT: discharge_hematocrit: Discharge Ht expressed as a fraction
                [0,1]
        """

        
        cdef double dph, htd, x

        # The fit function assumes diameters in [micron]. Conversion to default 
        # units is performed as required.    
        d = d / self._sf['um -> du']
        
        if invivo:
            dph = self.physical_vessel_diameter(d)
            x = 1 + 1.7 * exp(-0.415*dph) - 0.6 * exp(-0.011*dph)
            htt=tube_ht*(d / dph)**2
            htd = 0.5*(x - sqrt(-4*htt*x + x**2 + 4*htt))/(x - 1)
            if htd > 0.99:
                htd = 1.0
            return htd
            # Note that the above differs from the (incorrect) formula in Pries
            # et al. 2005, which would result in: return htd / (d / dph)**2
        else:
            x = 1 + 1.7 * exp(-0.415*d) - 0.6 * exp(-0.011*d)
            if d < 1000:
                htd = 0.5*(x - sqrt(-4*tube_ht*x + x**2 + 4*tube_ht))/(x - 1)
            else:
                htd=tube_ht
            if htd > 0.99:
                htd = 1.0
            return htd


    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    
    
    cpdef discharge_hematocrit(self, double diameter, str kind, 
                               bint invivo=0):
        """Convenience function that first calls 'tube_hematocrit' and then 
        passes the result to 'tube_to_discharge_hematocrit'
        INPUT: diameter: Vessel diameter
               kind: Type of blood vessel. This may be either 'a' or 'v' for 
                     'artery' or 'vein' respectively.
               invivo: Boolean, whether or not to consider ESL influence.
        OUTPUT: Hematocrit [fraction ([0,1])]
        WARNING: This function only produces reliable results for diameters 
                 between 7 and 70 micron!
        """                                   

        cdef double sf, tube_ht 

        # The fit function assumes diameters in [micron]. Conversion to default 
        # units is performed as required.
        sf = self._sf['um -> du']
        
        # If d is out of bounds, clip to boundary:
        diameter = min(max(diameter / sf, 7), 70) * sf

        tube_ht = self.tube_hematocrit(diameter, kind)
        return self.tube_to_discharge_hematocrit(tube_ht, diameter, invivo)


    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------


    cpdef relative_apparent_blood_viscosity(self, double diameter, 
                                            double discharge_hematocrit,
                                            bint invivo):
        """Returns the relative apparent blood-viscosity value, according to
        the given vessel specifications and discharge hematocrit.

        'Apparent', because the viscosity of blood as a suspension is not an
        intrinsic fluid property, but depends on the vessel size.
        'Relative', because the viscosity is scaled to (divided by) the
        viscosity of the solvent, i.e. the viscosity of blood plasma (as such,
        it should also be independent of temperature, Barbee 1973).
        Fit function taken from Pries 1992 (Blood viscosity in tube flow:
        dependence on diameter and hematocrit). It is assumed that the relative
        apparent viscosity is a fraction of dynamic (as opposed to kinematic)
        viscosities, although this is not stated explicitly in the paper. The
        error would be small if this is not the case, because the densities of
        plasma and RBCs do not differ significantly (<10%).
        INPUT: diameter: Vessel diameter (in microns)
               discharge_hematocrit: Discharge Ht expressed as a fraction [0,1]
               invivo: This boolean determines whether to compute in vivo
                       viscosity or in vitro.
        OUTPUT: Relative apparent blood viscosity [1.0]
        """

        cdef double d, ht_d, nu45, c, nu, deff
        
        #if discharge_hematocrit=1 we use 0.99 for the calculation, otherwise the function yields inf)
        if discharge_hematocrit == 1.0:
            discharge_hematocrit = 0.99
                                   
        # The fit function assumes diameters in [micron]. Conversion from 
        # default units is performed as required.
        sf = self._sf['um -> du']
        
        if invivo:
            d = self.physical_vessel_diameter(diameter) / sf
        else:
            d = diameter / sf

        ht_d = discharge_hematocrit
        # relative apparent viscosity at Ht = 0.45:
        nu45 = 220.0 * exp(-1.3*d) + 3.2 - 2.44 * exp(-0.06*d**0.645)
        # exponent c:
        c = (0.8 + exp(-0.075*d)) * \
            (-1.0 + 1.0/(1.0 + 10.0**-11.0 * d**12.0)) + \
            1.0/(1.0 + 10.0**-11.0 * d**12.0)
        # relative apparent viscosity:           
        nu = 1.0 + (nu45 - 1.0) * ((1.0 - ht_d)**c-1.0) / \
                    ((1.0 - 0.45)**c-1.0)
        if invivo:
            deff = self.effective_vessel_diameter(diameter, 
                                                  discharge_hematocrit)
            nu = nu * (diameter/deff)**4.

        return nu
       
    # -------------------------------------------------------------------------


    cpdef dynamic_blood_viscosity(self, double diameter,bint invivo, 
                                  str kind='a', double discharge_ht=-1.0,
                                  double tube_ht=-1.0,str plasmaType='default'):
        """Returns the dynamic viscosity of blood at 37 degrees centigrate,
        depending on the size and type of the blood vessel.
        INPUT: diameter: Vessel diameter.
               invivo: Boolean, whether the physiological blood characteristics
                       are calculated using the invivo (=True) or invitro (=False)
                       equations
               kind: (Optional) Type of blood vessel. This may be either 'a' or  
                     'v' for 'artery' or 'vein' respectively.
               tube_ht: (Optional)Tube hematocrit. Empirical data used, if not 
                        provided.
               discharge_ht: (Optional) Discharge hematocrit. Empirical data used,
                              if not provided.
        OUTPUT: Dynamic viscosity of blood
        WARNING: This function only produces reliable results for diameters 
                 between 7 and 70 micron. Diameters outside this interval will 
                 be adjusted to match the respective boundaries for the 
                 intermediate computations, i.e. [7,70] for the hematocrit 
                 computation and [3.3,1978] for the viscosity computation. That
                 is, in the ranges [3.3,7.0] and [70.0,1978.0] nu is computed 
                 using inaccurate ht values but correct diameter values in an 
                 attempt to minimize error.
        """
        
        cdef double sf, tmpDiameter, nu_rel

        sf = self._sf['um -> du']
        
        # Ensure that the tube_ht computation is in the data-backed diameter 
        # range:
        tmpDiameter = min(max(7.0*sf, diameter), 70.0*sf)

        # Compute discharge hematocrit, if required:
        if discharge_ht == -1.0:
            if tube_ht == -1.0:
                tube_ht = self.tube_hematocrit(tmpDiameter, kind)
            discharge_ht = self.tube_to_discharge_hematocrit(tube_ht, tmpDiameter,invivo)
        
        # compute nu_blood = nu_rel * nu_plasma:
        tmpDiameter = min(max(3.3*sf, diameter), 1978.0*sf)    
        nu_rel = self.relative_apparent_blood_viscosity(tmpDiameter, 
                                                        discharge_ht, invivo)
        
        return nu_rel * self.dynamic_plasma_viscosity(plasmaType=plasmaType)
               


    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    cpdef dynamic_plasma_viscosity(self,str plasmaType='default'):
        """Returns the dynamic viscosity of human plasma at 37 degrees 
        centigrate, as reported in 'Plasma viscosity: A forgotten variable' by 
        Kesmarky et al, 2008.
        INPUT: None
        OUTPUT: Dynamic viscosity of human plasma.
        """
        
        # The value reported by Kesmarky is scaled from [Pa s] to default 
        # units:
        if plasmaType == 'default':
            return 0.0012 * self._sf['Pa*s -> du']
        elif plasmaType == 'human2':
            return 0.001339 * self._sf['Pa*s -> du']
        elif plasmaType == 'francesco':
            return 0.00196 * self._sf['Pa*s -> du']
        

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    cpdef physical_esl_thickness(self, double diameter):
        """Returns the physical width of the endothelial surface layer (which 
        is independent of hematocrit), according to the empirical model of 
        Pries and Secomb in their 2005 paper 'Microvascular blood viscosity in 
        vivo and the endothelial surface layer'.
        INPUT: diameter: Anatomical diameter of the blood vessel.
        OUTPUT: Physical width of the ESL.
        """
        
        cdef double sf, doff, dcrit, d50, eamp, ewidth, epeak, wmax

        # The empirical fit function is designed for length units of microns. 
        # Hence, we need the scaling factor to and from default units:
        sf = self._sf['um -> du']
        diameter = diameter / sf
        
        # Parameters optimized for a hematocrit dependent ESL-impact on flow:
        doff = 2.4
        dcrit = 10.5
        d50 = 100.
        eamp = 1.1
        ewidth = 0.03
        epeak = 0.6
        wmax = 2.6

        if diameter <= doff:
            was = 0.
            wpeak = 0.
        elif diameter <= dcrit:
            was = (diameter-doff) / (diameter+d50-2*doff) * wmax
            wpeak = eamp * (diameter-doff) / (dcrit-doff)
        else:
            was = (diameter-doff) / (diameter+d50-2*doff) * wmax
            wpeak = eamp * exp(-ewidth*(diameter - dcrit))
        
        return (was + wpeak * epeak) * sf
        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    cpdef esl_thickness(self, double diameter):
        """Returns the physical width of the endothelial surface layer (which 
        is independent of hematocrit). The function is based on the empirical model of 
        Pries and Secomb in their 2005 paper 'Microvascular blood viscosity in 
        vivo and the endothelial surface layer'.
        However the constants ewidth and epeak have been changed and a maximum esl 
        thickness has been defined. Maximum thickness at 150 mu m = 2.04 mu m, thickness in 
        capillaries ranging from 0.06 (d = 3 mu m) to 0.47 (d = 12 mu m)
        INPUT: diameter: Anatomical diameter of the blood vessel.
        OUTPUT: Physical width of the ESL.
        """
        
        cdef double sf, doff, dcrit, d50, eamp, ewidth, epeak, wmax

        # The empirical fit function is designed for length units of microns. 
        # Hence, we need the scaling factor to and from default units:
        sf = self._sf['um -> du']
        diameter = diameter / sf
        
        # Parameters optimized for a hematocrit dependent ESL-impact on flow:
        doff = 2.4
        dcrit = 10.5
        d50 = 100.
        eamp = 1.1
        ewidth = 0.001
        epeak = 0.5
        wmax = 2.6
        dtop = 150

        if diameter > dtop:
            was = (dtop-doff) / (dtop+d50-2*doff) * wmax
            wpeak = eamp * exp(-ewidth*(dtop - dcrit))
        else:
            if diameter <= doff:
                was = 0.
                wpeak = 0.
            elif diameter <= dcrit:
                was = (diameter-doff) / (diameter+d50-2*doff) * wmax
                wpeak = eamp * (diameter-doff) / (dcrit-doff)
            else:
                was = (diameter-doff) / (diameter+d50-2*doff) * wmax
                wpeak = eamp * exp(-ewidth*(diameter - dcrit))
        
        return (was + wpeak * epeak) * sf
        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    cpdef effective_esl_thickness(self, double diameter, double htd):
        """Returns both the effective width of the endothelial surface layer 
        (which depends on the discharge hematocrit), according to the empirical
        model of Pries and Secomb in their 2005 paper 'Microvascular blood 
        viscosity in vivo and the endothelial surface layer'.
        INPUT: diameter: Anatomical diameter of the blood vessel.
               htd: Discharge hematocrit.
        OUTPUT: Effective width of the ESL (dependent on the hematocrit).
        """
        
        cdef double sf, doff, dcrit, d50, eamp, ewidth, epeak, ehd, wmax

        # The empirical fit function is designed for length units of microns. 
        # Hence, we need the scaling factor to and from default units:
        sf = self._sf['um -> du']
        diameter = diameter / sf
        
        # Parameters optimized for a hematocrit dependent ESL-impact on flow:
        doff = 2.4
        dcrit = 10.5
        d50 = 100.
        eamp = 1.1
        ewidth = 0.03
        epeak = 0.6
        ehd = 1.18
        wmax = 2.6

        if diameter <= doff:
            was = 0.
            wpeak = 0.
        elif diameter <= dcrit:
            wpeak = eamp * (diameter-doff) / (dcrit-doff)
            was = (diameter-doff) / (diameter+d50-2*doff) * wmax
        else:
            wpeak = eamp * exp(-ewidth*(diameter - dcrit))
            was = (diameter-doff) / (diameter+d50-2*doff) * wmax

        return (was + wpeak * (1 + htd * ehd)) * sf
        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    cpdef physical_vessel_diameter(self, double diameter):
        """Returns the physical vessel diameter, which is the anatomical 
        diameter minus twice the physical width of the endothelial surface 
        layer (which is hematocrit independent), according to the empirical 
        model of Pries and Secomb in their 2005 paper 'Microvascular blood 
        viscosity in vivo and the endothelial surface layer'.
        INPUT: diameter: Anatomical diameter of the blood vessel.
        OUTPUT: weff: Effective width of the ESL (dependent on the hematocrit).
        """
        
        cdef double wph = self.physical_esl_thickness(diameter)
        
        return diameter - 2*wph

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    cpdef effective_vessel_diameter(self, double diameter, double htd):
        """Returns the effective vessel diameter, which is the anatomical 
        diameter minus twice the effective width of the endothelial surface 
        layer (which depends on the discharge hematocrit), according to the 
        empirical model of Pries and Secomb in their 2005 paper 'Microvascular 
        blood viscosity in vivo and the endothelial surface layer'.
        INPUT: diameter: Anatomical diameter of the blood vessel.
               htd: Discharge hematocrit.
        OUTPUT: weff: Effective width of the ESL (dependent on the hematocrit).
        """
        
        cdef double weff = self.effective_esl_thickness(diameter, htd)
        
        return diameter - 2*weff

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    cpdef phase_separation_effect(self, double fqb, double da, double db, 
                                  double df, double htd):
        """Returns the fraction of red blood cells entering a daughter branch, 
        based on the fractional blood flow into that branch. This is an 
        empirical relation, published by Pries et al. in 2005 'Microvascular blood viscosity in 
        vivo and the endothelial surface layer'. Note that it only
        applies to a bifurcation of coordination number 3 (that is one mother 
        and two daughter branches)!
        INPUT: fqb: Fractional blood flow into the daughter branch.
               da: Diameter of daughter branch.
               db: Diameter of other daughter branch.
               df: Diameter of mother vessel.
               htd: Discharge hematocrit of the mother vessel.
        OUTPUT: fqe: Fractional red blood cell flow in the daughter vessel.
        WARNING: The minimum diameter of the parent vessel is artificially
                 constrained to 1.4 micron. Moreover, the argument of the logit
                 function is restricted to the interval [0,1].
        """
        
        cdef double sf, eps, C, A, B, X0, x, fqe
        # The empirical fit function is designed for length units of microns. 
        # Hence, we need the scaling factor to and from default units:
        sf = self._sf['um -> du']
        da, db, df = da / sf, db / sf, df / sf
        df = max(1.4, df)
        eps = 1e-10
        
        C = (1 - htd) / df
        A = -13.29 * (((da/db)**2 - 1) / ((da/db)**2 + 1)) * C
        B = 1 + 6.98 * C
        X0 = 0.964 * C
        x = (fqb - X0) / (1 - 2*X0)

        # In extreme cases, the fit function will fail, due to the argument of
        # the logit function being <=0 or >=1. Set fqe to 0 or 1 manually:
        if x <= 0. or x >= 1.:
            fqe = 0. if fqb < 0.5 else 1.
        else:
            fqe = g_math.inverse_logit(A + B * g_math.logit(x))

        return fqe

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    cpdef phase_separation_effectStep(self, double v1, double v2):
        """Returns the fraction of red blood cells entering a daughter branch,
        based on the fractional blood flow into that branch. This is an
        empirical relation, published by Pries et al. in 1990 'Blood flow in
        microvascular networks - experiments and simulation'. Note that it only
        applies to a bifurcation of coordination number 3 (that is one mother
        and two daughter branches)!
        INPUT: v1: velocity in daughter branch 1
               v2: velocity in daughter branch 2
        OUTPUT: fqe: Fractional red blood cell flow in the daughter vessel.
        WARNING: The minimum diameter of the parent vessel is artificially
                 constrained to 1.4 micron. Moreover, the argument of the logit
                 function is restricted to the interval [0,1].
        """

        cdef double fqe

        # In extreme cases, the fit function will fail, due to the argument of
        # the logit function being <=0 or >=1. Set fqe to 0 or 1 manually:
        if v1 >= v2:
            fqe = 1.
        else:
            fqe = 0.

        return fqe


    # -------------------------------------------------------------------------
    cpdef fahraeus_effect(self, double diameter, double htd, bint invivo=1):
        """Returns the fraction of tube hematocrit by discharge hematocrit 
        either in vivo or in vitro. This is an empirical relation, published by 
        Pries et al. in 1990 'Blood flow in microvascular networks - 
        experiments and simulation'. The extension to in-vivo was presented in
        a 2005 paper 'Microvascular blood viscosity in vivo and the endothelial
        surface layer'.
        INPUT: diameter: Diameter of the blood vessel.
               htd: Discharge hematocrit.
               invivo: This boolean determines whether to compute the in vivo 
                       effect (the default) or in vitro.
        OUTPUT: Fraction of tube by discharge hematocrit.
                Note that this is the 'rat blood' version of the formula. The
                diameter dependent exponents have been scaled with the cube
                root of the mean red blood cell volume (rat 55fl, human 92fl).
        """
        cdef double sf, d, tbd

        # The fit function assumes diameters in [micron]. Conversion from 
        # default units is performed as required.
        sf = self._sf['um -> du']
        
        if invivo:
            d = self.physical_vessel_diameter(diameter) / sf
        else:
            d = diameter / sf
        
        tbd = htd + (1 - htd) * (1 + 1.7 * exp(-0.415 * d) - 
                                 0.6 * exp(-0.011 * d))
        
        if invivo:
            tbd = tbd * (diameter / d)**2
            
        return tbd

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    cpdef dp2dr(self, double oldPressure, double newPressure, double oldRadius,
                double vesselComplianceParameter=2.0, double ICPressure=-1.0):
        """Returns radius change of a compliant vessel based on change in 
        pressure.
        
        Vessel compliance as described by Mandeville et al. 1999
        P-P_ic = (V/A)^b
        A = V_0 / (P_0 - P_ic)^(1/b)
        <==> R = R_0 * ((P-P_ic) / (P_0-P_ic))^(1/(2*b))
        
        
        R:    vessel radius
        V:    vessel volume = pi * radius^2 * length
        P:    vessel pressure = mean of the two node pressures
        P_ic: intra-cranial pressure = 10 mmHg (Albeck et al. 1991, 
                                                Alperin et al. 2000)
        b:    vascular compliance parameter = 2 
                  (1<=b<=3 Dunn et al. 2005, Jones et al. 2001, 2002,
                           Mandeville et al. 1999, Sheth et al. 2004b)
                  currently b=2 (following Boas et al. 2008), but this may be
                  changed to depend on vessel size.
        subscript '0' denotes the old values (i.e. at the last time step)
        
        Kong et al. have extended the above Balloon/Windkessel model with an 
        additional state variable (Kong et al. 2004, see also Zheng et al. 
        2005), however values for the additional parameters are not (yet)
        readily available.
        
        INPUT: oldPressure: Pressure at the old timestep
               newPressure: Pressure at the new timestep
               oldRadius: Radius at the old timestep
               vesselComplianceParameter: Dimensionless vessel compliance 
                                          parameter (optional, default is 2.0)
               ICPressure: Intra-cranial pressure (optional, see above for 
                           default)
        OUTPUT: Radius change [length]
        """     
        # The ICPressure value reported by Albeck and Alterin is scaled to 
        # default units:
        if ICPressure == -1.0:
            ICPressure = 10 * self._sf['mmHg -> du']
            
        return oldRadius * \
               ((newPressure-ICPressure) / (oldPressure-ICPressure))** \
               (1.0/(2.0*vesselComplianceParameter))                           
             
             
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------


    cpdef intra_cranial_pressure(self):
        """The intra-cranial pressure as reported by Albeck et al. 1991, 
        Alperin et al. 2000 is 10 mmHg.
        INPUT: None
        OUTPUT: intra-cranial pressure scaled to default units.
        """
        return 10 * self._sf['mmHg -> du']
        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    
    
    cpdef dA_by_dp(self, double oldPressure, double newPressure, double oldRadius,
                 double vesselComplianceParameter=2.0, double ICPressure=-1.0):
        """Returns the derivative cross-sectional area by pressure of a 
        compliant vessel.
        
        Vessel compliance as described by Mandeville et al. 1999
        P-P_ic = (V/A)^b
        A = V_0 / (P_0 - P_ic)^(1/b)
        <==> R = R_0 * ((P-P_ic) / (P_0-P_ic))^(1/(2*b))
        
        
        R:    vessel radius
        V:    vessel volume = pi * radius^2 * length
        P:    vessel pressure = mean of the two node pressures
        P_ic: intra-cranial pressure = 10 mmHg (Albeck et al. 1991, 
                                                Alperin et al. 2000)
        b:    vascular compliance parameter = 2 
                  (1<=b<=3 Dunn et al. 2005, Jones et al. 2001, 2002,
                           Mandeville et al. 1999, Sheth et al. 2004b)
                  currently b=2 (following Boas et al. 2008), but this may be
                  changed to depend on vessel size.
        subscript '0' denotes the old values (i.e. at the last time step)
        
        Kong et al. have extended the above Balloon/Windkessel model with an 
        additional state variable (Kong et al. 2004, see also Zheng et al. 
        2005), however values for the additional parameters are not (yet) 
        readily available.
        
        INPUT: oldPressure: Pressure at the old timestep
               newPressure: Pressure at the new timestep
               oldRadius: Radius at the old timestep
               vesselComplianceParameter: Dimensionless vessel compliance 
                                          parameter
               ICPressure: Intra-cranial pressure
        OUTPUT: Radius change [length]
        """   
        cdef double areaChangeFactor
        # The ICPressure value reported by Albeck and Alterin is scaled to 
        # default units:    
        if ICPressure == -1.0:
            ICPressure = 10 * self._sf['mmHg -> du']
            
        areaChangeFactor = ((newPressure-ICPressure) /
                            (oldPressure-ICPressure))**\
                           (1.0/vesselComplianceParameter) - 1.0
        if areaChangeFactor == 0.0:
            return 0.0
        else:  
            return M_PI * oldRadius**2.0 * areaChangeFactor / \
                   (newPressure - oldPressure)


    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------


    cpdef conductance(self, double diameter, double length, double nu_blood,
                      double alpha=1.0):
        """Returns the conductance of a blood vessel depending on its diameter 
        and length, as well as the dynamic viscosity of blood.
        INPUT: diameter: Vessel diameter.
               length: Vessel length.
               nu_blood: Dynamic blood viscosity [pressure * time].
               alpha: Factor that models the departure of the lumen from a
                      circular cross-section. Optional, default = 1.0.
        OUTPUT: Vessel conductance [volume / pressure / time] = 
                                   [length^4 time / mass].
        WARNING: The vessel conductance does not include density. 
                 I.e. pressure_difference * conductance = volume_flow
        """
        
        return alpha * M_PI * diameter**4 / (128 * nu_blood * length)

        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------


    cpdef effective_rbc_length(self, double diameter):
        """Returns the effective length of a red blood cell (RBC) within a
        blood vessel of given diameter.        
        INPUT: diameter: The diameter of the containing vessel.
        OUTPUT: The effective length of an RBC. 
        """
        # TODO: find literature values
        return 5.0 * self._sf['um -> du']
        
        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------


    cpdef rbc_volume(self, str species='rat'):
        """Returns the volume of a red blood cell (RBC)
        INPUT: None.
        OUTPUT: The volume of an RBC. 
        """
        # The red blood cell volumes differ between species. According to
        # Baskurt and coworkers (1997), the respective value for human and rat
        # erythrocytes is 89.16 and 56.51 fl respectively.
        # Note: a more recent work by Windberger et al. (2003) lists
        # considerably lower values.

        if species == 'rat':
            return 56.51 * self._sf['fL -> du']
        elif species == 'human':
            return 89.16 * self._sf['fL -> du']
        elif species == 'mouse':
            return 49.0 * self._sf['fL -> du']
        elif species == 'francesco':
            return 60.0 * self._sf['fL -> du']
        elif species == 'human2':
            return 92.0 * self._sf['fL -> du']


    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
        
       
    cpdef apparent_intrinsic_viscosity(self, double diameter=-1.0):
        """Returns the apparent intrinsic viscosity.
        This quantity actually depends on the vessel diameter, as well as on
        the blood velocity, as shown in 
        Numerical Simulation of Cell Motion in Tube Flow, Pozrikidis 2004. 
        Currently, however, a fixed value is chosen that lies within the range
        of computed values and was predicted theoretically by Einstein.
        INPUT: diameter: The diameter of the containing vessel.
        OUTPUT: The apparent intrinisc viscosity.
        """

        return 2.5
        
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------


    cpdef velocity_factor(self, double diameter, bint invivo, str kind='?', 
                          double discharge_ht=-1.0, double tube_ht=-1.0):
        """Returns the factor by which red blood cells (RBCs) are faster than
        the mean velocity.
        Hd = V Ht v F / (V v)  <==> F = Hd / Ht
        This function makes use of the empirical data provided / compiled by
        Pries and coworkers.
        INPUT: diameter: The diameter of the containing vessel.
               invivo: Boolean, whether or not to consider ESL influence.
               kind: Type of blood vessel. This may be either 'a' or 'v' for 
                     'artery' or 'vein' respectively.
               tube_ht: Tube hematocrit. Empirical value taken if not provided.
               discharge_ht: Discharge hematocrit. Empirical value take if not 
                             provided.
        OUTPUT: The factor by which the RBC speed exceeds the mean blood
                velocity.
        """
        cdef double htd, htt

        if kind == '?':
            kind = 'a'
        
        if discharge_ht != -1.0:
            htd = discharge_ht
            htt = self.discharge_to_tube_hematocrit(htd, diameter, invivo)
        elif tube_ht != -1.0:
            htt = tube_ht
            htd = self.tube_to_discharge_hematocrit(htt, diameter, invivo)
        else:
            htt = self.tube_hematocrit(diameter, kind)
            htd = self.tube_to_discharge_hematocrit(htt, diameter, invivo)

        return htd/htt

