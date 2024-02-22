"""
GNU General Public License, version 3 (GPLv3)

Copyright (C) 2023, Alec O'Dell and Maria C. Babiuc Hamilton  <babiuc@marshall.edu> 
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
"""


"""
import all necessary libraries
"""
import os,sys
from scipy.integrate import odeint
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import scipy.constants as sc
import numpy as np
import math as mt
import cmath as cm
from scipy.optimize import *
from scipy.optimize import curve_fit
from mpmath import *



"""global vars"""
BNS1tΛ, BNS2tΛ = 791, 1540

"""------------------------"""
"""------------------------"""

"""Function Bank:"""

#radial sep
def r_PN(x):
        rho0PN = 1.
        rho1PN = -1.+(1./3.)*η
        rho2PN = (1./36.)*η*(171.+4.*η)
        rho3PN = (-(24257./2520.)+(41./192.)*(np.pi)**2)*η \
            -(37./12.)*η**2+(2./81.)*η**3

        r_PN = ((rho0PN)*(np.power(x,-1))+(rho1PN)*((x)**(0)) \
                +(rho2PN)*((x)**(1))+(rho3PN)*((x)**(2)))
            
        return r_PN

#ϕTide function generic
def AnaϕTide(x,n1,n1p5,n2,n2p5,n3,d1,d1p5,d2):
    tΛ1, tΛ2 = 1.0, 1.0
    kTeff = (3./32.)*(tΛ1+tΛ2) 
    
    num = (1. + n1*x + n1p5*x**(1.5) + n2*x**2 + n2p5*x**(2.5) + n3*x**3)
    den = (1. + d1*x + d1p5*x**(1.5) + d2*x**2)
    
    ϕTide = (13./(8.*η))*(kTeff)*(x**(2.5))*(num/den)

    return (ϕTide)

#ϕTide function for BNS1
def AnaϕTide1(x,n1,n1p5,n2,n2p5,n3,d1,d1p5,d2):
    tΛ1, tΛ2 = BNS1tΛ, BNS1tΛ
    kTeff = (3./32.)*(tΛ1+tΛ2) 
    
    num = (1. + n1*x + n1p5*x**(1.5) + n2*x**2 + n2p5*x**(2.5) + n3*x**3)
    den = (1. + d1*x + d1p5*x**(1.5) + d2*x**2)
    
    ϕTide = (13./(8.*η))*(kTeff)*(x**(2.5))*(num/den)

    return (ϕTide)

#ϕTide function for BNS2
def AnaϕTide2(x,n1,n1p5,n2,n2p5,n3,d1,d1p5,d2):
    tΛ1, tΛ2 = BNS2tΛ, BNS2tΛ
    kTeff = (3./32.)*(tΛ1+tΛ2) 
    
    num = (1. + n1*x + n1p5*x**(1.5) + n2*x**2 + n2p5*x**(2.5) + n3*x**3)
    den = (1. + d1*x + d1p5*x**(1.5) + d2*x**2)
    
    ϕTide = (13./(8.*η))*(kTeff)*(x**(2.5))*(num/den)

    return (ϕTide)

# Find the best fit among the analytical tide models: PN, F1 and F2
def compare_models_residuals(y_data, y_pred1, y_pred2, y_pred3, params1, params2, params3):
    # Calculate the SSR for each set of predictions
    SSR1 = np.sum((y_data - y_pred1) ** 2)
    SSR2 = np.sum((y_data - y_pred2) ** 2)
    SSR3 = np.sum((y_data - y_pred3) ** 2)
    
    # Find the minimum SSR and choose the corresponding parameters
    min_SSR = min(SSR1, SSR2, SSR3)
    if min_SSR == SSR1:
        chosen_params = params1
        identifier = 'params1'
    elif min_SSR == SSR2:
        chosen_params = params2
        identifier = 'params2'
    else:
        chosen_params = params3
        identifier = 'params3'
    
    i#print(f"Chosen parameters ({identifier}): {chosen_model_params}")
    return chosen_params, identifier

#Compares the residuals of two fits and prints the chosen parameters
def compare_residuals(y_data1, y_pred1, y_data2, y_pred2, params1, params2):    
    # Calculate the SSR for both sets of parameters
    SSR1 = np.sum((y_data1 - y_pred1) ** 2)
    SSR2 = np.sum((y_data2 - y_pred2) ** 2)
    
    # Compare the SSRs and choose the set of parameters with the lower SSR
    if SSR1 < SSR2:
        print("Parameters from the first dataset provide a better fit.")
        chosen_params = params1
        identifier = 'params1'
    else:
        print("Parameters from the second dataset provide a better fit.")
        chosen_params = params2
        identifier = 'params2'
    
    #print(f"Chosen parameters: {chosen_params}")
    return chosen_params

# Calculates R_squared 
def calculate_r_squared(y_actual, y_predicted):
    # Calculate the total sum of squares (TSS)
    tss = np.sum((y_actual - np.mean(y_actual)) ** 2)
    # Calculate the residual sum of squares (RSS)
    rss = np.sum((y_actual - y_predicted) ** 2)
    # Calculate R^2
    r_squared = 1 - (rss / tss)

    #print(f"R² for the fit: {r_squared}")
    return r_squared


# determine the end frequency for tapering: eq. 11, https://arxiv.org/pdf/1804.02235.pdf
#merger frequency function 
def BNSωh(tΛ1,tΛ2):
    kTeff = (3./32.)*(tΛ1+tΛ2)
    
    ωh0 = 0.3586
    n1 = 3.354e-2
    n2 = 4.3153e-5
    d1 = 7.542e-2
    d2 = 2.236e-4
    
    num = (1 + n1*(kTeff) + n2*(kTeff)**2)
    den = (1 + d1*(kTeff) + d2*(kTeff)**2)
    
    ωh = ωh0*(num/den)
    
    return (ωh)

#Analytic tide amplitude function for BNS1
def AnaAmpTide(x, d,p):
    tΛ1, tΛ2 = 1.0, 1.0
    kTeff = (3./32.)*(tΛ1+tΛ2)
    
    fact = (η/21.)*(np.sqrt((np.pi)/(5.)))*kTeff
    num = (x**6)*(672.-11.*x)
    den = (1.+(d*x**p))
    
    AmpTide = fact*num/den

    return (AmpTide)

def AnaAmpTide1(x, d,p):
    tΛ1, tΛ2 = BNS1tΛ, BNS1tΛ
    kTeff = (3./32.)*(tΛ1+tΛ2)
    
    fact = (η/21.)*(np.sqrt((np.pi)/(5.)))*kTeff
    num = (x**6)*(672.-11.*x)
    den = (1.+(d*x**p))
    
    AmpTide = fact*num/den

    return (AmpTide)

#Analytic tide amplitude function for BNS2
def AnaAmpTide2(x, d,p):
    tΛ1, tΛ2 = BNS2tΛ, BNS2tΛ
    kTeff = (3./32.)*(tΛ1+tΛ2)
    
    fact = (η/21.)*(np.sqrt((np.pi)/(5.)))*kTeff
    num = (x**6)*(672.-11.*x)
    den = (1.+(d*x**p))
    
    AmpTide = fact*num/den

    return (AmpTide)

#zeroth order Luminosity 
#with factor of 1/2 to get rid of extra factor of 2 in ω_hyb
def L0hyb(v_array,ω_array):
    L0hyb=(4/15)*(mz**2)*((v_array)**2)*(((1/2)*ω_array)**4)
    return L0hyb

#Hybrid Luminosity
def LEMhyb(i):
    LEMhyb=1+((11/2)*(i**2))+(16*(i**4))+(35*(i**6))+(65*(i**8))  \
            +((217/2)*(i**10))+(168*(i**12))+(246*(i**14)) \
            +(345*(i**16))+((935/2)*(i**18))+((616)*(i**20))
    return LEMhyb


"""---------------------------------------------------"""
"""---------------------------------------------------"""
class BBHinspiral():
    
    def initialize(globalvars):
        globals().update(globalvars)
        # Schwarzschild radius
        rS = float((2.0*G*m*Ms/(c**2))/1000) 

        #initial and final radial separations in units of solar mass 
        Ri = 1000*(r0-(rA+rB))/cL #initial separation between the surfaces
        RT = 1000*(rA+rB)/cL       #radius where neutron stars touch

        """
        The light ring: this is an important quantity marking when the 
        PNN approximation breaks down and the matching to BoB happens
        It is given in terms of the Schwarzchild radius and the spin of the remnant
        Will take it's location around 2rS.
        """
        fLR = 2.21              # Light Ring factor: TUNE TO CHANGE THE MATCHING
        RLR = fLR*rS*1000/cL  # Light Ring radius

        """
        initial and final value of the frequencies and velocities from Kepler's Law 
        """
        Ω0_SI  = float((G*m*Ms/(r0*1000)**3)**(1/2))  #initial
        Ωt_SI  = float((G*m*Ms/((rA+rB)*1000)**3)**(1/2))  #when stars touch
        ΩLR_SI  = float((G*m*Ms/(2*rS*1000)**3)**(1/2))  #when stars touch
        # note that velocities do not depend on mass
        v0_SI = float((G*m*Ms/(r0*1000))**(1/2)) #initial
        vt_SI = float((G*m*Ms/((rA+rB)*1000))**(1/2)) ##when stars touch
        vLR_SI = float((G*m*Ms/(2*rS*1000))**(1/2)) ##at light ring

        #Notation for frequencies: 
        #capital letters for orbital quantities, lowercase for GW quantities
        Ω0  = cT*Ω0_SI #initial
        Ωt  = cT*Ωt_SI #when stars touch
        ΩLR = cT*ΩLR_SI #final/light ring

        """
        post Newtonian parameter/Integration Bounds (v/c)^2
        initial and at light ring
        """
        MΩ0  = m*Ω0 #float((R0)**(-3/2))   #initial
        MΩt  = m*Ωt #float((2*RNS)**(-3/2))  #when stars touch
        MΩLR = m*ΩLR #float((RLR)**(-3/2)) #final/light ring

        # rescaled radii
        R0oM = R0/m
        RioM = Ri/m
        RToM = RT/m
        RLRoM = RLR/m

        x0  = (v0_SI/c)**2   #initial
        xt  = (vt_SI/c)**2   #touch
        xLR = (vLR_SI/c)**2  #light ring

        TcSI = float((5/256)*((c**5)/(G**3)*(r0)**4/(η*((m*Ms)**3))))
        Tc = float((5/256)*((R0)**4/(η*(m**3))))
        Tt = float((5/256)*((RT)**4/(η*(m**3))))
        Tc_BBH = float((5/256)*((R0oM)**4/(η*(1**3))))
        
        dt= 0.1 #0.05 #time step
        t0=0 #initial time set at t=0

        MtPNarr=np.arange(t0,MTPNf+dt, dt, dtype=np.float64)
        
        return globals().update(locals())


    def XPN():

        def xiPN(x,MtPNarr):
            
            xi0PN = (64./5.)*η    
            
            xi0p5PN = 0.
            
            xi1PN = -(4./105.)*η*(743.+924.*η)
            
            xi1p5PN = 160.85*η
            
            xi2PN = (2.*η*(34103.+122949.*η+59472.*η**2))/2835.
            
            xi2p5PN = (-248.874-950.018*η)*η
            
            xi3PN = (1./4677750.)*η*(8951189331.+275.*(-56198689.+2045736.*(np.pi)**2)*η \
                     +36152325.*η**2-129475500.*η**3+976250880.*(np.log(x/x0)))

            
            xi3PNHT = η*(-22.2337-(109568./525.)*(np.euler_gamma) \
                       -(109568./525.)*(np.log(x**(3./2.)/x0)))
            
            xi3p5PN = (1./189.)*(np.pi)*η*(-2649.+143470.*η+147112.*η**2)
            
            xi4PN = -(388./1215.)*η*(-6191.63+29700.4*η-14837.5*η**2+(18929389./10864.)*η**3 \
                      +η**4+(np.euler_gamma)*(-1133.77+(369792./3395.)*η) \
                      +(-566.885+(20506176./3395.)*η)*(np.log(x)))
            
            xi4p5PN = (64./5.)*η*((205./6.)*(np.pi)**3*η+(np.pi)*((343801320119./745113600.) \
                       -(6848./105.)**(np.euler_gamma)-(51438847./48384.)*η+(42680611./145152.)*η**2 \
                       +(9731./1344.)*η**3-(13696./105.)*(np.log(2)))-(3424./105.)*(np.pi)*(np.log(x)))
            
            xi5PN = -(11567./4860.)*η*(-3862.58+10777.1*η-16459.*η**2+2765.87*η**3 \
                      -158.934*η**4+η**5+(np.euler_gamma)*(32.3594-628.209*η \
                      +13.3968*η**2)+(16.1797-5484.07*η-6163.97*η**2)*(np.log(x)))
            
            xi5p5PN = (98374./945.)*η*(433.679+(np.euler_gamma)*(20.4465-4.19889*η) \
                      -1411.45*η+1595.21*η**2-199.563*η**3+(np.pi)*η**4 \
                      +(10.2233-232.842*η)*(np.log(x)))
            
            xi6PN = -(155377./131220.)*η*(-140112.-1436.89*(np.euler_gamma)**2+511358.*η \
                      -361163.*η**2+82689.*η**3-7012.92*η**4+157.381*η**5 \
                      +η**6+(np.euler_gamma)*(26642.9-26708.6*η+720.722*η**2 \
                      +12.9198*η**3)+(13321.4-1436.89*(np.euler_gamma)-75688.9*η \
                      + 23085.7*η**2+26072.7*η**3)*(np.log(x))-359.221*(np.log(x))**2)
            
            xiPN = (((x)**5))*(xi0PN*((x)**0)+xi0p5PN*((x)**(1/2))+xi1PN*((x)**1) \
                             +xi1p5PN*((x)**(3/2))+xi2PN*((x)**2)+xi2p5PN*((x)**(5/2)) \
                             +(xi3PN+xi3PNHT)*((x)**3)+xi3p5PN*((x)**(7/2)) \
                             +xi4PN*((x)**4)+xi4p5PN*((x)**(9/2))+xi5PN*((x)**5) \
                             +xi5p5PN*((x)**(11/2))+xi6PN*((x)**6))    
            return (xiPN)
        

        xPN=odeint(xiPN,x0,MtPNarr,hmin=1e-32) #x(t) as a 1 column array
            
        xPNarr = np.array(xPN).flatten()
            
        return globals().update(locals())


    def vPN():
        vPNarr=xPNarr**(1/2)
        return globals().update(locals())

    def rPN():
        

        rPNarr = np.array(r_PN(xPN)).flatten()
        rdotarr=np.gradient(rPNarr,MtPNarr)

        return globals().update(locals())

    def Times():
                # find position where stars touch, and light ring
        """
        Change the domain if not found
        """
        t, tT, tLR = 0,0,0

        tT = np.abs(rPNarr - RT/m).argmin()
        #print("stars touch index",(2*RNS/rPNarr[tT]), tT)

        tLR = np.abs(rPNarr - RLR/m).argmin()
        #print("light ring index",(RLR/rPNarr[tLR]), tLR)

        tX = np.abs(xPNarr - xLR).argmin()

        numLR = tLR
        numT = tT
        numX = tX
        numL = len(MtPNarr)-1
        
        return globals().update(locals())

        
    def MΩPN():
        MΩPN = (xPN)**(3/2)
        MΩPNarr = np.array(MΩPN).flatten()
        MΩdotPNarr = np.gradient(MΩPNarr,MtPNarr)
        return globals().update(locals())

    def MΦPN():
        # Create cubic interpolators for Ω(t) to be able to integrate it.
        #https://stackoverflow.com/questions/41860451/odeint-with-multiple-parameters-time-dependent
        MΩPNinterp = interp1d(MtPNarr, MΩPNarr, bounds_error=False, fill_value="extrapolate",kind = "cubic")

        # Create the function required by the ODE
        def MΦPNdot (y, t):
            MΦPNdot = MΩPNinterp(t)
            return MΦPNdot

        MΩ0 = (xPNarr[0])**(3/2)  # Initial Condition

        # Solving the ODE
        MΦPN = odeint(MΦPNdot, MΩ0, MtPNarr) #hmin=1e-32

        MΦPNarr = np.array(MΦPN).flatten()

        return globals().update(locals())
        
        
    def Waveform():
        """
        Building the PN Amplitude and Strain: attention, this is for BBH!
        """
        reAPNarr = (1./(rPNarr)-(rdotarr**2) + (rPNarr**2)*(MΩPNarr**2))
        imAPNarr = 2.*(rPNarr)*(rdotarr)*(MΩPNarr)

        hplusPNarr = -2*η*((reAPNarr)*np.cos(2*MΦPNarr) + (imAPNarr)*np.sin(2*MΦPNarr))
        hcrosPNarr = -2*η*((reAPNarr)*np.sin(2*MΦPNarr) - (imAPNarr)*np.cos(2*MΦPNarr))

        AmpPNarr = np.sqrt(((hplusPNarr)**2)+((hcrosPNarr)**2))
        
        return globals().update(locals())
        
        
    def Orbits():
        """
        Individual orbits, must use the orbital phase
        """
        rAPNarr = (mB/m)*rPNarr
        rBPNarr = -(mA/m)*rPNarr

        rAPNxarr = rAPNarr*np.cos(MΦPNarr)
        rAPNyarr = rAPNarr*np.sin(MΦPNarr)

        rBPNxarr = rBPNarr*np.cos(MΦPNarr)
        rBPNyarr = rBPNarr*np.sin(MΦPNarr)
        return globals().update(locals())

    
    def retrievequantities():
        b=globals()
        return b

"""-----------------------------------------------------"""

class BBHmerger():
    def initialize(globalvars):
        globals().update(globalvars)
        """system parameters"""
        Q22QNM = q1 + q2*(1 - χf)**(q3) #Quality Factor, Dominant 22 mode
        MfΩ22QNM = f1 +f2*(1 - χf)**(f3) #QNM Frequency Dominant 22 Mode 
        χeff = χNS*(1 - q)/(1 + q) 
        EGW = E0 + E2*χeff**(2) + E4*χeff**(4)
        Mf = (1 - EGW - MD)
        Ω22QNM = MfΩ22QNM/(Mf)
        τ = 2*(Q22QNM/Ω22QNM) #Damping Time
        ΩQNM = Ω22QNM/2 #final frequency of fundamental QNM
        """Set initial values as the final values of inspiral stage/PN expans. at light ring
        This can be tuned"""
        A0 = 1; t0 = 0
        MΩi = MΩPNarr[numLR] #assume light ring 4 for freq of PN
        MΩdoti = MΩdotPNarr[numLR] #assume light ring 4 for angular accel at light ring freq of PN
        """Initial value for the time of overlap of models """
        ti = t0-((τ/2)*np.log((((ΩQNM**4)-(MΩi**4))/(2*τ*(MΩi**3)*MΩdoti))-1))
        #we define the time array for the BoB
        tBoBmax = 275 #250
        Nmin = int(ti/dt)
        tBoBmin = Nmin*dt
        Nmax = int((tBoBmax-tBoBmin)/dt)
        MtBoBarr = np.arange(tBoBmin, tBoBmax+dt, dt, dtype=np.float64)
        return globals().update(locals())
    
        """frequency of fundamental mode as funct of t"""
    def MΩBoB():
        kB = (((ΩQNM**4)-(MΩi**4))/(1-np.tanh((ti-t0)/τ)))
        def M_ΩBoB(t):
            M_ΩBoB = ((MΩi**4)+(kB*(np.tanh((t-t0)/τ)-np.tanh((ti-t0)/τ))))**(1/4)
            return M_ΩBoB
        MΩBoBarr = np.array(M_ΩBoB(MtBoBarr)).flatten()
        MΩdotBoBarr=np.gradient(MΩBoBarr,MtBoBarr)
        #find maximum for BoB angular acceleration
        maxMΩdotBoB = np.max(MΩdotBoBarr)
        # Get the indices of maximum element in numpy array
        tMΩdotmaxBoB = np.where(MΩdotBoBarr == maxMΩdotBoB) #location of max BoB amplitude
        return globals().update(locals()) 


    def MΦBoB(): 
        def M_ΦdotBoB (y, t):
            M_ΦdotBoB = ((MΩi**4)+(kB*(np.tanh((t-t0)/τ)-np.tanh((ti-t0)/τ))))**(1/4)
            return M_ΦdotBoB
        MΦBoBnum = odeint(M_ΦdotBoB, MΩi, MtBoBarr)
        MΦBoBnumarr = np.array(MΦBoBnum).flatten()
        return globals().update(locals())
    
    def Waveform():
        """Ang. Freq"""
        def M_ΩBoB(t):
            M_ΩBoB = ((MΩi**4)+(kB*(np.tanh((t-t0)/τ)-np.tanh((ti-t0)/τ))))**(1/4)
            return M_ΩBoB
        """The amplitude of the Weyl scalar Ψ4"""
        def Ψ4(t):
            Ψ4 = A0/np.cosh((t-t0)/τ)
            return Ψ4
        """The BoB amplitude for the dominant mode"""
        def AmpBoB(t):
            AmpBoB = ((Ψ4(t))/(M_ΩBoB(t))**2)
            return AmpBoB
        AmpBoBarr = np.array(AmpBoB(MtBoBarr)).flatten()
        hplusBoBarr = -AmpBoBarr*np.cos(2*MΦBoBnumarr)
        hcrosBoBarr = +AmpBoBarr*np.sin(2*MΦBoBnumarr)
        return globals().update(locals())
    

    def retrievequantities():
        b=globals()
        return b


"""-----------------------------------------------------"""

class BBHhybrid():
    def initialize(globalvars):
        globals().update(globalvars)
        
        """make hybrid time array"""
        #translate the time array from the PN approx/inspiral stage 
        MtarrshiftPN=np.zeros(numLR)
        p=0
        for e in MtPNarr[0:numLR]:
            MtarrshiftPN[p]=e - MtPNarr[numLR]
            p=p+1
        #translate the time array for freq. from the BoB approx/merger stage 
        MtarrshiftBoB=np.zeros(len(MtBoBarr))
        p=0
        for e in MtBoBarr:
            MtarrshiftBoB[p]= e - MtBoBarr[0]
            p=p+1
        #Building hybrid time array    
        Mtarrhyb=np.zeros(len(MtarrshiftPN)+len(MtarrshiftBoB))
        p=0
        for e in range(len(MtarrshiftPN)):
            Mtarrhyb[p]=MtarrshiftPN[e]
            p=p+1
        for e in range(0,len(MtarrshiftBoB)):
            Mtarrhyb[p]=MtarrshiftBoB[e]
            p=p+1
            
        return globals().update(locals())

    def Mϕhyb():
        # Fit the phase
        lastΦPN = MΦPNarr[numLR]
        firstΦNR = MΦBoBnumarr[0]
        ΔΦ = lastΦPN - firstΦNR
        #Building Hybrid Phase
        Mϕhybarr=np.zeros(len(Mtarrhyb))
        p=0
        for e in range(len(MtarrshiftPN)):
            Mϕhybarr[p]=2*MΦPNarr[e]
            p=p+1
        for e in range(0,len(MtarrshiftBoB)):
            Mϕhybarr[p]=2*(MΦBoBnumarr[e] + ΔΦ)
            p=p+1
        return globals().update(locals())

    def Mωhyb():
    # Fit the frequency
        lastΩPN = MΩPNarr[numLR]
        firstΩNR = MΩBoBarr[0]
        ΔΩ = lastΩPN - firstΩNR
    #Building Hybrid Angular Frequency 
        Mωhybarr=np.zeros(len(Mtarrhyb))
        p=0
        for e in range(len(MtarrshiftPN)):
            Mωhybarr[p]=2*MΩPNarr[e]
            p=p+1
        for e in range(len(MtarrshiftBoB)):
            Mωhybarr[p]=2*(MΩBoBarr[e]+ΔΩ)
            p=p+1
        return globals().update(locals())

    def Xhyb():
        #Calculate the hybrid PN parameter
        Xhybarr = (Mωhybarr/2.)**(2/3)
        return globals().update(locals())
        

    def Rhyb():
        def r_PN(x):
            rho0PN = 1.
            rho1PN = -1.+(1./3.)*η
            rho2PN = (1./36.)*η*(171.+4.*η)
            rho3PN = (-(24257./2520.)+(41./192.)*(np.pi)**2)*η \
                  -(37./12.)*η**2+(2./81.)*η**3

            r_PN = ((rho0PN)*(np.power(x,-1))+(rho1PN)*((x)**(0)) \
                    +(rho2PN)*((x)**(1))+(rho3PN)*((x)**(2)))
            
            return r_PN
        #Calculate the hybrid separation 
        Rhybarr = np.array(r_PN(Xhybarr)).flatten()
        return globals().update(locals())

    def Orbits():
        """
        Now will calculate the individual orbits for the whole evolution
        """
        rAhybarr = (mB/m)*Rhybarr
        rBhybarr = -(mA/m)*Rhybarr

        rAhybxarr = rAhybarr*np.cos(Mϕhybarr/(2))
        rAhybyarr = rAhybarr*np.sin(Mϕhybarr/(2))

        rBhybxarr = rBhybarr*np.cos(Mϕhybarr/(2))
        rBhybyarr = rBhybarr*np.sin(Mϕhybarr/(2))
        return globals().update(locals())

    def Normalized_Amp():

        #define the function for the amplitude
        def Amp(hplus,hcros):
            Amp=np.sqrt(((hplus)**2)+((hcros)**2))
            return Amp

        #find maximum for BoB amplitude
        maxAmpBoB = np.max(abs(AmpBoBarr))
        # Get the indices of maximum element in numpy array
        tAmaxBoB = np.where(AmpBoBarr == maxAmpBoB) #location of max BoB amplitude
        #form BoB norm amplitude 
        NormAmpBoBarr = AmpBoBarr/maxAmpBoB
        # find first BoB amplitude
        firstNormAmpBoB = NormAmpBoBarr[0]
        #find PN amplitude when stars touch
        lastAmpPN = AmpPNarr[numLR]
        #form PN norm amplitude
        NormAmpPNarr = AmpPNarr*firstNormAmpBoB/lastAmpPN 
        #form the normalized hybrid amplitude
        Amphybarr=np.zeros(len(Mtarrhyb))
        p=0
        for e in range(len(MtarrshiftPN)):
            Amphybarr[p]=NormAmpPNarr[e]
            p=p+1
        for e in range(0,len(MtarrshiftBoB)):
            Amphybarr[p]=NormAmpBoBarr[e]
            p=p+1
        #find maximum for BoB angular acceleration
        maxAhyb = np.max(Amphybarr)
        # Get the indices of maximum element in numpy array
        maxMtarrhyb = np.where(Amphybarr == maxAhyb) #location of max BoB amplitude
        numM = maxMtarrhyb[0][0]
        
        return globals().update(locals())

    def translated_time():
        MtarrhybM=np.zeros(len(Mtarrhyb))
        p=0
        for e in range(len(MtarrhybM)):
            MtarrhybM[p]=Mtarrhyb[e] - Mtarrhyb[numM]
            p=p+1
        numF = len(MtarrhybM)
        
        return globals().update(locals())
    
    def Waveform():
        #Finally, form the strain of the gravitational waves
        #define the functions
        def hplushyb(Amp,ϕ):
            hplushyb=-Amp*np.cos(ϕ)
            return hplushyb

        def hcroshyb(Amp,ϕ):
            hcroshyb=+Amp*np.sin(ϕ)
            return hcroshyb
        #transform them into array to plot
        hplushybarr = np.array(hplushyb(Amphybarr,Mϕhybarr)).flatten()
        hcroshybarr = np.array(hcroshyb(Amphybarr,Mϕhybarr)).flatten()
        
        return globals().update(locals())
    

    def retrievequantities():
        b=globals()
        return b

"""--------------------------------------------------------"""

class SXSBBH():
    def initialize(globalvars):
        globals().update(globalvars)

        #location of max SXS BBH amplitude
        BBHAmax = np.max(BBHAmp)
        i = 0
        for i in range(len(BBHAmp)):
            if (BBHAmp[i] == BBHAmax):
                BBHiM = i
                break
            
        #location of first SXS point, tuned to the Analytic BBH start time location        
        BBHti0 = np.abs(BBHTime - MtarrhybM[0]).argmin()
        
        #calculating the BBH frequency
        BBHω=np.gradient(BBHϕ,BBHTime)
        #location of first SXS point, tuned to the Analytic BBH start frequency location        
        BBHωi0 = np.abs(BBHω - Mωhybarr[0]).argmin()
                
        return globals().update(locals())

    def align():

        #imposing window for match at the begining of Ana BBH
        #location of first window in SXS strain, after match to the Ana BBH start frequency location
        winSXS1 = np.diff(np.sign(np.diff(BBHh22plus[BBHωi0:BBHωi0+100])))
        itSXS1 = np.where(winSXS1 == 2)[0] + 1
        iSXS1 = itSXS1[0]+1

        winSXS2 = np.diff(np.sign(np.diff(BBHh22plus[BBHωi0+100:BBHωi0+250])))
        itSXS2 = 100+np.where(winSXS2 == 2)[0] + 1
        iSXS2 = itSXS2[0]

        #location of first window in Ana BBH strain
        winPN = np.diff(np.sign(np.diff(hplusPNarr[0:2000])))
        itPN = np.where(winPN == 2)[0] + 1
        iPN = itPN[0]

        # start the BBH to fit with the given window - Ana starts at min!
        BBHwi0 = BBHωi0 + iSXS1

        # note that the final Ana time is smaller than the final BBH time. 
        #Should stop there!
        iTfinBBH = np.abs((BBHTime-BBHTime[BBHwi0]) - (MtarrhybM[-1]-MtarrhybM[0])).argmin()

        # form the BBH time array used for interpolation and calculating difference
        BBHTarr = np.zeros(len(MtarrhybM))
        p=0
        for e in range(len(MtarrhybM)):
            BBHTarr[p]=BBHTime[BBHwi0]+p*dt
            p=p+1
        
        #interpolating BBH phase
        BBHϕint = interp1d(BBHTime[BBHwi0:iTfinBBH],BBHϕ[BBHwi0:iTfinBBH], bounds_error=False, fill_value="extrapolate",kind = "cubic")
        BBHϕintarr = np.array(BBHϕint(BBHTarr)).flatten()

        # form the phase difference 
        ΔϕAnaBBH = abs((Mϕhybarr-Mϕhybarr[0]) - (BBHϕintarr-BBHϕintarr[0]))

        #interpolating BBH amplitude
        BBHAint = interp1d(BBHTime[BBHwi0:iTfinBBH],BBHAmp[BBHwi0:iTfinBBH], bounds_error=False, fill_value="extrapolate",kind = "cubic")
        BBHAintarr = np.array(BBHAint(BBHTarr)).flatten()

        # form the differences of amplitudes
        ΔAmpAnaBBH = abs((BBHAintarr-BBHAintarr[0])-(Amphybarr-Amphybarr[0]))
        # form the ratios of amplitudes
        ΔAmpAnaBBH = abs((BBHAintarr-BBHAintarr[0])-(Amphybarr-Amphybarr[0]))

        #location of last SXS point   

        BBHtF = BBHTime[-1]
        BBHiF = len(BBHTime)-1
        
        # pick the initial point for the SXS BBH array
        BBHi0 =   BBHti0 #BBHωi0 
        Δt0 = (BBHTime[BBHi0]- MtarrhybM[0])
        
        # difference in amplitude at the initial time
        BBHΔAmp = Amphybarr[0] - BBHAmp[BBHi0]
        return globals().update(locals())
        
    def MωBBH():
        MωBBHarr=np.gradient(BBHϕ[BBHi0:BBHiF],BBHTime[BBHi0:BBHiF])
        return globals().update(locals())
    
    def MωdotBBH():
        MωdotBBHarr=np.gradient(MωBBHarr,BBHTime[BBHi0:BBHiF])
        return globals().update(locals())

    def retrievequantities():
        b=globals()
        return b

"""--------------------------------------------------------"""
class BBHvsBNS():
    #funct to find frequency at reference phase
    #def dfunO2(fip1, fim1, h):
    #    return (fip1 - fim1)/(2*h)
    
    #funct to find frequency at reference phase
    #def dfunO4(fip2, fip1, fim1, fim2, h):    
    #    return (-fip2 + 8*fip1 - 8*fim1 + fim2)/(12*h)
    
    def initialize(globalvars):
        globals().update(globalvars)
        
        #correct the time arrays with reference time - where initial data error become insignificant
        TrefBBH = 282
        TrefBNS1 = 1195 #1278
        TrefBNS2 = 985 # 959
##########
        #build the fitting window for numeric BNS1/BBH
        BNS1ω=np.gradient(BNS1ϕ,BNS1Time)
        BBH1ωi0 = np.abs(BBHω - BNS1ω[TrefBNS1]).argmin()

        # determine the timesteps for 1 period
        dtBBH1 = BBHTime[BBH1ωi0+1]-BBHTime[BBH1ωi0]
        dtBNS1 = BNS1Time[TrefBNS1+1]-BNS1Time[TrefBNS1]
        PBNS1 = 2.0*np.pi/BNS1ω[TrefBNS1]
        numPBBH1 = 200 #int(PBNS1/dtBBH1)
        numPBNS1 = 500 #int(PBNS1/dtBNS1)

        #location of first min in BBH strain with respect to BNS1
        leftBBH1 = np.diff(np.sign(np.diff(BBHh22plus[BBH1ωi0-20:BBH1ωi0+100])))
        ileftBBH1 = -20 + np.where(leftBBH1 == 2)[0] + 1
        ilBBH1 = ileftBBH1[0]+1

        #location of second min in BBH strain with respect to BNS1
        rightBBH1 = np.diff(np.sign(np.diff(BBHh22plus[BBH1ωi0+100:BBH1ωi0+numPBBH1])))
        irightBBH1 = 100+np.where(rightBBH1 == 2)[0] + 1
        irBBH1 = irightBBH1[0]

        # start the BBH to fit with the given window
        BBH1wi0 = BBH1ωi0 + ilBBH1

        #location of first min in BNS1 strain
        leftBNS1 = np.diff(np.sign(np.diff(BNS1h22plus[TrefBNS1-50:TrefBNS1+100])))
        ileftBNS1 = -50 + np.where(leftBNS1 == 2)[0] + 1
        ilBNS1 = ileftBNS1[0]

        #location of second min in BNS1 strain
        rightBNS1 = np.diff(np.sign(np.diff(BNS1h22plus[TrefBNS1+100:TrefBNS1+numPBNS1])))
        irightBNS1 = 100+np.where(rightBNS1 == 2)[0] + 1
        irBNS1 = irightBNS1[0]

        # start BNS1 to fit with the given window
        BNS1wi0 = TrefBNS1 + ilBNS1

        # scale the strain for BNS1
        scalehBNS1 = (BBHh22plus[BBH1wi0])/(BNS1h22plus[BNS1wi0])
#########
        #build the fitting window for numeric BNS2/BBH
        BNS2ω=np.gradient(BNS2ϕ,BNS2Time)
        BBH2ωi0 = np.abs(BBHω - BNS2ω[TrefBNS2]).argmin()

        # determine the timesteps for 1 period
        dtBBH2 = BBHTime[BBH2ωi0+1]-BBHTime[BBH2ωi0]
        dtBNS2 = BNS2Time[TrefBNS2+1]-BNS2Time[TrefBNS2]
        PBNS2 = 2.0*np.pi/BNS2ω[TrefBNS2]
        numPBBH2 = 250 #int(PBNS2/dtBBH2)
        numPBNS2 = 450 #int(PBNS2/dtBNS2)

        #location of first min in BBH strain with respect to BNS2
        leftBBH2 = np.diff(np.sign(np.diff(BBHh22plus[BBH2ωi0:BBH2ωi0+100])))
        ileftBBH2 = np.where(leftBBH2 == 2)[0] + 1
        ilBBH2 = ileftBBH2[0]+1

        #location of second min in BBH strain with respect to BNS1
        rightBBH2 = np.diff(np.sign(np.diff(BBHh22plus[BBH2ωi0+100:BBH2ωi0+numPBBH2])))
        irightBBH2 = 100+np.where(rightBBH2 == 2)[0] + 1
        irBBH2 = irightBBH2[0]

        # start the BBH to fit with the given window
        BBH2wi0 = BBH2ωi0 + ilBBH2

        #location of first min in BNS2 strain
        leftBNS2 = np.diff(np.sign(np.diff(BNS2h22plus[TrefBNS2:TrefBNS2+100])))
        ileftBNS2 = np.where(leftBNS2 == 2)[0] + 1
        ilBNS2 = ileftBNS2[0]

        #location of second min in BNS2 strain
        rightBNS2 = np.diff(np.sign(np.diff(BNS2h22plus[TrefBNS2+100:TrefBNS2+numPBNS2])))
        irightBNS2 = 100+np.where(rightBNS2 == 2)[0] + 1
        irBNS2 = irightBNS2[0]

        # start the BBH to fit with the given window
        BNS2wi0 = TrefBNS2 + ilBNS2

        # scale the strain for BNS1
        scalehBNS2 = (BBHh22plus[BBH2wi0])/(BNS2h22plus[BNS2wi0])

#################################################################

        BBH1Δϕref = BBHϕ[BBH1wi0] - BNS1ϕ[BNS1wi0]
        BBH2Δϕref = BBHϕ[BBH2wi0] - BNS2ϕ[BNS2wi0]
        BNS1Ampref = BNS1Amp/BNS1Amp[BNS1wi0]
        BBH1Ampωref = BBHAmp/BBHAmp[BBH1wi0]
        BNS2Ampref = BNS2Amp/BNS2Amp[BNS2wi0]
        BBH2Ampωref = BBHAmp/BBHAmp[BBH2wi0]
        return globals().update(locals())
    
    def align():

################## new implementation ################        

        #location of start point for BNS1 in BBHTarr
        iTinBBH1 = np.abs(BBHTarr - BBHTime[BBH1wi0]).argmin()

        #location of end point for BNS1 in BBHTarr
        iTfinBNS1 = np.abs((BNS1Time - BNS1Time[BNS1wi0])-(BBHTarr[-1]-BBHTarr[iTinBBH1])).argmin()

        #build and interpolate the BNS1 array
        BNS1Tarr = np.zeros(len(BBHTarr[iTinBBH1:]))
        p=0
        for e in range(len(BBHTarr[iTinBBH1:])):
            BNS1Tarr[p]=BNS1Time[BNS1wi0]+p*dt
            p=p+1

        #interpolating BNS1 phase
        BNS1ϕint = interp1d(BNS1Time[BNS1wi0:iTfinBNS1],BNS1ϕ[BNS1wi0:iTfinBNS1], bounds_error=False, fill_value="extrapolate",kind = "cubic")
        BNS1ϕintarr = np.array(BNS1ϕint(BNS1Tarr)).flatten()

        #interpolating BNS1 amplitude
        BNS1Aint = interp1d(BNS1Time[BNS1wi0:iTfinBNS1],BNS1Amp[BNS1wi0:iTfinBNS1], bounds_error=False, fill_value="extrapolate",kind = "cubic")
        BNS1Aintarr = np.array(BNS1Aint(BNS1Tarr)).flatten()

        #form the phase difference for BNS1
        ΔϕBNS1BBH = ((BNS1ϕintarr-BNS1ϕintarr[0]) - (BBHϕintarr[iTinBBH1:]-BBHϕintarr[iTinBBH1]))
        ΔϕBNS1Ana = ((BNS1ϕintarr-BNS1ϕintarr[0]) - (Mϕhybarr[iTinBBH1:]-Mϕhybarr[iTinBBH1]))
        #form the differences of amplitudes for BNS1
        ΔAmpBNS1BBH = abs((BBHAintarr[iTinBBH1:]/BBHAintarr[iTinBBH1])-(BNS1Aintarr/BNS1Aintarr[0]))

        #location of start point for BNS1 in BBHTarr
        iTinBBH2 = np.abs(BBHTarr - BBHTime[BBH2wi0]).argmin()

        #location of end point for BBHTarr in BNS2
        iTfinBBH2 = np.abs((BBHTarr - BBHTarr[iTinBBH2])-(BNS2Time[-1]-BNS2Time[BNS2wi0])).argmin()
        # Find max phase (stop here?)
        iTmaxBNS2ϕ = BNS2ϕ.argmax()
        iTfinBNS2 = np.abs((BBHTarr - BBHTarr[iTinBBH2])-(BNS2Time[iTmaxBNS2ϕ]-BNS2Time[BNS2wi0])).argmin()

        #build and interpolate the BNS2 array
        BNS2Tarr = np.zeros(len(BBHTarr[iTinBBH2:iTfinBBH2]))
        p=0
        for e in range(len(BBHTarr[iTinBBH2:iTfinBBH2])):
            BNS2Tarr[p]=BNS2Time[BNS2wi0]+p*dt
            p=p+1

        #interpolating BNS2 phase
        BNS2ϕint = interp1d(BNS2Time[BNS2wi0:],BNS2ϕ[BNS2wi0:], bounds_error=False, fill_value="extrapolate",kind = "cubic")
        BNS2ϕintarr = np.array(BNS2ϕint(BNS2Tarr)).flatten()

        #interpolating BNS2 amplitude
        BNS2Aint = interp1d(BNS2Time[BNS2wi0:],BNS2Amp[BNS2wi0:], bounds_error=False, fill_value="extrapolate",kind = "cubic")
        BNS2Aintarr = np.array(BNS2Aint(BNS2Tarr)).flatten()

        #form the phase difference for BNS2
        ΔϕBNS2BBH = ((BNS2ϕintarr-BNS2ϕintarr[0]) - (BBHϕintarr[iTinBBH2:iTfinBBH2]-BBHϕintarr[iTinBBH2]))
        ΔϕBNS2Ana = ((BNS2ϕintarr-BNS2ϕintarr[0]) - (Mϕhybarr[iTinBBH2:iTfinBBH2]-Mϕhybarr[iTinBBH2]))
        #form the differences of amplitudes for BNS2
        ΔAmpBNS2BBH = abs((BBHAintarr[iTinBBH2:iTfinBBH2]/BBHAintarr[iTinBBH2])-(BNS2Aintarr/BNS2Aintarr[0]))

        # the initial time for match
        numBNS1ref = iTinBBH1 #numtBNS1ref
        numBNS2ref = iTinBBH2 #numtBNS2ref

        #scale the phase with the reference phase amplitude of BBH
        #Ana1Δϕref = Mϕhybarr[numBNS1ref] - BNS1ϕ[TrefBNS1]
        #Ana2Δϕref = Mϕhybarr[numBNS2ref] - BNS2ϕ[TrefBNS2]

        #scale the amplitudes with the reference phase amplitude of BBH
        #Amphyb1ref = Amphybarr/Amphybarr[numBNS1ref]
        #Amphyb2ref = Amphybarr/Amphybarr[numBNS2ref]        

        return globals().update(locals())

    def matchphase():
        #it's time to find the last point of the BNS time array
        # note that the final BBH time is smaller than the final BNS1 time. Should stop there!
        
        ##create a shifted time array just for convenience in plotting: 
        BNS1Tshift = np.zeros(len(BNS1Tarr))
        p=0
        for e in range(len(BNS1Tarr)):
            BNS1Tshift[p]=BNS1Tarr[e] - BNS1Tarr[0]
            p=p+1
            
        BNS2Tshift = np.zeros(len(BNS2Tarr))
        p=0
        for e in range(len(BNS2Tarr)):
            BNS2Tshift[p]=BNS2Tarr[e] - BNS2Tarr[0]
            p=p+1

        numBNS1fin = len(MtarrhybM)
        numBNS2fin = iTfinBBH2 #len(AnaBBH2Tarr)+numBNS2ref
        
        AnaBBH1ϕintarr = Mϕhybarr[numBNS1ref:numBNS1fin]
        AnaBBH2ϕintarr = Mϕhybarr[numBNS2ref:numBNS2fin]

        BBH1ϕintarr = BBHϕintarr[numBNS1ref:numBNS1fin]
        BBH2ϕintarr = BBHϕintarr[numBNS2ref:numBNS2fin]

        #ΔTrefBNS12 = (BNS2Time[TrefBNS2]-BNS1Time[TrefBNS1])
        ΔTrefBNS12 = (BBHTarr[numBNS2ref]-BBHTarr[numBNS1ref])
        ΔϕrefBNS12 = BBHϕ[BBH2wi0]-BBHϕ[BBH1wi0]
        return globals().update(locals())

    def matchAmp():
        #repeat the procedure for amplitudes
        #for BNS1 
        BBH1Ampintarr = BBHAintarr[numBNS1ref:numBNS1fin]
        BNS1Ampintarr = BNS1Aintarr
        AnaBBH1Aintarr = Amphybarr[numBNS1ref:numBNS1fin]
        
        #for BNS2 
        BBH2Ampintarr = BBHAintarr[numBNS2ref:numBNS2fin]
        BNS2Ampintarr = BNS2Aintarr
        AnaBBH2Aintarr = Amphybarr[numBNS2ref:numBNS2fin]

        ΔArefBNS12 = (BBHAmp[BBH2wi0]-BBHAmp[BBH1wi0])/BBH1Ampintarr[0]
        return globals().update(locals())
        
        
    def retrievequantities():
        b=globals()
        return b

"""--------------------------------------------------------"""
class TidalPhase():
    
    def initialize(globalvars):
        
        globals().update(globalvars)
        
        # now we can take the differences between the Amplitudes to find the tidal correction
        ΔBNS1ATideNR = ((BNS1Aintarr/BNS1Aintarr[0])-(BBHAintarr[numBNS1ref:numBNS1fin]/BBHAintarr[numBNS1ref]))/BBHAintarr[numBNS1ref:numBNS1fin]
        ΔBNS2ATideNR = ((BNS2Aintarr/BNS2Aintarr[0])-(BBHAintarr[numBNS2ref:numBNS2fin]/BBHAintarr[numBNS2ref]))/BBHAintarr[numBNS2ref:numBNS2fin]

        #find the merger location as the position where the change in amplitude is maximum
        mrgBBHindex=np.where(BBHAintarr==np.max(BBHAintarr))[0][0]
        mrgBNS1index=np.where(BNS1Aintarr==np.max(BNS1Aintarr))[0][0]
        mrgBNS2index=np.where(ΔBNS2ATideNR==np.max(ΔBNS2ATideNR))[0][0]
        """***compare the above line of code to the one above it***"""
        
        # now we can take the differences between the phases to find the tidal correction
        BNS1ϕTideNR = (ΔϕBNS1BBH/BNS1tΛ) 
        BNS1ϕTideAna = (ΔϕBNS1Ana/BNS1tΛ) 

        BNS2ϕTideNR = (ΔϕBNS2BBH/BNS2tΛ)  
        BNS2ϕTideAna = (ΔϕBNS2Ana/BNS2tΛ) 

        # find the numerical location of the merger
        BNS1ωarr=np.gradient(BNS1ϕintarr, BNS1Tarr)
        BNS2ωarr=np.gradient(BNS2ϕintarr, BNS2Tarr)

        # BNS1ωh = BNS1ωarr[mrgBNS1index]
        # BNS2ωh = BNS2ωarr[mrgBNS2index]

        # find the analytical location of the merger AnaTarray,BNS2ϕinterparr
        BNS1ωh = BNSωh(BNS1tΛ, BNS1tΛ) 
        BNS2ωh = BNSωh(BNS2tΛ, BNS2tΛ) 

        # find the merger frequency, pn param and velocity
        #Calculate the merger frequency and PN parameter
        X1mrg = (BNS1ωh/2.)**(2/3)
        X2mrg = (BNS2ωh/2.)**(2/3)
        #Calculate the merger velocity
        V1mrg = (X1mrg)**(1/2)
        V2mrg = (X2mrg)**(1/2)

        #location of estimated merger in the Xhybarr array 
        numX1mrg = np.abs(Xhybarr - X1mrg).argmin() #mrgBNS1index
        numX2mrg = np.abs(Xhybarr - X2mrg).argmin() #mrgBNS2index

        #the PN 
        n1_PN = 3115/624
        n1p5_PN = 0 #-5*(np.pi)/2
        n2_PN = 0 #28024205/1100736
        n2p5_PN = 0 #-4283*(np.pi)/312
        n3_PN = 0
        d1_PN = 0
        d1p5_PN = 0
        d2_PN = 0
        coeffs_PN=np.array([n1_PN,n1p5_PN,n2_PN,n2p5_PN,n3_PN,d1_PN,d1p5_PN,d2_PN])

        #first fit, from 1804.02235.pdf
        c1_f1 = 1817/364
        n1_f1 = -17.941
        n1p5_f1 = 57.983 
        n2_f1 = -298.876 
        n2p5_f1 = 964.192
        n3_f1 = -936.844
        d1p5_f1 = 43.446
        d1_f1 = n1_f1 - c1_f1
        d2_f1 = 0.0
        coeffs_F1=np.array([n1_f1, n1p5_f1, n2_f1, n2p5_f1, n3_f1, d1_f1, d1p5_f1, d2_f1])

        #second first fit, from 1905.06011.pdf
        c1_f2 = 3115/624
        c1p5_f2 = -5*(np.pi)/2
        c2_f2 = 28024205/1100736
        c2p5_f2 = -4283*(np.pi)/312
        n2p5_f2 = 312.48173 
        n3_f2 = -342.15498 
        d1_f2 = -20.237200 
        d2_f2 = -5.361630
        n1_f2   = c1_f2 + d1_f2
        n1p5_f2 = (c1_f2*c1p5_f2 - c2p5_f2 - c1p5_f2*d1_f2 + n2p5_f2)/c1_f2
        n2_f2   = c2_f2 + c1_f2*d1_f2 + d2_f2
        d1p5_f2 = - (c2p5_f2 + c1p5_f2*d1_f2 - n2p5_f2)/c1_f2
        coeffs_F2=np.array([n1_f2, n1p5_f2, n2_f2, n2p5_f2, n3_f2, d1_f2, d1p5_f2, d2_f2])

        return globals().update(locals())

    def firstfits():

        # BNS1
        # PN Tide, into the postmerger, BNS 1 
        BNS1ϕTidePN = AnaϕTide1(Xhybarr[numBNS1ref:numBNS1fin],*coeffs_PN)
        BNS1ϕTidePN = BNS1ϕTidePN -BNS1ϕTidePN[0]
        
        # Fit 1 Tide, into the postmerger BNS 1
        BNS1ϕTideF1 = AnaϕTide1(Xhybarr[numBNS1ref:numBNS1fin],*coeffs_F1)
        BNS1ϕTideF1 = BNS1ϕTideF1 - BNS1ϕTideF1[0]

        # Fit 2 Tide, into the postmerger BNS 1
        BNS1ϕTideF2 = AnaϕTide1(Xhybarr[numBNS1ref:numBNS1fin],*coeffs_F2)
        BNS1ϕTideF2 = BNS1ϕTideF2 - BNS1ϕTideF2[0]

        # Fit 2 Tide, check if XNRarr is better, into the postmerger BNS 1
        # check if replacing with XNRarr makes a difference
        BBHωintarr=np.gradient(BBHϕintarr,BBHTarr)
        #XNRarr = (BBHωintarr/2.)**(2/3)
        #BNS1ϕTideF2NR = AnaϕTide1(XNRarr[numBNS1ref:numBNS1fin],*coeffs_F2) - AnaϕTide1(XNRarr[numBNS1ref],*coeffs_F2)

        # BNS2
        # PN Tide, into the postmerger, BNS 2 
        BNS2ϕTidePN = AnaϕTide2(Xhybarr[numBNS2ref:numBNS2fin],*coeffs_PN)
        BNS2ϕTidePN = BNS2ϕTidePN -BNS2ϕTidePN[0]

        # Fit 1 Tide, into the postmerger BNS 2
        BNS2ϕTideF1 = AnaϕTide2(Xhybarr[numBNS2ref:numBNS2fin],*coeffs_F1)
        BNS2ϕTideF1 = BNS2ϕTideF1 -BNS2ϕTideF1[0]

        # Fit 2 Tide, into the postmerger BNS 2
        BNS2ϕTideF2 = AnaϕTide2(Xhybarr[numBNS2ref:numBNS2fin],*coeffs_F2)
        BNS2ϕTideF2 = BNS2ϕTideF2 -BNS2ϕTideF2[0]

        return globals().update(locals())

    def curvefits():
        #now do the curve fit starting from the sets of coeff's from publications (listed in initalize function in this class)
        
        maxcorrectionIDX1=np.where(BNS1ϕTideNR==np.max(BNS1ϕTideNR))[0][0]
        maxcorrectionIDX2=np.where(BNS2ϕTideNR==np.max(BNS2ϕTideNR))[0][0]

        numIDX1 = maxcorrectionIDX1
        numIDX2 = maxcorrectionIDX2

        numBNS1off = len(BNS1Tshift)-len(Xhybarr[numBNS1ref:numX1mrg])
        numBNS2off =  len(BNS2Tshift)-len(Xhybarr[numBNS2ref:numX2mrg])
 
        numBNS1max = len(BNS1Tshift)-len(Xhybarr[numBNS1ref:numIDX1])
        numBNS2max =  len(BNS2Tshift)-len(Xhybarr[numBNS2ref:numIDX2])

        #########FOR BNS1:
        #BNS1, fit up to analytic ϕ merger
        #ΔϕBNS1AnaPN0 = AnaϕTide(Xhybarr[numBNS1ref],*coeffs_PN)
        #ΔϕBNS1AnaF10 = AnaϕTide(Xhybarr[numBNS1ref],*coeffs_F1)
        #ΔϕBNS1AnaF20 = AnaϕTide(Xhybarr[numBNS1ref],*coeffs_F2)

        # New fits, fitted arrays, no tidal coefficient
        # PN model fit
        popt, pcov = curve_fit(AnaϕTide, Xhybarr[numBNS1ref:numX1mrg],BNS1ϕTideNR[0:-numBNS1off],p0=coeffs_PN, maxfev=5000, ftol=1e-14, xtol=1e-14)
        #popt, pcov = curve_fit(AnaϕTide, Xhybarr[numBNS1ref:numX1mrg],BNS1ϕTideAna[0:-numBNS1off]+ΔϕBNS1AnaPN0,p0=coeffs_PN, maxfev=5000, ftol=1e-14, xtol=1e-14)
        pBNS1fit_PN=popt
        BNS1ϕTidemrg_PN = AnaϕTide1(Xhybarr[numBNS1ref:numBNS1fin],*pBNS1fit_PN)
        BNS1ϕTidemrg_PN = BNS1ϕTidemrg_PN - BNS1ϕTidemrg_PN[0]

        # Merger fit, F1 coefficients
        popt, pcov = curve_fit(AnaϕTide,Xhybarr[numBNS1ref:numX1mrg],BNS1ϕTideNR[0:-numBNS1off],p0=coeffs_F1, maxfev=5000, ftol=1e-14, xtol=1e-14)
        #popt, pcov = curve_fit(AnaϕTide,Xhybarr[numBNS1ref:numX1mrg],BNS1ϕTideAna[0:-numBNS1off]+ΔϕBNS1AnaF10,p0=coeffs_F1, maxfev=5000, ftol=1e-14, xtol=1e-14)
        pBNS1fit_F1=popt
        BNS1ϕTidemrg_F1 = AnaϕTide1(Xhybarr[numBNS1ref:numBNS1fin],*pBNS1fit_F1)
        BNS1ϕTidemrg_F1 = BNS1ϕTidemrg_F1 - BNS1ϕTidemrg_F1[0]

        # Merger fit, F2 coefficients
        popt, pcov = curve_fit(AnaϕTide,Xhybarr[numBNS1ref:numX1mrg],BNS1ϕTideNR[0:-numBNS1off],p0=coeffs_F2, maxfev=5000, ftol=1e-14, xtol=1e-14)
        #popt, pcov = curve_fit(AnaϕTide,Xhybarr[numBNS1ref:numX1mrg],BNS1ϕTideAna[0:-numBNS1off]+ΔϕBNS1AnaF20,p0=coeffs_F2, maxfev=5000, ftol=1e-14, xtol=1e-14)
        pBNS1fit_F2=popt
        BNS1ϕTidemrg_F2 = AnaϕTide1(Xhybarr[numBNS1ref:numBNS1fin],*pBNS1fit_F2)
        BNS1ϕTidemrg_F2 = BNS1ϕTidemrg_F2 - BNS1ϕTidemrg_F2[0]

        #########FOR BNS2:
        #BNS2, fit up to ϕ merger
        #ΔϕBNS2AnaPN0 = AnaϕTide(Xhybarr[numBNS2ref],*coeffs_PN)
        #ΔϕBNS2AnaF10 = AnaϕTide(Xhybarr[numBNS2ref],*coeffs_F1)
        #ΔϕBNS2AnaF20 = AnaϕTide(Xhybarr[numBNS2ref],*coeffs_F2)

        # Merger fit, PN coefficients
        popt, pcov = curve_fit(AnaϕTide, Xhybarr[numBNS2ref:numX2mrg],BNS2ϕTideNR[0:-numBNS2off],p0=coeffs_PN, maxfev=5000, ftol=1e-14, xtol=1e-14)
        #popt, pcov = curve_fit(AnaϕTide, Xhybarr[numBNS2ref:numX2mrg],BNS2ϕTideAna[0:-numBNS2off]+ΔϕBNS2AnaPN0,p0=coeffs_PN, maxfev=5000, ftol=1e-14, xtol=1e-14)
        pBNS2fit_PN=popt
        BNS2ϕTidemrg_PN = AnaϕTide2(Xhybarr[numBNS2ref:numBNS2fin],*pBNS2fit_PN)
        BNS2ϕTidemrg_PN = BNS2ϕTidemrg_PN - BNS2ϕTidemrg_PN[0]

        # Merger fit, F1 coefficients
        popt, pcov = curve_fit(AnaϕTide,Xhybarr[numBNS2ref:numX2mrg],BNS2ϕTideNR[0:-numBNS2off],p0=coeffs_F1, maxfev=5000, ftol=1e-14, xtol=1e-14)
        #popt, pcov = curve_fit(AnaϕTide,Xhybarr[numBNS2ref:numX2mrg],BNS2ϕTideAna[0:-numBNS2off]+ΔϕBNS2AnaF10,p0=coeffs_F1, maxfev=5000, ftol=1e-14, xtol=1e-14)
        pBNS2fit_F1=popt
        BNS2ϕTidemrg_F1 = AnaϕTide2(Xhybarr[numBNS2ref:numBNS2fin],*pBNS2fit_F1)
        BNS2ϕTidemrg_F1 = BNS2ϕTidemrg_F1 - BNS2ϕTidemrg_F1[0]

        # Merger fit, F2 coefficients
        popt, pcov =curve_fit(AnaϕTide,Xhybarr[numBNS2ref:numX2mrg],BNS2ϕTideNR[0:-numBNS2off],p0=coeffs_F2, maxfev=5000, ftol=1e-14, xtol=1e-14)
        #popt, pcov =curve_fit(AnaϕTide,Xhybarr[numBNS2ref:numX2mrg],BNS2ϕTideAna[0:-numBNS2off]+ΔϕBNS2AnaF20,p0=coeffs_F2, maxfev=5000, ftol=1e-14, xtol=1e-14)
        pBNS2fit_F2=popt
        BNS2ϕTidemrg_F2 = AnaϕTide2(Xhybarr[numBNS2ref:numBNS2fin],*pBNS2fit_F2)
        BNS2ϕTidemrg_F2 = BNS2ϕTidemrg_F2 - BNS2ϕTidemrg_F2[0]

        # form time arrays for calculating best fit:
        ΔTmrg = BNS1Tshift[mrgBNS1index]-BNS2Tshift[mrgBNS2index]-ΔTrefBNS12

        iTiniBNS2BNS1 = np.abs(BNS1Tshift - ΔTrefBNS12).argmin()
        iTfinBNS2BNS1 = np.abs(BNS1Tshift - (ΔTrefBNS12+BNS2Tshift[-1])).argmin()+1

        #ΔϕBNS1 = (BNS1ϕintarr[iTiniBNS2BNS1:iTfinBNS2BNS1]-BNS1ϕintarr[iTiniBNS2BNS1]) - (BBHϕintarr[iTinBBH2:iTfinBBH2]-BBHϕintarr[iTinBBH2])
        #ΔϕBNS2 = (BNS2ϕintarr-BNS2ϕintarr[0]) - (BBHϕintarr[iTinBBH2:iTfinBBH2]-BBHϕintarr[iTinBBH2])
        #ΔϕBNS1 = BNS1ϕTideAna[iTiniBNS2BNS1:iTfinBNS2BNS1] - BNS1ϕTideAna[iTiniBNS2BNS1] 
        #ΔϕBNS2 = BNS2ϕTideAna - BNS2ϕTideAna[0]
        ΔϕBNS1 = BNS1ϕTideNR[iTiniBNS2BNS1:iTfinBNS2BNS1] - BNS1ϕTideNR[iTiniBNS2BNS1]
        ΔϕBNS2 = BNS2ϕTideNR - BNS2ϕTideAna[0]

        ΔϕBNSAnaPN = AnaϕTide(Xhybarr[numBNS2ref:numBNS2fin],*coeffs_PN)
        ΔϕBNSAnaPN = ΔϕBNSAnaPN - ΔϕBNSAnaPN[0] #no tidal coeff.
        ΔϕBNSAnaF1 = AnaϕTide(Xhybarr[numBNS2ref:numBNS2fin],*coeffs_F1)
        ΔϕBNSAnaF1 = ΔϕBNSAnaF1 - ΔϕBNSAnaF1[0] #no tidal coeff.
        ΔϕBNSAnaF2 = AnaϕTide(Xhybarr[numBNS2ref:numBNS2fin],*coeffs_F2)
        ΔϕBNSAnaF2 = ΔϕBNSAnaF2 - ΔϕBNSAnaF2[0] #no tidal coeff.

        # comparison interval arrays, BNS1
        ΔϕBNS1FitPN = AnaϕTide(Xhybarr[numBNS2ref:numBNS2fin],*pBNS1fit_PN)
        ΔϕBNS1FitPN = ΔϕBNS1FitPN - ΔϕBNS1FitPN[0] #no tidal coeff.
 
        ΔϕBNS1FitF1 = AnaϕTide(Xhybarr[numBNS2ref:numBNS2fin],*pBNS1fit_F1)
        ΔϕBNS1FitF1 = ΔϕBNS1FitF1 - ΔϕBNS1FitF1[0] # no tidal coeff.

        ΔϕBNS1FitF2 = AnaϕTide(Xhybarr[numBNS2ref:numBNS2fin],*pBNS1fit_F2)
        ΔϕBNS1FitF2 = ΔϕBNS1FitF2 - ΔϕBNS1FitF2[0] #no tidal coeff

        # comparison interval arrays, BNS2
        ΔϕBNS2FitPN = AnaϕTide(Xhybarr[numBNS2ref:numBNS2fin],*pBNS2fit_PN)
        ΔϕBNS2FitPN = ΔϕBNS2FitPN - ΔϕBNS2FitPN[0]

        ΔϕBNS2FitF1 = AnaϕTide(Xhybarr[numBNS2ref:numBNS2fin],*pBNS2fit_F1)
        ΔϕBNS2FitF1 = ΔϕBNS2FitF1 - ΔϕBNS2FitF1[0]

        ΔϕBNS2FitF2 = AnaϕTide(Xhybarr[numBNS2ref:numBNS2fin],*pBNS2fit_F2)
        ΔϕBNS2FitF2 = ΔϕBNS2FitF2 - ΔϕBNS2FitF2[0]

        # Best Fit
        BNS1ϕTidemrg = BNS1ϕTidemrg_F2
        BNS2ϕTidemrg = BNS2ϕTidemrg_F2

        # BNS1 pBNS1fit_F2 coefficients applied to BNS2
        BNS2ϕTidemrg_BNS1 = AnaϕTide2(Xhybarr[numBNS2ref:numBNS2fin],*pBNS1fit_F2)
        BNS2ϕTidemrg_BNS1 = BNS2ϕTidemrg_BNS1 - BNS2ϕTidemrg_BNS1[0]

        # form time arrays for calculating best fit:
        ΔTmrg = BNS1Tshift[mrgBNS1index]-BNS2Tshift[mrgBNS2index]-ΔTrefBNS12

        iTiniBNS2BNS1 = np.abs(BNS1Tshift - ΔTrefBNS12).argmin()
        iTfinBNS2BNS1 = np.abs(BNS1Tshift - (ΔTrefBNS12+BNS2Tshift[-1])).argmin()+1

        #ΔϕBNS1 = (BNS1ϕintarr[iTiniBNS2BNS1:iTfinBNS2BNS1]-BNS1ϕintarr[iTiniBNS2BNS1]) - (BBHϕintarr[iTinBBH2:iTfinBBH2]-BBHϕintarr[iTinBBH2])
        #ΔϕBNS2 = (BNS2ϕintarr-BNS2ϕintarr[0]) - (BBHϕintarr[iTinBBH2:iTfinBBH2]-BBHϕintarr[iTinBBH2])
        #ΔϕBNS1 = BNS1ϕTideAna[iTiniBNS2BNS1:iTfinBNS2BNS1] - BNS1ϕTideAna[iTiniBNS2BNS1] 
        #ΔϕBNS2 = BNS2ϕTideAna - BNS2ϕTideAna[0]
        ΔϕBNS1 = BNS1ϕTideNR[iTiniBNS2BNS1:iTfinBNS2BNS1] - BNS1ϕTideNR[iTiniBNS2BNS1]
        ΔϕBNS2 = BNS2ϕTideNR - BNS2ϕTideAna[0]

        ΔϕBNSAnaPN = AnaϕTide(Xhybarr[numBNS2ref:numBNS2fin],*coeffs_PN)
        ΔϕBNSAnaPN = ΔϕBNSAnaPN - ΔϕBNSAnaPN[0] #no tidal coeff.
        ΔϕBNSAnaF1 = AnaϕTide(Xhybarr[numBNS2ref:numBNS2fin],*coeffs_F1)
        ΔϕBNSAnaF1 = ΔϕBNSAnaF1 - ΔϕBNSAnaF1[0] #no tidal coeff.
        ΔϕBNSAnaF2 = AnaϕTide(Xhybarr[numBNS2ref:numBNS2fin],*coeffs_PN)
        ΔϕBNSAnaF2 = ΔϕBNSAnaF2 - ΔϕBNSAnaF2[0] #no tidal coeff.

        return globals().update(locals())
        
    def retrievequantities():
        b=globals()
        return b

"""-------------------------------------------"""
class Tapering():
    def initialize(globalvars):
        globals().update(globalvars)
        
        #the analytic phase arrays for BBH and BNS1
        #MϕBBH1hybarr = Mϕhybarr[numBNS1ref:numBNS1fin]-Mϕhybarr[numBNS1ref]
        MϕBBH1hybarr = AnaBBH1ϕintarr - AnaBBH1ϕintarr[0]
        MϕBNS1hybarr = MϕBBH1hybarr + (BNS1ϕTidemrg)
        MϕBNS1NRarr = BNS1ϕintarr-BNS1ϕintarr[0]
        
        #the analytic phase arrays for BBH and BNS2
        #MϕBBH2hybarr = Mϕhybarr[numBNS2ref:numBNS2fin]-Mϕhybarr[numBNS2ref]
        MϕBBH2hybarr = AnaBBH2ϕintarr - AnaBBH2ϕintarr[0]
        MϕBNS2hybarr = MϕBBH2hybarr + (BNS2ϕTidemrg)
        MϕBNS2NRarr = BNS2ϕintarr-BNS2ϕintarr[0]

        #the analytic phase arrays for BBH and BNS2 fit with new BNS1 coefficients
        #MϕBBH2hybarr = Mϕhybarr[numBNS2ref:numBNS2fin]-Mϕhybarr[numBNS2ref]
        MϕBNS12hybarr = MϕBBH2hybarr + (BNS2ϕTidemrg_BNS1)

        #too noisy to be of any use - attempt to taper
        MϕBNS1diff=MϕBNS1hybarr-MϕBNS1NRarr
        MϕBNS1diff_dot=np.gradient(MϕBNS1diff,BNS1Tshift) 

        #too noisy to be of any use - attempt to taper
        MϕBNS2diff=MϕBNS2hybarr-MϕBNS2NRarr
        MϕBNS2diff_dot=np.gradient(MϕBNS2diff,BNS2Tshift)

        #too noisy to be of any use - attempt to taper
        MϕBNS12diff=MϕBNS12hybarr-MϕBNS2NRarr
        MϕBNS12diff_dot=np.gradient(MϕBNS12diff,BNS2Tshift)

        return globals().update(locals())       

    def taper():

        #outputs where derivative of difference wrt time goes from negative to positive...possible cutoff points (want to find element after the "small hump around t=3000")
        cutoffs1=[]
        for i in range(len(MϕBNS1diff_dot)-1):
            if MϕBNS1diff_dot[i+1]*MϕBNS1diff_dot[i]<=0:
                cutoffs1.append(i)
        #outputs where derivative of difference wrt time goes from negative to positive...possible cutoff points (want to find element after the "small hump around t=3000")
        cutoffs2=[]
        for i in range(len(MϕBNS2diff_dot)-1):
            if MϕBNS2diff_dot[i+1]*MϕBNS2diff_dot[i]<=0:
                cutoffs2.append(i)

        cutoffs12=[]
        for i in range(len(MϕBNS12diff_dot)-1):
            if MϕBNS12diff_dot[i+1]*MϕBNS12diff_dot[i]<=0:
                cutoffs12.append(i)

        #find Tcutoff at the peak phase after merger or at t = 200M)
        iT1pm = min(numIDX1-mrgBNS1index,2000)
        iT2pm = min(numIDX2-mrgBNS2index, 2000)
        iTcutoff1 = np.abs((cutoffs1 - mrgBNS1index) - iT1pm).argmin()
        iTcutoff2 = np.abs((cutoffs2 - mrgBNS2index) - iT2pm).argmin()
        iTcutoff12 = np.abs((cutoffs12 - mrgBNS2index) - iT2pm).argmin()
        #print(cutoffs1[iTcutoff1], mrgBNS1index)
        #print(cutoffs2[iTcutoff2], mrgBNS2index)

        #generating all "corrected/tapered phase-candidates" at once and making/appending to a list 
        #where each element in the list is one of these "phase arrays"
        Listofphasearrays1=[]
        for i in range(len(cutoffs1)):
            MϕBNS1Tidal=np.zeros(len(BNS1Tshift))
            for idx in range(0,cutoffs1[i]):
                MϕBNS1Tidal[idx]=BNS1ϕTidemrg[idx]+AnaBBH1ϕintarr[idx]-(BNS1ϕTidemrg[0]+AnaBBH1ϕintarr[0])
            for idx in range(cutoffs1[i],len(BNS1Tshift)):
                MϕBNS1Tidal[idx]=AnaBBH1ϕintarr[idx]+(BNS1ϕTidemrg[cutoffs1[i]-1])-(BNS1ϕTidemrg[0]+AnaBBH1ϕintarr[0])    
            Listofphasearrays1.append( MϕBNS1Tidal)

        Listofphasearrays2=[]
        for i in range(len(cutoffs2)):
            MϕBNS2Tidal=np.zeros(len(BNS2Tshift))
            for idx in range(0,cutoffs2[i]):
                MϕBNS2Tidal[idx]=BNS2ϕTidemrg[idx]+AnaBBH2ϕintarr[idx]-(BNS2ϕTidemrg[0]+AnaBBH2ϕintarr[0])
            for idx in range(cutoffs2[i],len(BNS2Tshift)):
                MϕBNS2Tidal[idx]=AnaBBH2ϕintarr[idx]+(BNS2ϕTidemrg[cutoffs2[i]-1])-(BNS2ϕTidemrg[0]+AnaBBH2ϕintarr[0])    
            Listofphasearrays2.append(MϕBNS2Tidal)

        Listofphasearrays12=[]
        for i in range(len(cutoffs12)):
            MϕBNS12Tidal=np.zeros(len(BNS2Tshift))
            for idx in range(0,cutoffs12[i]):
                MϕBNS12Tidal[idx]=BNS2ϕTidemrg_BNS1[idx]+AnaBBH2ϕintarr[idx]-(BNS2ϕTidemrg_BNS1[0]+AnaBBH2ϕintarr[0])
            for idx in range(cutoffs12[i],len(BNS2Tshift)):
                MϕBNS12Tidal[idx]=AnaBBH2ϕintarr[idx]+(BNS2ϕTidemrg_BNS1[cutoffs2[i]-1])-(BNS2ϕTidemrg_BNS1[0]+AnaBBH2ϕintarr[0])    
            Listofphasearrays12.append(MϕBNS12Tidal)

        MϕBNS1hybarr_hvs=Listofphasearrays1[iTcutoff1]
        MϕBNS2hybarr_hvs=Listofphasearrays2[iTcutoff2]
        MϕBNS12hybarr_hvs=Listofphasearrays12[iTcutoff12]

        return globals().update(locals())
        
    def retrievequantities():
        b=globals()
        return b
"""--------------------------------------------"""

class TidalAmp():
    def initialize(globalvars):
        globals().update(globalvars)
        # now we can take the differences between the Amplitudes to find the tidal correction

        BNS1AmpTideNR = ((BNS1Aintarr/BNS1Aintarr[0])/(BBHAintarr[numBNS1ref:numBNS1fin]/BBHAintarr[numBNS1ref])) - 1.
        BNS2AmpTideNR = ((BNS2Aintarr/BNS2Aintarr[0])/(BBHAintarr[numBNS2ref:numBNS2fin]/BBHAintarr[numBNS2ref])) - 1.

        Tcutoff1 = cutoffs1[iTcutoff1]
        Tcutoff2 = cutoffs2[iTcutoff2]
        Tcutoff12 = cutoffs2[iTcutoff12]

        ###FOR BNS1:
        # the analytical approximation for merger amplitude
        kTeff1 = (3./32.)*(BNS1tΛ+BNS1tΛ)
        AnaAmrg1 = η*1.6498*(1.+((2.5603e-2)*(kTeff1))-((1.024e-5)*((kTeff1)**2))) \
                /(1+(4.7278e-2)*(kTeff1))*scalehBNS1 # must scale with the NR strain
        # the analytical approximation for merger X parameter already calculated!!
        #AnaMΩmrg1 = 0.1793*(q)**(1/2)*(1.+ (3.354e-2)*kTeff1 + (4.315e-5)*kTeff1**2) \
        #        /(1.+ (7.542e-2)*kTeff1 + (2.236e-4)*kTeff1**2)
        AnaXmrg1 = X1mrg #(AnaMΩmrg1)**(2/3)
        # consider Xmrg when the stars are touching, making first contact
        Xc1 = 2.8/RT 

        # the numerical values for merger amplitude
        Xmerge1 = (BNS1ωarr[mrgBNS1index]/2.)**(2/3)
        fact1 = BBH1Ampintarr[0]/BNS1Ampintarr[0]
        Amerge1 = BNS1Ampintarr[mrgBNS1index]*fact1

        #computing "initial value"/"initial guess" for free parameter d
        p1 = 1 #power of XpN
        dA1=(1./Xmerge1)*((AnaAmpTide1(Xmerge1,0, p1)/Amerge1)-1.)
        dAnaA1=(1./AnaXmrg1)*((AnaAmpTide1(AnaXmrg1,0, p1)/AnaAmrg1)-1.)
#        dAnaA1=(1./Xc1)*((AnaAmpTide1(Xc1,0, p1)/Xc1)-1.)

        # PN fit
        initialPNAcoeffs1 = np.array([0,p1])
        # numerical fit
        initialAnaAcoeffs1 = np.array([dAnaA1,p1])
        initialAcoeffs1 = np.array([dA1,p1])
        
        ###FOR BNS 2:
        # the analytical approximation for merger amplitude
        kTeff2 = (3./32.)*(BNS2tΛ+BNS2tΛ)
        AnaAmrg2 = η*1.6498*(1.+((2.5603e-2)*(kTeff2))-((1.024e-5)*((kTeff2)**2))) \
                /(1+(4.7278e-2)*(kTeff2))*scalehBNS2 # must scale with the NR strain
        # the analytical approximation for merger X parameter already calculated!
        #AnaMΩmrg2 = 0.1793*(q)**(1/2)*(1.+ (3.354e-2)*kTeff2 + (4.315e-5)*kTeff2**2) \
        #        /(1.+ (7.542e-2)*kTeff2 + (2.236e-4)*kTeff2**2)
        AnaXmrg2 = X2mrg #(AnaMΩmrg2)**(2/3)
        # consider Xmrg when the stars are touching, making first contact
        Xc2 = 2.7/RT 

        # the numerical values for merger amplitude
        Xmerge2 = (BNS2ωarr[mrgBNS2index]/2.)**(2/3)
        fact2 = BBH2Ampintarr[0]/BNS2Ampintarr[0]
        Amerge2=BNS2Ampintarr[mrgBNS2index]*fact2

        #computing "initial value"/"initial guess" for free parameter d
        p2 = 1 #power of XpN
        dA2=(1./Xmerge2)*((AnaAmpTide2(Xmerge2,0, p2)/Amerge2)-1.)
        dAnaA2=(1./AnaXmrg2)*((AnaAmpTide2(AnaXmrg2,0, p2)/AnaAmrg2)-1.)
#        dAnaA2=(1./Xc2)*((AnaAmpTide2(Xc2,0, p2)/Xc2)-1.)
        
        # PN fit
        initialPNAcoeffs2 = np.array([0,p2])
        # numerical fit
        initialAnaAcoeffs2 = np.array([dAnaA2,p2])
        initialAcoeffs2 = np.array([dA2,p2])

        return globals().update(locals())

    def AnaApprox():
        # Amplitude BNS1, for numerical and analytic merger
        BNS1PNATide = AnaAmpTide1(Xhybarr[numBNS1ref:numBNS1fin],*initialPNAcoeffs1)#-AnaAmpTide1(Xhybarr[numBNS1ref],*initialPNAcoeffs1)
        BNS1PNATide = BNS1PNATide - BNS1PNATide[0]
        # numerical fit amplitude, for numerical and analytic merger
        BNS1ATide = AnaAmpTide1(Xhybarr[numBNS1ref:numBNS1fin],*initialAcoeffs1) #-AnaAmpTide1(Xhybarr[numBNS1ref],*initialAcoeffs1)
        BNS1ATide = BNS1ATide - BNS1ATide[0]
        BNS1AnaATide = AnaAmpTide1(Xhybarr[numBNS1ref:numBNS1fin],*initialAnaAcoeffs1) #-AnaAmpTide1(Xhybarr[numBNS1ref],*initialAnaAcoeffs1)
        BNS1AnaATide = BNS1AnaATide - BNS1AnaATide[0]

        # Amplitude BNS2, for numerical and analytic merger
        BNS2PNATide = AnaAmpTide2(Xhybarr[numBNS2ref:numBNS1fin],*initialPNAcoeffs2) #-AnaAmpTide2(Xhybarr[numBNS2ref],*initialPNAcoeffs2)
        BNS2PNATide = BNS2PNATide - BNS2PNATide[0]
        BNS2ATide = AnaAmpTide2(Xhybarr[numBNS2ref:numBNS2fin],*initialAcoeffs2) #-AnaAmpTide2(Xhybarr[numBNS2ref],*initialAcoeffs2)
        BNS2ATide = BNS2ATide - BNS2ATide[0]
        BNS2AnaATide = AnaAmpTide2(Xhybarr[numBNS2ref:numBNS2fin],*initialAnaAcoeffs2)#-AnaAmpTide2(Xhybarr[numBNS2ref],*initialAnaAcoeffs2)
        BNS2AnaATide = BNS2AnaATide - BNS2AnaATide[0]

        return globals().update(locals())

    def curvefits():
        ###Amplitude, FOR BNS1:
        #BNS1 PN fitting
        #ΔABNS1PN = AnaAmpTide1(Xhybarr[numBNS1ref:numBNS1ref+mrgBNS1index],*initialPNAcoeffs1)
        popt, pcov =curve_fit(AnaAmpTide1,Xhybarr[numBNS1ref:numBNS1ref+mrgBNS1index],BNS1AmpTideNR[0:mrgBNS1index],p0=initialPNAcoeffs1, maxfev=5000, ftol=1e-14, xtol=1e-14)
        matchedPNAcoeffs1=popt
        BNS1PNATidefit = AnaAmpTide1(Xhybarr[numBNS1ref:numBNS1ref+Tcutoff1],*matchedPNAcoeffs1)
        BNS1PNATidefit = BNS1PNATidefit - BNS1PNATidefit[0]

        #BNS1 numeric fitting A
        #ΔABNS1A = AnaAmpTide1(Xhybarr[numBNS1ref:numBNS1ref+mrgBNS1index],*initialAcoeffs1)
        popt, pcov =curve_fit(AnaAmpTide1,Xhybarr[numBNS1ref:numBNS1ref+mrgBNS1index],BNS1AmpTideNR[0:mrgBNS1index],p0=initialAcoeffs1, maxfev=5000, ftol=1e-14, xtol=1e-14)
        matchedAcoeffs1=popt
        BNS1ATidefit = AnaAmpTide1(Xhybarr[numBNS1ref:numBNS1ref+Tcutoff1],*matchedAcoeffs1)
        BNS1ATidefit = BNS1ATidefit - BNS1ATidefit[0]

        #BNS1 numeric fitting Ana
        #ΔABNS1AnaA = AnaAmpTide1(Xhybarr[numBNS1ref:numBNS1ref+mrgBNS1index],*initialAnaAcoeffs1)
        popt, pcov =curve_fit(AnaAmpTide1,Xhybarr[numBNS1ref:numBNS1ref+mrgBNS1index],BNS1AmpTideNR[0:mrgBNS1index],p0=initialAnaAcoeffs1, maxfev=5000, ftol=1e-14, xtol=1e-14)
        matchedAnaAcoeffs1=popt
        BNS1AnaATidefit = AnaAmpTide1(Xhybarr[numBNS1ref:numBNS1ref+Tcutoff1],*matchedAnaAcoeffs1)
        BNS1AnaATidefit = BNS1AnaATidefit - BNS1AnaATidefit[0]

        #attempt to cutoff the amplitude at merger
        AmpBNS1premrg=np.zeros(Tcutoff1)
        p=0
        for i in BNS1ATidefit[0:Tcutoff1]:
            AmpBNS1premrg[p]=(1.+i)*AnaBBH1Aintarr[p]/AnaBBH1Aintarr[0]
            p=p+1

        #print("AmpBNS1premrg",len(AmpBNS1premrg), len(AnaBBH1Aintarr))

        A1_min = 0
        A1_max = AmpBNS1premrg[mrgBNS1index]
        Aclip1 = np.clip(AmpBNS1premrg[mrgBNS1index:Tcutoff1], A1_min,A1_max)

        # could we apply a hanning tapering window?
        AmpBNS1pm=np.zeros(mrgBNS1index)
        p=0
        for i in BNS1ATidefit[0:mrgBNS1index]:
            AmpBNS1pm[p]=(1.+i)*AnaBBH1Aintarr[p]
            p=p+1
        #AmpBNS1pm = (1.+BNS1ATidefit[0:mrgBNS1index])*AnaBBH1Aintarr[0:mrgBNS1index]/AnaBBH1Aintarr[0]

        AmpBNS1han = AmpBNS1pm[-1]*np.hanning(2*(Tcutoff1-(mrgBNS1index+1)))
        halfPoint1 = np.argmax(AmpBNS1han)
        AmpBNS1han = AmpBNS1han[halfPoint1:]

        ###For BNS2:
        #BNS2 PN fitting up to merger
        #ΔABNS2PN = AnaAmpTide2(Xhybarr[numBNS2ref:numBNS2ref+mrgBNS2index],*initialPNAcoeffs2)
        popt, pcov =curve_fit(AnaAmpTide2,Xhybarr[numBNS2ref:numBNS2ref+mrgBNS2index],BNS2AmpTideNR[0:mrgBNS2index],p0=initialPNAcoeffs2, maxfev=5000, ftol=1e-14, xtol=1e-14)
        matchedPNAcoeffs2=popt
        BNS2PNATidefit = AnaAmpTide2(Xhybarr[numBNS2ref:numBNS2ref+Tcutoff2],*matchedPNAcoeffs2)
        BNS2PNATidefit = BNS2PNATidefit - BNS2PNATidefit[0]

        #BNS2 numeric fitting A
        #ΔABNS2A = AnaAmpTide2(Xhybarr[numBNS2ref:numBNS2ref+mrgBNS2index],*initialAcoeffs2)
        popt, pcov =curve_fit(AnaAmpTide2,Xhybarr[numBNS2ref:numBNS2ref+mrgBNS2index],BNS2AmpTideNR[0:mrgBNS2index],p0=initialAcoeffs2, maxfev=5000, ftol=1e-14, xtol=1e-14)
        matchedAcoeffs2=popt
        BNS2ATidefit = AnaAmpTide2(Xhybarr[numBNS2ref:numBNS2ref+Tcutoff2],*matchedAcoeffs2)
        BNS2ATidefit = BNS2ATidefit - BNS2ATidefit[0]

        #BNS2 numeric fitting Ana
        #ΔABNS2AnaA = AnaAmpTide2(Xhybarr[numBNS2ref:numBNS2ref+mrgBNS2index],*initialAnaAcoeffs2)
        popt, pcov =curve_fit(AnaAmpTide2,Xhybarr[numBNS2ref:numBNS2ref+mrgBNS2index],BNS2AmpTideNR[0:mrgBNS2index],p0=initialAnaAcoeffs2, maxfev=5000, ftol=1e-14, xtol=1e-14)
        matchedAnaAcoeffs2=popt
        BNS2AnaATidefit = AnaAmpTide2(Xhybarr[numBNS2ref:numBNS2ref+Tcutoff2],*matchedAnaAcoeffs2)
        BNS2AnaATidefit = BNS2AnaATidefit - BNS2AnaATidefit[0]

        #attempt to cutoff the amplitude at merger
        # numeric fit
        AmpBNS2premrg=np.zeros(Tcutoff2)
        p=0
        for i in BNS2ATidefit[0:Tcutoff2]:
            AmpBNS2premrg[p]=(1.+i)*AnaBBH2Aintarr[p]/AnaBBH2Aintarr[0]
            p=p+1

        A2_min = 0
        A2_max = AmpBNS2premrg[mrgBNS2index]
        Aclip2 = np.clip(AmpBNS2premrg[mrgBNS2index:Tcutoff2], A2_min,A2_max)

        # could we apply a hanning tapering window?
        AmpBNS2pm=np.zeros(mrgBNS2index)
        p=0
        for i in BNS2ATidefit[0:mrgBNS2index]:
            AmpBNS2pm[p]=(1.+i)*AnaBBH2Aintarr[p]
            p=p+1
        #AmpBNS2pm = (1.+BNS2ATidefit[0:mrgBNS2index])*AnaBBH2Aintarr[0:mrgBNS2index]/AnaBBH2Aintarr[0]

        AmpBNS2han = AmpBNS2pm[-1]*np.hanning(2*(Tcutoff2-(mrgBNS2index+1)))
        halfPoint2 = np.argmax(AmpBNS2han)
        AmpBNS2han = AmpBNS2han[halfPoint2:]

        # comparison interval arrays, BNS1 amplitude
        ΔABNS1 = BNS1AmpTideNR[iTiniBNS2BNS1:iTfinBNS2BNS1] - BNS1AmpTideNR[iTiniBNS2BNS1]
        ΔABNS1A = BNS1ATide[iTiniBNS2BNS1:iTfinBNS2BNS1] - BNS1ATide[iTiniBNS2BNS1]

        ΔABNS1FitPN = AnaAmpTide1(Xhybarr[numBNS2ref:numBNS2fin],*matchedPNAcoeffs1)
        ΔABNS1FitPN = ΔABNS1FitPN - ΔABNS1FitPN[0]

        ΔABNS1FitA = AnaAmpTide1(Xhybarr[numBNS2ref:numBNS2fin],*matchedAcoeffs1)
        ΔABNS1FitA = ΔABNS1FitA - ΔABNS1FitA[0]

        ΔABNS1FitAna = AnaAmpTide1(Xhybarr[numBNS2ref:numBNS2fin],*matchedAnaAcoeffs1)
        ΔABNS1FitAna = ΔABNS1FitAna - ΔABNS1FitAna[0]

        # comparison interval arrays, BNS2 amplitude
        ΔABNS2 = BNS2AmpTideNR - BNS2AmpTideNR[0]
        ΔABNS2A = BNS2ATide - BNS2ATide[0]

        ΔABNS2FitPN = AnaAmpTide2(Xhybarr[numBNS2ref:numBNS2fin],*matchedPNAcoeffs2)
        ΔABNS2FitPN = ΔABNS2FitPN - ΔABNS2FitPN[0]

        ΔABNS2FitA = AnaAmpTide2(Xhybarr[numBNS2ref:numBNS2fin],*matchedAcoeffs2)
        ΔABNS2FitA = ΔABNS2FitA - ΔABNS2FitA[0]

        ΔABNS2FitAna = AnaAmpTide2(Xhybarr[numBNS2ref:numBNS2fin],*matchedAnaAcoeffs2)
        ΔABNS2FitAna = ΔABNS2FitAna - ΔABNS2FitAna[0]

        return globals().update(locals())
        
    def retrievequantities():
        b=globals()
        return b
        
class ReconstructVars():
    # reconstruct all the variables up to the cutoff time
    def initialize(globalvars):
        globals().update(globalvars)
        Tcutoff1 = cutoffs1[iTcutoff1]
        Tcutoff2 = cutoffs2[iTcutoff2]
        # the new time arrays
        BNS1Tnew = BNS1Tshift[0:Tcutoff1]
        BNS2Tnew = BNS2Tshift[0:Tcutoff2]
        # reconstructed Mϕ for BNS1/NR
        MϕBNS1new = MϕBNS1hybarr_hvs[0:Tcutoff1] - MϕBNS1hybarr_hvs[0]
        MϕBNS1NRnew = MϕBNS1NRarr[0:Tcutoff1] - MϕBNS1NRarr[0]
        # reconstructed Mϕ for BNS2/NR
        iΔTrefBNS12 = int(ΔTrefBNS12/dt) 
        #ΔϕBNS2BNS1 = AnaBBH1ϕintarr[iΔTrefBNS12]-AnaBBH1ϕintarr[0]
        MϕBNS2new = MϕBNS2hybarr_hvs[0:Tcutoff2]-MϕBNS2hybarr_hvs[0] #+ΔϕBNS2BNS1
        MϕBNS2NRnew = MϕBNS2NRarr[0:Tcutoff2] - MϕBNS2NRarr[0] #+ΔϕBNS2BNS1
        ##calculate the angular frequencies for BNS1
        MωBNS1new = np.gradient(MϕBNS1new, BNS1Tnew)
        MωBNS1NRnew  = np.gradient(MϕBNS1NRnew, BNS1Tnew)
        ##calculate the angular frequencies for BNS2
        MωBNS2new = np.gradient(MϕBNS2new, BNS2Tnew)
        MωBNS2NRnew  = np.gradient(MϕBNS2NRnew, BNS2Tnew)
        # calculate the PN parameter for BNS1
        XBNS1new = np.sign(MωBNS1new/2.)*(np.abs(MωBNS1new/2.))**(2/3)
        XBNS1NRnew  = np.sign(MωBNS1NRnew/2.)*(np.abs(MωBNS1NRnew/2.))**(2/3)
        # calculate the PN parameter for BNS2
        XBNS2new = np.sign(MωBNS2new/2.)*(np.abs(MωBNS2new/2.))**(2/3)
        XBNS2NRnew  = np.sign(MωBNS2NRnew/2.)*(np.abs(MωBNS2NRnew/2.))**(2/3)
        #calculate the orbital velocity for BNS1
        VBNS1new = np.sign(XBNS1new)*(np.abs(XBNS1new))**(1/2)
        VBNS1NRnew  = np.sign(XBNS1NRnew)*(np.abs(XBNS1NRnew))**(1/2)
        #calculate the orbital velocity for BNS2
        VBNS2new = np.sign(XBNS2new)*(np.abs(XBNS2new))**(1/2)
        VBNS2NRnew  = np.sign(XBNS2NRnew)*(np.abs(XBNS2NRnew))**(1/2)
        # calculate the Orbital Sep for BNS1
        RBNS1new = np.array(r_PN(XBNS1new)).flatten()
        RBNS1NRnew  = np.array(r_PN(XBNS1NRnew)).flatten()
        # calculate the Orbital Sep for BNS2
        RBNS2new = np.array(r_PN(XBNS2new)).flatten()
        RBNS2NRnew  = np.array(r_PN(XBNS2NRnew)).flatten()
        #calculate the radial velocity for BNS1
        VRBNS1new = np.gradient(RBNS1new, BNS1Tnew)
        VRBNS1NRnew  = np.gradient(RBNS1NRnew, BNS1Tnew)
        #calculate the radial velocity for BNS2
        VRBNS2new = np.gradient(RBNS2new, BNS2Tnew)
        VRBNS2NRnew  = np.gradient(RBNS2NRnew, BNS2Tnew)
        #position of first star in BNS1/NR:
        rABNS1new = (mB/m)*RBNS1new 
        rABNS1NRnew  = (mB/m)*RBNS1NRnew 
        #position of second star in BNS1/NR:
        rBBNS1new = -(mA/m)*RBNS1new 
        rBBNS1NRnew  = -(mA/m)*RBNS1NRnew 
        #orbital velocity of first star BNS1/NR
        vABNS1new = (mB/m)*VBNS1new 
        vABNS1NRnew  = (mB/m)*VBNS1NRnew 
        #orbital velocity of second star BNS1/NR
        vBBNS1new = -(mA/m)*VBNS1new 
        vBBNS1NRnew  = -(mA/m)*VBNS1NRnew 
        #radial velocity of first star BNS1/NR
        vARBNS1new = (mB/m)*VRBNS1new 
        vARBNS1NRnew  = (mB/m)*VRBNS1NRnew 
        #radial velocity of second star BNS1/NR
        vBRBNS1new = -(mA/m)*VRBNS1new 
        vBRBNS1NRnew  = -(mA/m)*VRBNS1NRnew 
        #position of first star in BNS2/NR:
        rABNS2new = (mB/m)*RBNS2new 
        rABNS2NRnew  = (mB/m)*RBNS2NRnew 
        #position of second star in BNS2/NR:
        rBBNS2new = -(mA/m)*RBNS2new 
        rBBNS2NRnew  = -(mA/m)*RBNS2NRnew 
        #orbital velocity of first star BNS2/NR
        vABNS2new = (mB/m)*VBNS2new 
        vABNS2NRnew  = (mB/m)*VBNS2NRnew 
        #orbital velocity of second star BNS2/NR
        vBBNS2new = -(mA/m)*VBNS2new 
        vBBNS2NRnew  = -(mA/m)*VBNS2NRnew 
        #radial velocity of first star BNS2/NR
        vARBNS2new = (mB/m)*VRBNS2new 
        vARBNS2NRnew  = (mB/m)*VRBNS2NRnew 
        #radial velocity of second star BNS2/NR
        vBRBNS2new = -(mA/m)*VRBNS2new 
        vBRBNS2NRnew  = -(mA/m)*VRBNS2NRnew 
        # reconstructed Amp and Mϕ BNS1/BNS2
        AmpBNS1new = np.concatenate((AmpBNS1pm[0:mrgBNS1index], AmpBNS1han))
        AmpBNS2new = np.concatenate((AmpBNS2pm[0:mrgBNS2index], AmpBNS2han))
        AmpBNS1NRnew = fact1*BNS1Ampintarr[0:Tcutoff1] 
        AmpBNS2NRnew = fact2*BNS2Ampintarr[0:Tcutoff2]

        # calculate all reconstructed quantities for hybrid BBH
        MωBBH1hybarr = np.gradient(MϕBBH1hybarr, BNS1Tshift)
        MωBBH2hybarr = np.gradient(MϕBBH2hybarr, BNS2Tshift)
        XBBH1hybarr = np.sign(MωBBH1hybarr/2.)*(np.abs(MωBBH1hybarr/2.))**(2/3)
        XBBH2hybarr = np.sign(MωBBH2hybarr/2.)*(np.abs(MωBBH2hybarr/2.))**(2/3)
        VBBH1hybarr = np.sign(XBBH1hybarr)*(np.abs(XBBH1hybarr))**(1/2)
        VBBH2hybarr = np.sign(XBBH2hybarr)*(np.abs(XBBH2hybarr))**(1/2)
        RBBH1hybarr = np.array(r_PN(XBBH1hybarr)).flatten()
        RBBH2hybarr = np.array(r_PN(XBBH2hybarr)).flatten()
        VRBBH1hybarr = np.gradient(RBBH1hybarr, BNS1Tshift)
        VRBBH2hybarr = np.gradient(RBBH2hybarr, BNS2Tshift)
        rABBH1hybarr = (mB/m)*RBBH1hybarr 
        rBBBH1hybarr = -(mA/m)*RBBH1hybarr 
        rABBH2hybarr = (mB/m)*RBBH2hybarr 
        rBBBH2hybarr = -(mA/m)*RBBH2hybarr 
        vABBH1hybarr = (mB/m)*VBBH1hybarr 
        vBBBH1hybarr = -(mA/m)*VBBH1hybarr 
        vABBH2hybarr = (mB/m)*VBBH2hybarr 
        vBBBH2hybarr = -(mA/m)*VBBH2hybarr 
        vARBBH1hybarr = (mB/m)*VRBBH1hybarr 
        vBRBBH1hybarr = -(mA/m)*VRBBH1hybarr 
        vARBBH2hybarr = (mB/m)*VRBBH2hybarr 
        vBRBBH2hybarr = -(mA/m)*VRBBH2hybarr 

        return globals().update(locals())

    def retrievequantities():
        b=globals()
        return b
